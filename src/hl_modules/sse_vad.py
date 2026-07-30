"""
High-level training module for joint separation → VAD fine-tuning.

Pipeline
--------
For each training batch:
    With probability `clean_audio_prob`:
        clean_gt_audio  →  VAD model  →  VAD loss
    With probability `1 - clean_audio_prob`:
        noisy_mixture  →  frozen separation model  →  VAD model  →  VAD loss

Only the VAD model parameters are updated; the separation model is frozen.

State-dict conventions
----------------------
sep_ckpt  : checkpoint produced by sse.Module.dump_state()  (key "model")
vad_ckpt  : either a Silero JIT file (*.jit) or a plain PyTorch state-dict
            saved by torch.save({ "model": vad_model.state_dict() })
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils as utils
from src.metrics.metrics import Metrics
from src.models.customized_silero_vad.silero_vad_model import SileroVAD


class Module(object):
    import lightning_fabric as L

    VAD_CHUNK_SIZE = 512    # samples at 16 kHz (32 ms)
    VAD_SR         = 16000

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self,
                 fabric,
                 sr: int,
                 # Separation model
                 sep_model: str,
                 sep_model_params: dict,
                 sep_model_ckpt: str,
                 # VAD model
                 vad_model_ckpt: str,          # JIT path or state-dict path
                 vad_sample_rate: int = 16000,
                 # Training knobs
                 clean_audio_prob: float = 0.5,
                 # Optimizer
                 optimizer: str = "torch.optim.AdamW",
                 optimizer_params: dict = None,
                 scheduler=None, scheduler_params=None,
                 # Loss
                 loss: str = "src.losses.VADLoss.VADLoss",
                 loss_params: dict = None,
                 # Misc
                 metrics=None,
                 grad_clip=None,
                 use_compile=False):

        self.fabric = fabric
        self.sr     = sr
        self.clean_audio_prob = clean_audio_prob

        # ---- Separation model (frozen) ----
        self.sep_model = utils.import_attr(sep_model)(**sep_model_params)
        sep_state = fabric.load(sep_model_ckpt)
        if 'model' in sep_state:
            sep_state = sep_state['model']
        # sep_state may be a lightning-fabric model wrapper; unwrap if needed
        if hasattr(sep_state, 'state_dict'):
            sep_state = sep_state.state_dict()
        self.sep_model.load_state_dict(sep_state, strict=True)
        for p in self.sep_model.parameters():
            p.requires_grad_(False)
        self.sep_model.eval()

        # ---- VAD model (trainable) ----
        if vad_model_ckpt.endswith('.jit'):
            self.vad_model = SileroVAD.from_jit(vad_model_ckpt,
                                                 sample_rate=vad_sample_rate)
        else:
            self.vad_model = SileroVAD(sample_rate=vad_sample_rate)
            ckpt = torch.load(vad_model_ckpt, map_location='cpu')
            sd   = ckpt.get('model', ckpt)
            if hasattr(sd, 'state_dict'):
                sd = sd.state_dict()
            self.vad_model.load_state_dict(sd, strict=True)

        # Alias expected by train.py: fabric.setup(hl_module.model, hl_module.optimizer)
        self.model = self.vad_model

        # ---- Metrics ----
        metrics = metrics or []
        self.metrics      = [Metrics(m) for m in metrics]
        self.metric_values = {}
        self.monitor       = 'val/loss'
        self.monitor_mode  = 'min'
        self.mode          = None
        self.epoch         = 0

        # ---- Loss ----
        loss_params = loss_params or {}
        self.loss_fn = utils.import_attr(loss)(**loss_params)

        # ---- Optimizer (VAD parameters only) ----
        optimizer_params = optimizer_params or {'lr': 1e-4}
        self.optimizer  = utils.import_attr(optimizer)(
            self.vad_model.parameters(), **optimizer_params)
        self.grad_clip  = grad_clip

        if grad_clip is not None:
            print(f"USING GRAD CLIP: {grad_clip}")

        # ---- Scheduler ----
        self.scheduler = self._init_scheduler(scheduler, scheduler_params)

    # ------------------------------------------------------------------
    # VAD forward helpers
    # ------------------------------------------------------------------

    def _run_vad(self, audio: torch.Tensor) -> torch.Tensor:
        """[B, T] → [B, n_chunks].  Delegates to SileroVAD.forward_sequence."""
        return 

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step(self, batch, batch_idx, step='train'):
        inputs, targets = batch
        mixture = inputs['mixture']        # [B, C, T]
        tgt     = targets['target']        # [B, C, T]
        vad_gt  = targets['vad_gt']        # [B, n_chunks] — chunk-level labels from dataset
        batch_size = mixture.shape[0]

        # Move sep model to same device as data the first time it runs
        data_device = mixture.device
        if next(self.sep_model.parameters()).device != data_device:
            self.sep_model = self.sep_model.to(data_device)

        # Choose audio source for VAD input
        if step == 'train' and torch.rand(1).item() < self.clean_audio_prob:
            vad_input = tgt[:, 0, :]      # [B, T]
        else:
            with torch.no_grad():
                sep_out   = self.sep_model({'local_audio': mixture})
                separated = sep_out['output']          # [B, C_out, T]
            separated = separated.detach()
            vad_input = separated[:, 0, :]             # [B, T]

        # Run VAD — dataset already provides chunk-level vad_gt, no downsampling needed
        probs = self.model(vad_input, sr=self.VAD_SR)  # [B, n_chunks]

        # Align lengths (guard against off-by-one at clip boundaries)
        n = min(probs.shape[1], vad_gt.shape[1])
        loss = self.loss_fn(probs[:, :n], vad_gt[:, :n].float())

        # print(loss, probs.shape, vad_gt.shape, targets["num_target_speakers"])

        with torch.no_grad():
            self.log_metric(f'{step}/loss', loss.item(),
                            batch_size=batch_size, on_step=(step == 'train'),
                            on_epoch=True)

            # Accuracy (hard threshold at 0.5)
            pred_hard = (probs[:, :n] >= 0.5).float()
            gt_hard   = (vad_gt[:, :n] >= 0.5).float()
            acc = (pred_hard == gt_hard).float().mean()
            self.log_metric(f'{step}/vad_acc', acc.item(), batch_size=batch_size,
                            on_step=False, on_epoch=True)

        return loss, batch_size

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def train(self):
        self.model.train()
        self.sep_model.eval()
        self.mode = 'train'

    def eval(self):
        self.model.eval()
        self.mode = 'val'

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step='val')

    def reset_grad(self):
        self.optimizer.zero_grad()

    def backprop(self):
        if self.grad_clip is not None:
            total_norm = self.fabric.clip_gradients(
                self.model, self.optimizer, max_norm=self.grad_clip,
                error_if_nonfinite=False)
            if not torch.isfinite(total_norm):
                print(f"WARNING: non-finite gradient norm ({total_norm:.4f}), skipping step")
                self.optimizer.zero_grad()
                return
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def load_state(self, path):
        state = {
            'model': self.model,
            'sep_model': self.sep_model,
            'optimizer': self.optimizer,
            'current_epoch': self.epoch,
            'metric_values': self.metric_values,
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler
        self.fabric.load(path, state)
        self.epoch         = state['current_epoch']
        self.metric_values = state['metric_values']

    def dump_state(self, path):
        state = dict(
            model         = self.model,
            sep_model     = self.sep_model,
            optimizer     = self.optimizer,
            current_epoch = self.epoch,
            metric_values = self.metric_values,
        )
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler
        self.fabric.save(path, state)

    # ------------------------------------------------------------------
    # Metric helpers (same pattern as sse.Module)
    # ------------------------------------------------------------------

    def get_current_lr(self):
        for pg in self.optimizer.param_groups:
            return pg['lr']

    def log_metric(self, name, value, batch_size=1,
                   on_step=False, on_epoch=True,
                   prog_bar=True, sync_dist=True):
        epoch_str = self.epoch
        if epoch_str not in self.metric_values:
            self.metric_values[epoch_str] = {}
        if name not in self.metric_values[epoch_str]:
            self.metric_values[epoch_str][name] = dict(step=None, epoch=None)

        if isinstance(value, torch.Tensor):
            value = value.item()

        if on_step:
            if self.metric_values[epoch_str][name]['step'] is None:
                self.metric_values[epoch_str][name]['step'] = []
            self.metric_values[epoch_str][name]['step'].append(value)

        if on_epoch:
            if self.metric_values[epoch_str][name]['epoch'] is None:
                self.metric_values[epoch_str][name]['epoch'] = 0
                self.metric_values[epoch_str][name]['num_elements'] = 0
            self.metric_values[epoch_str][name]['epoch'] += value * batch_size
            self.metric_values[epoch_str][name]['num_elements'] += batch_size

    def get_avg_metric_at_epoch(self, metric, epoch=None):
        if epoch is None:
            epoch = self.epoch
        mv = self.metric_values[epoch][metric]
        return mv['epoch'] / mv['num_elements']

    def gather_metrics(self):
        for metric in self.metric_values[self.epoch]:
            vals = self.fabric.all_gather(self.metric_values[self.epoch][metric]['epoch'])
            nums = self.fabric.all_gather(self.metric_values[self.epoch][metric]['num_elements'])
            self.metric_values[self.epoch][metric]['epoch']        = torch.sum(vals)
            self.metric_values[self.epoch][metric]['num_elements'] = torch.sum(nums)

    def on_epoch_start(self):
        pass

    def on_epoch_end(self, best_path, wandb_run=None):
        assert self.epoch + 1 == len(self.metric_values)
        self.gather_metrics()

        monitor_last = self.get_avg_metric_at_epoch(self.monitor)

        save = True
        for epoch in range(len(self.metric_values) - 1):
            prev = self.get_avg_metric_at_epoch(self.monitor, epoch)
            if self.monitor_mode == 'min' and monitor_last > prev:
                save = False
                break
            if self.monitor_mode == 'max' and monitor_last < prev:
                save = False
                break

        if save and self.fabric.global_rank == 0:
            print("Best checkpoint — saving...")
            self.dump_state(best_path)

        if self.fabric.global_rank == 0:
            val_loss = self.get_avg_metric_at_epoch('val/loss')
            val_acc  = self.get_avg_metric_at_epoch('val/vad_acc')
            print(f'Val loss: {val_loss:.4f}   Val acc: {val_acc:.4f}')

        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics=monitor_last)
            else:
                self.scheduler.step()

        self.epoch += 1

    # ------------------------------------------------------------------
    # Scheduler init (same as sse.Module)
    # ------------------------------------------------------------------

    def _init_scheduler(self, scheduler, scheduler_params):
        if scheduler is None:
            return None
        if scheduler == 'sequential':
            schedulers, milestones = [], []
            for sp in scheduler_params:
                s = utils.import_attr(sp['name'])(self.optimizer, **sp['params'])
                schedulers.append(s)
                milestones.append(sp['epochs'])
            for i in range(1, len(milestones)):
                milestones[i] += milestones[i - 1]
            milestones.pop()
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers, milestones)
        return utils.import_attr(scheduler)(self.optimizer, **scheduler_params)
