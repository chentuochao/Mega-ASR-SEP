import os

import torch
import torch.nn as nn
import torch.optim as optim
# import wandb
import torch
from numpy import mean
from src.metrics.metrics import Metrics
import src.utils as utils


class Module(object):
    import lightning_fabric as L
    
    def __init__(self, 
                 fabric: L.Fabric, # Required fabric object
                 sr, 
                 model, model_params,
                 optimizer, optimizer_params,
                 scheduler=None, scheduler_params=None,
                 loss=None, loss_params=None, 
                 metrics=[], init_ckpt=None,
                 grad_clip = None, use_dp=True,
                 samples_per_speaker_number=5, use_compile=False):
        
        self.fabric = fabric

        self.model = utils.import_attr(model)(**model_params)
        if use_compile:
            print("Compiling...")
            self.model.compile()
            print("Compile finished")
        
        self.sr = sr

        # Log a val sample every this many intervals
        self.samples_per_speaker_number = samples_per_speaker_number
        
        # Initialize metrics
        self.metrics = [Metrics(metric) for metric in metrics]

        # Metric values
        self.metric_values = {}
        
        # Dataset statistics
        self.statistics = {}
       
        # Assine metric to monitor, and how to judge different models based on it
        # i.e. How do we define the best model (Here, we minimize val loss)
        self.monitor = 'val/loss'
        self.monitor_mode = 'min'

        # Mode, either train or val
        self.mode = None

        self.val_samples = {}
        self.train_samples = {}

        self.input_snr_calculated = False
        self.input_snr = []
        self.snr_metric = Metrics("snr")

        # Initialize loss function
        self.loss_fn = utils.import_attr(loss)(**loss_params)
        
        # Initaize weights if checkpoint is provided
        # Warning: This will only load the weights of the module
        # called "model" in this class
        if init_ckpt is not None:
            state = self.fabric.load(init_ckpt)['model']
            self.model.load_state_dict(state)

         # Initialize optimizer
        self.optimizer = utils.import_attr(optimizer)(self.model.parameters(), **optimizer_params)
        self.optim_name = optimizer
        self.opt_params = optimizer_params

        # Grad clip
        self.grad_clip = grad_clip

        if self.grad_clip is not None:
            print(f"USING GRAD CLIP: {self.grad_clip}")
        else:
            print("NOT USING GRAD CLIP")

        # Initialize scheduler
        self.scheduler = self.init_scheduler(scheduler, scheduler_params)
        self.scheduler_name = scheduler
        self.scheduler_params = scheduler_params
        
        self.epoch = 0
    
    def load_state(self, path, for_onnx=False):
        state = {"model": self.model,
                 "optimizer": self.optimizer,
                 "current_epoch": self.epoch,
                 "metric_values": self.metric_values,
                 "statistics": self.statistics}
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler
        
        self.fabric.load(path, state)
        self.epoch = state['current_epoch']
        self.metric_values = state['metric_values']
        self.statistics = state['statistics']

    def dump_state(self, path):        
        state = dict(model = self.model,
                     optimizer = self.optimizer,
                     current_epoch = self.epoch,
                     metric_values= self.metric_values, 
                     statistics = self.statistics)
        
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler
        self.fabric.save(path, state)

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def on_epoch_start(self):
        pass

    def get_avg_metric_at_epoch(self, metric, epoch = None):
        if epoch is None:
            epoch = self.epoch
        
        return self.metric_values[epoch][metric]['epoch'] / self.metric_values[epoch][metric]['num_elements']

    def gather_metrics(self):
        for metric in self.metric_values[self.epoch]:
            vals = self.fabric.all_gather(self.metric_values[self.epoch][metric]['epoch'])
            nums = self.fabric.all_gather(self.metric_values[self.epoch][metric]['num_elements'])

            self.metric_values[self.epoch][metric]['epoch'] = torch.sum(vals)
            self.metric_values[self.epoch][metric]['num_elements'] = torch.sum(nums)

    def on_epoch_end(self, best_path, wandb_run=None):
        assert self.epoch + 1 == len(self.metric_values), \
            "Current epoch must be equal to length of metrics (0-indexed)"

        # Gather metrics from multiple processes
        self.gather_metrics()

        monitor_metric_last = self.get_avg_metric_at_epoch(self.monitor)

        # Go over all epochs
        save = True
        for epoch in range(len(self.metric_values) - 1):
            monitor_metric_at_epoch = self.get_avg_metric_at_epoch(self.monitor, epoch)
            
            if self.monitor_mode == 'max':
                # If there is any model with monitor larger than current, then
                # this is not the best model
                if monitor_metric_last < monitor_metric_at_epoch:
                    save = False
                    break

            if self.monitor_mode == 'min':
                # If there is any model with monitor smaller than current, then
                # this is not the best model
                if monitor_metric_last > monitor_metric_at_epoch:
                    save = False
                    break
        
        # If this is best, save it
        if save:
            if self.fabric.global_rank == 0:
                print("Current checkpoint is the best! Saving it...")
            self.dump_state(best_path)
        
        val_loss = self.get_avg_metric_at_epoch('val/loss')
        val_si_sdr_i = self.get_avg_metric_at_epoch('val/si_sdr_i')

        if self.fabric.global_rank == 0:
            print(f'Val loss: {val_loss:.02f}')
            print(f'Val SI-SDRi: {val_si_sdr_i:.02f}dB')
        
        self.train_samples.clear()
        self.val_samples.clear()
        
        if self.scheduler is not None:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                # Get last metric
                self.scheduler.step(metrics=monitor_metric_last)
            else:
                self.scheduler.step()

        self.epoch += 1

    def log_statistic(self, name, value, reduction='mean'):
        if name not in self.statistics:
            self.statistics[name] = dict(logged=False, data=[], reduction=reduction)
        
        self.statistics[name]['data'].append(value)

    def log_metric(self, name, value, batch_size=1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True):
        """
        Logs a metric
        value must be the AVERAGE value across the batch
        Must provide batch size for accurate average computation
        """
        
        epoch_str = self.epoch
        if epoch_str not in self.metric_values:
            self.metric_values[epoch_str] = {}

        if (name not in self.metric_values[epoch_str]):
            self.metric_values[epoch_str][name] = dict(step=None, epoch=None)
        
        if type(value) == torch.Tensor:
            value = value.item()

        if on_step:            
            if self.metric_values[epoch_str][name]['step'] is None:
                self.metric_values[epoch_str][name]['step'] = []
            
            self.metric_values[epoch_str][name]['step'].append(value)
        
        if on_epoch:
            if self.metric_values[epoch_str][name]['epoch'] is None:
                self.metric_values[epoch_str][name]['epoch'] = 0
                self.metric_values[epoch_str][name]['num_elements'] = 0
             
            self.metric_values[epoch_str][name]['epoch'] += (value * batch_size)
            self.metric_values[epoch_str][name]['num_elements'] += batch_size

    def _step(self, batch, batch_idx, step='train'): 
        inputs, targets = batch
        batch_size = inputs['mixture'].shape[0]
        
        # Forward pass
        mix = inputs['mixture'].clone()
        
        outputs = self.model({"local_audio": inputs['mixture']})
        
        est: torch.Tensor
        gt: torch.Tensor
        
        # Only 1 source
        est = outputs['output'].clone()
        out_channel = est.shape[1]
        gt = targets['target'].clone()
        gt = gt[:, :out_channel, :]
        mix = mix[:, :out_channel, :]
        # print(est.shape, gt.shape)
        n_speakers = targets['num_target_speakers']

        # Compute loss
        loss = self.loss_fn(est=est, gt=gt).mean()

        est_detached = est.detach().clone()
        
        with torch.no_grad():
            # Log loss
            self.log_metric(f'{step}/loss', loss.item(), batch_size=batch_size, on_step=(step == 'train'), on_epoch=True, prog_bar=True, sync_dist=True)

            # Log metrics
            for metric in self.metrics:
                metric_val = metric(est=est_detached, gt=gt, mix=mix)
                for i in range(batch_size):
                    val = metric_val[i].item()
                    self.log_metric(f'{step}/{metric.name}', val, batch_size=1,
                            on_step=False, on_epoch=True, prog_bar=True,
                            sync_dist=True)

            # Log input snr
            if (f'stat/{step}_input_snr' not in self.statistics) or (not self.statistics[f'stat/{step}_input_snr']['logged']):
                for i in range(batch_size):
                    if n_speakers[i] > 0:
                        snr_val = self.snr_metric(est=mix[i].unsqueeze(0), gt=gt[i].unsqueeze(0), mix=mix[i].unsqueeze(0))
                        
                        self.log_statistic(f'stat/{step}_input_snr', snr_val.item(), reduction='histogram')
        
        # Create collection of things to show in a sample on wandb
        sample = {
            'mixture': mix,
            'output': est_detached,
            'target': gt,
            'n_tgt_speakers': n_speakers,
        }

        return loss, sample

    def train(self):
        self.model.train()
        self.mode = 'train'
    
    def eval(self):
        self.model.eval()
        self.mode = 'val'

    def training_step(self, batch, batch_idx):
        loss, sample = self._step(batch, batch_idx, step='train')

        n_speakers = sample['n_tgt_speakers']
        for i in range(n_speakers.shape[0]):
            spk_num = n_speakers[i].item()
            if spk_num not in self.train_samples:
                self.train_samples[spk_num] = []
            
            if len(self.train_samples[spk_num]) < self.samples_per_speaker_number:
                sample_at_batch = {}
                for k in sample:
                    sample_at_batch[k] = sample[k][i]
                self.train_samples[spk_num].append(sample_at_batch)
        
        return loss, n_speakers.shape[0]

    def validation_step(self, batch, batch_idx):
        loss, sample = self._step(batch, batch_idx, step='val')
        
        n_speakers = sample['n_tgt_speakers']
        for i in range(n_speakers.shape[0]):
            spk_num = n_speakers[i].item()
            if spk_num not in self.val_samples:
                self.val_samples[spk_num] = []
            
            if len(self.val_samples[spk_num]) < self.samples_per_speaker_number:
                sample_at_batch = {}
                for k in sample:
                    sample_at_batch[k] = sample[k][i]
                self.val_samples[spk_num].append(sample_at_batch)
        
        return loss, n_speakers.shape[0]
    
    def reset_grad(self):
        self.optimizer.zero_grad()

    def backprop(self):
        # Gradient clipping
        if self.grad_clip is not None:
            self.fabric.clip_gradients(self.model, self.optimizer, max_norm=self.grad_clip)
        
        self.optimizer.step()

    def init_scheduler(self, scheduler, scheduler_params):
        if scheduler is not None:
            if scheduler == 'sequential':
                schedulers = []
                milestones = []
                for scheduler_param in scheduler_params:
                    sched = utils.import_attr(scheduler_param['name'])(self.optimizer, **scheduler_param['params'])
                    schedulers.append(sched)
                    milestones.append(scheduler_param['epochs'])

                # Cumulative sum for milestones
                for i in range(1, len(milestones)):
                    milestones[i] = milestones[i-1] + milestones[i]

                # Remove last milestone as it is implied by num epochs
                milestones.pop()

                scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers, milestones)
            else:
                scheduler = utils.import_attr(scheduler)(self.optimizer, **scheduler_params)

        return scheduler

