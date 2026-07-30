import torch
import torch.nn as nn
import torch
from src.metrics.metrics import Metrics
from src.metrics.metrics import compute_decay
import src.utils as utils
import numpy as np
import pickle


def gather_variable_length(metrics, fabric):
    # Serialize local list of strings
    local_bytes = pickle.dumps(metrics)
    local_size = torch.tensor([len(local_bytes)], device=fabric.device)

    # Gather all sizes first
    all_sizes = fabric.all_gather(local_size)
    max_size = all_sizes.max().item()

    # Pad serialized data to max size
    local_tensor = torch.zeros(max_size, dtype=torch.uint8, device=fabric.device)
    local_tensor[:len(local_bytes)] = torch.tensor(list(local_bytes), dtype=torch.uint8, device=fabric.device)

    # Gather all padded data
    all_tensors = fabric.all_gather(local_tensor)

    # Deserialize each rank's list
    all_metrics = []
    for i, size in enumerate(all_sizes):
        data = bytes(all_tensors[i][:size])
        all_metrics.append(pickle.loads(data))

    return all_metrics


class MonauralWrapper(nn.Module):
    def __init__(self, model, model_params):
        super(MonauralWrapper, self).__init__()
        self.left = utils.import_attr(model)(**model_params)

    def forward(self, x_11, x_12, comm_delay):
        """
        x_11: Data at device 1 from device 1
        x_12: Data at device 1 from device 2

        embedding_audio: embedding_audio
        embedding_vec: embedding_vec 
        """
        
        # print(x_11.shape, x_12.shape, comm_delay)
        left_denoised = self.left(dict(local_audio=x_11, remote_audio=x_12), delay_chunks = comm_delay)['output']

        return {'output':left_denoised}

    def compile(self):
        self.left.compile()
        return self

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
                 grad_clip = None,
                 single_channel=True,
                 samples_per_speaker_number=5, use_compile=False):
        
        self.fabric = fabric
        
        self.single_ch = single_channel
        if single_channel:
            self.model = MonauralWrapper(model, model_params)

        if self.fabric.global_rank == 0:
            print(f"Model size: {(utils.model_size(self.model)/1e3):.03f}K")
            # print("+"*50)
            # for name, module in self.model.named_modules():
            #     print(name, ":", module)
            # print("+"*50)
        
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

        # Print all modules recursively
        if init_ckpt is not None:
            print("[Full two model] loading weights from init_ckpt ...")
            state = self.fabric.load(init_ckpt)['model']
            self.model.load_state_dict(state)
        else:
            print("Initilizing from scrtach")
        

         # Initialize optimizer
        self.optimizer = utils.import_attr(optimizer)(self.model.parameters(), **optimizer_params)
        self.optim_name = optimizer
        self.opt_params = optimizer_params

        # Grad clip
        self.grad_clip = grad_clip

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
        if for_onnx:
            print("Loading for onnx ...")
            self.fabric.load(path, state)
        else:
            print("Loading for CPU/GPU ...")
            torch.serialization.add_safe_globals([torch.optim.lr_scheduler.LinearLR])
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

        # print(f"[{self.fabric.global_rank}] Dumping")
        self.fabric.save(path, state)
        # print(f"[{self.fabric.global_rank}] Saved")

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
        metrics = list(self.metric_values[self.epoch].keys())
        # List of lists
        all_metrics = gather_variable_length(metrics, self.fabric)

        # MUST traverse in sorted order to make sure metrics aren't mixed up during gather()!!
        all_metrics = sorted(list(set([x for y in all_metrics for x in y])))
        
        # all_metrics = self.fabric.all_gather(metrics)
        # print(f"[{self.fabric.global_rank}]", all_metrics)
        
        for metric in all_metrics:
            # If GPU doesn't have this metric, add it
            if metric not in self.metric_values[self.epoch]:
                self.metric_values[self.epoch][metric] = {}
                self.metric_values[self.epoch][metric]['epoch'] = 0.0
                self.metric_values[self.epoch][metric]['num_elements'] = 0.0
            
            vals = self.fabric.all_gather(self.metric_values[self.epoch][metric]['epoch'])
            nums = self.fabric.all_gather(self.metric_values[self.epoch][metric]['num_elements'])

            # print(f"[{self.fabric.global_rank}] metric: {metric}\t val: {vals}\t, num_els:{nums}")
    
            self.metric_values[self.epoch][metric]['epoch'] = torch.sum(vals)
            self.metric_values[self.epoch][metric]['num_elements'] = torch.sum(nums)

        # print(f"[{self.fabric.global_rank}] Metrics:", self.metric_values[self.epoch])

    def on_epoch_end(self, best_path, wandb_run=None):
        assert self.epoch + 1 == len(self.metric_values), \
            "Current epoch must be equal to length of metrics (0-indexed)"
        # print(f"[{self.fabric.global_rank}] Gathering")
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
        
        # print(f"[{self.fabric.global_rank}] Save value: {save}")

        # If this is best, save it
        if save:
            if self.fabric.global_rank == 0:
                print("Current checkpoint is the best! Saving it...")
            # print(f"[{self.fabric.global_rank}] Saving")
            self.dump_state(best_path)
        
        val_loss = self.get_avg_metric_at_epoch('val/loss')
        val_si_sdr_i = self.get_avg_metric_at_epoch('val/si_sdr_i')
        val_snr_i = self.get_avg_metric_at_epoch('val/snr_i')
        val_si_snr_i = self.get_avg_metric_at_epoch('val/si_snr_i')
        # try:
        #     val_decay = self.get_avg_metric_at_epoch('val/decay')
        # except:
        #     print("[ERROR] extracting decay is causing issue")

        
        if self.fabric.global_rank == 0:
            print(f'Val loss: {val_loss:.02f}')
            print(f'Val SI-SDRi: {val_si_sdr_i:.02f}dB')
            print(f'Val SNRi: {val_snr_i:.02f}dB')
            print(f'Val SI-SNRi: {val_si_snr_i:.02f}dB')
            # try:
            #     print(f'Val Decay: {val_decay:.02f}dB')
            # except:
            #     print("[ERROR] printing decay is causing issue")
        
        if self.scheduler is not None:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                # Get last metric
                self.scheduler.step(metrics=monitor_metric_last)
            else:
                self.scheduler.step()

        if wandb_run is not None:
            if self.fabric.global_rank == 0:
                import wandb
                def log_audio(run, key, samples, sr):
                    columns = ['mixture', 'target', 'output']
                    wandb_samples = []
                    for i, sample in enumerate(samples):
                        for k in columns:
                            wandb_samples.append(wandb.Audio(
                                sample[k].permute(1, 0).cpu().numpy(),
                                sample_rate=sr, caption=f'{i}/{k}'))
                    run.log({key: wandb_samples}, commit=False, step=self.epoch + 1)

                # Log stuff on wandb
                wandb_run.log({'lr-Adam': self.get_current_lr()}, commit=False, step=self.epoch + 1)

                for metric in self.metric_values[self.epoch]:
                    wandb_run.log({metric: self.get_avg_metric_at_epoch(metric)}, commit=False, step=self.epoch + 1)
                
                for statistic in self.statistics:
                    if not self.statistics[statistic]['logged']:
                        data = self.statistics[statistic]['data']
                        reduction = self.statistics[statistic]['reduction']
                        if reduction == 'mean':
                            val = np.mean(data)
                        elif reduction == 'sum':
                            val = sum(data)
                        elif reduction == 'histogram':
                            data = [[d] for d in data]
                            table = wandb.Table(data=data, columns=[statistic])
                            val = wandb.plot.histogram(table, statistic, title=statistic)
                        else:
                            assert 0, f"Unknown reduction {reduction}."
                        wandb_run.log({statistic: val}, commit=False)
                        self.statistics[statistic]['logged'] = True
                
                # if (self.epoch < 10) or (self.epoch < 30 and (self.epoch % 5 == 0)) or (self.epoch % 10 == 0):
                #     for spk_num in self.train_samples:
                #         log_audio(wandb_run, f"train/audio_samples_{spk_num}spk", self.train_samples[spk_num], sr=self.sr)
                    
                #     for spk_num in self.val_samples:
                #         log_audio(wandb_run, f"val/audio_samples_{spk_num}spk", self.val_samples[spk_num], sr=self.sr)

                wandb_run.log({'epoch': self.epoch}, commit=True, step=self.epoch + 1)

        # print(f"[{self.fabric.global_rank}] Done")
        self.train_samples.clear()
        self.val_samples.clear()

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
                self.metric_values[epoch_str][name]['epoch'] = 0.0
                self.metric_values[epoch_str][name]['num_elements'] = 0.0
             
            self.metric_values[epoch_str][name]['epoch'] += (value * float(batch_size))
            self.metric_values[epoch_str][name]['num_elements'] += float(batch_size)

    def _step(self, batch, batch_idx, step='train'): 
        inputs, targets = batch

        x_11 = inputs['audio_at_dev1_from_dev1']
        x_12 = inputs['audio_at_dev1_from_dev2']

         # Reference channel for first device
        if self.single_ch:
            mix = x_11[:, 0:1]
        else:
            print("TODO ...")
        
        if 'drc_gain1' in inputs and inputs['drc_gain1'] is not None:
            apply_drc = True
            x_11 = x_11 * inputs['drc_gain1']
            x_12 = x_12 * inputs['drc_gain1']
        
        # Communication delay is fixed for all batches
        comm_delay = inputs['comm_delay_chunks']
        assert (comm_delay[0] == comm_delay).all()
        comm_delay = comm_delay[0]

        batch_size = x_11.shape[0]
        n_speakers = targets['num_target_speakers']
        n_interference_speakers = targets['num_interfering_speakers']
        num_noises = targets['num_noises']
        # gt_txt = targets['words_spoken'][0]

        if self.single_ch:
            gt = targets['target_at_dev1'] # [B, 1, t]
        else:
            gt = torch.cat([targets['target_at_dev1'], targets['target_at_dev2']], dim=1) # [B, 2, t]

        # Run forward
        outputs = self.model(x_11, x_12, comm_delay)
        
        est: torch.Tensor
        gt: torch.Tensor
        
       
        est = outputs['output'].clone()

        if apply_drc:
            if self.single_ch:
                est = est / inputs['drc_gain1']

        # debug = True
        # if debug:
        #     import os
        #     for i in range(mix.shape[0]):
        #         dirname = f'repro/{step}/epoch{self.epoch:04d}/batch{batch_idx:04d}_rank{self.fabric.global_rank}_item{i}'
        #         os.makedirs(dirname, exist_ok=True)
        #         utils.write_audio_file(os.path.join(dirname, 'mix.wav'), mix[i].cpu().numpy(), 16000)
        #         utils.write_audio_file(os.path.join(dirname, 'gt.wav'), gt[i].cpu().numpy(), 16000)        
        
        n_speakers = targets['num_target_speakers']
        # Compute loss
        loss = self.loss_fn(est=est, gt=gt).mean()

        est_detached = est.detach().clone()
        
        with torch.no_grad():
            # Log loss
            self.log_metric(f'{step}/loss', loss.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            # Log metrics
            for metric in self.metrics:
                if step == "train" and (metric.name == "WBPESQ" or metric.name == "NBPESQ" or metric.name == "STOI"):
                    continue
                metric_val = metric(est=est_detached, gt=gt, mix=mix)
                for i in range(batch_size):
                    #print(inputs['sample_dir'][i], batch_idx)
                    if n_speakers[i] > 0:
                        assert torch.abs(gt[i]).max() > 0, "Expected gt > 0"
                        #if metric.name == 'si_sdr_i' and metric_val[i] < -10:
                        #    print(metric_val[i].item(), inputs['sample_dir'][i], batch_idx)
                        val = metric_val[i].item()
                        self.log_metric(f'{step}/{metric.name}', val, batch_size=1,
                                on_step=False, on_epoch=True, prog_bar=True,
                                sync_dist=True)
            
            # Log metrics for zero speakers
            for i in range(batch_size):
                if n_speakers[i] == 0:
                    assert torch.abs(gt[i]).max() == 0, "Expected gt == 0"
                    decay = compute_decay(est_detached[i].unsqueeze(0), mix[i].unsqueeze(0)).item()
                    self.log_metric(f'{step}/decay', decay, batch_size=1,
                            on_step=False, on_epoch=True, sync_dist=True)
            
            # Log metrics for non-zero speakers
            for spk_num in range(1, max(n_speakers) + 1):
                for metric in self.metrics:
                    if metric.name == 'si_sdr_i':
                        for i in range(batch_size):
                            if n_speakers[i] == spk_num:
                                si_sdri_i_val = metric(est=est_detached[i].unsqueeze(0), gt=gt[i].unsqueeze(0), mix=mix[i].unsqueeze(0))
                                self.log_metric(f'{step}/{metric.name}_{spk_num}spk', si_sdri_i_val.item(), batch_size=1,
                                        on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                                # if si_sdri_i_val > 30:
                                #     import os
                                #     print("ERRONEOUS SI-SDRi VAL", si_sdri_i_val)
                                #     dirname = f'tests/batch{batch_idx:04d}_rank{self.fabric.global_rank}_item{i}'
                                #     os.makedirs(dirname, exist_ok=True)
                                #     utils.write_audio_file(os.path.join(dirname, 'mix.wav'), mix[i].cpu().numpy(), 16000)
                                #     utils.write_audio_file(os.path.join(dirname, 'out.wav'), est_detached[i].cpu().numpy(), 16000)
                                #     utils.write_audio_file(os.path.join(dirname, 'gt.wav'), gt[i].cpu().numpy(), 16000)

            # Log input snr
            if (f'stat/{step}_input_snr' not in self.statistics) or (not self.statistics[f'stat/{step}_input_snr']['logged']):
                for i in range(batch_size):
                    if n_speakers[i] > 0:
                        snr_val = self.snr_metric(est=mix[i].unsqueeze(0), gt=gt[i].unsqueeze(0), mix=mix[i].unsqueeze(0))
                        
                        self.log_statistic(f'stat/{step}_input_snr', snr_val.item(), reduction='histogram')                    
                    
                    self.log_statistic(f'stat/{step}_num_tgt_speakers', n_speakers[i].item(), reduction='histogram')
                    self.log_statistic(f'stat/{step}_num_interference_speakers', n_interference_speakers[i].item(), reduction='histogram')
                    self.log_statistic(f'stat/{step}_num_noises', num_noises[i].item(), reduction='histogram')
        
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

