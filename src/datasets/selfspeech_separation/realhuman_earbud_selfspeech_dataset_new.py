"""
Torch dataset object for synthetically rendered
spatial data
"""
from typing import Tuple, List, Dict

import torch
import numpy as np
import os
import lc3 

import src.utils as utils
from src.datasets.augmentations.audio_augmentations import AudioAugmentations
import glob
import soundfile as sf
import pandas as pd
# from torchmetrics.functional import signal_noise_ratio as snr
from torchmetrics.functional.audio import signal_noise_ratio as snr 

import scipy.signal as signal
import traceback
import pandas as pd
import librosa
from torchaudio.functional import resample


def compute_left_right_power_difference(audio, reference_channels = -1):
    # compute in dB
    left_power = 10 * torch.log10(torch.mean(audio[0]**2) + 1e-12)
    right_power = 10 * torch.log10(torch.mean(audio[reference_channels]**2) + 1e-12)
    return left_power - right_power

def drc(avg_sign, prev_val):
    a = np.abs(avg_sign)
    alpha = 0.999

    b, prev_val = signal.lfilter([1 - alpha], [1, -alpha], a, zi=[prev_val * alpha])
    
    b_dB = 20 * np.log10(b + 1e-6)

    tk = -25
    max_gain = 30
    
    gain_dB = np.clip(tk - b_dB, 0, max_gain)
    gain = 10 ** (gain_dB / 20)
    
    return gain.astype(np.float32), prev_val
def get_snr(target, mixture, EPS=1e-9):
    """
    Computes the average SNR across all channels
    """
    return snr(mixture, target).mean()

def get_scale_for_snr(target_snr, tgt, noise, reference_channels = None):
    """
    Returns the scale for a MULTICHANNEL noise signal to achieve an 
    average SNR (across all channels) equal to target snr.
    """

    denoised_tgt = tgt
    denoised_mix = tgt + noise

    if reference_channels is not None:
        # print(reference_channels)
        # print(denoised_tgt.shape)
        # print(denoised_mix.shape)
        denoised_tgt = denoised_tgt[reference_channels]
        denoised_mix = denoised_mix[reference_channels]
        
    current_snr = get_snr(denoised_tgt, denoised_mix)

    pwr = (current_snr - target_snr) / 20
    k = 10 ** pwr
    
    return k 


def get_subdirs(path):
    subdirs = sorted(os.listdir(path))
    subdirs = [subdir for subdir in subdirs if not subdir.startswith('.')]
    return subdirs

def apply_codec_artifacts(audio, sr):
    assert len(audio.shape) == 1, "Mono audio only"
    
    frame_duration = 10 # ms
    nchannels = 1
    bitrate = 16000 # 16kbps
    bitdepth = None # Float32

    # Setup encoder
    encoder = lc3.Encoder(int(frame_duration)*1000, sr, nchannels)#, libpath=args.libpath)
    frame_size = encoder.get_frame_bytes(bitrate)
    frame_length = encoder.get_frame_samples()
    bitrate = encoder.resolve_bitrate(frame_size)

    # Encode audio
    encoded = []
    for i in range(0, audio.shape[-1], frame_length):
        chunk = audio[..., i:i+frame_length]
        encoded_bytes = encoder.encode(chunk, frame_size)
        encoded.append(encoded_bytes)

    # Setup decoder
    decoder = lc3.Decoder(int(frame_duration)*1000, sr, nchannels)#, libpath=args.libpath)

    # Decode audio
    decoded_audio_chunks = []
    for enc_b in encoded:
        chunk = decoder.decode(enc_b, bitdepth)
        decoded_audio_chunks.append(chunk)
    
    decoded_audio = np.concatenate(decoded_audio_chunks)

    return decoded_audio


def apply_codec_artifacts_multichannel(audio, sr):
    assert len(audio.shape) == 2, "Multichannel audio only"
    num_channels = audio.shape[0]
    outs = []
    for i in range(num_channels):
        output_audio = apply_codec_artifacts(audio[i], sr)
        outs.append(output_audio)
    outs = np.stack(outs, axis = 0)
    return outs

class Dataset(torch.utils.data.Dataset):
    """
    Dataset of mixing self-speech and interference waveforms.
    """
    def __init__(self, 
            self_speech_dir: Dict[str, float], 
            interference_dir:  Dict[str, float], 
            noise_dir_config,
            split,
            samples_per_epoch,
            sr=16000,
            duration=5,
            augmentations = [],
            used_channels = [0, 3],
            far_sources_range = [0, 4],
            snr_min=-10, snr_max=5,
            use_drc=False,
            noise_prob = 0.0,
            compression_aug_ratio = 0.0,
            interleaving_prob = 0.0,
            swap_left_right = False,
            use_SE = False
        ):
        super().__init__()

        # self_speech_dir is list of dicts, with keys is the room name and values is the sampling probability of the room
        # interference_dir is list of dicts, with keys is the room name and values is the sampling probability of the room
        self.duration = int(duration * sr)
        self.sr = sr
        self.snr_range = (snr_min, snr_max)
        # self.reference_channels = used_channels  # reference channel to compute SNR
        self.used_channels = used_channels
        self.use_drc = use_drc
        self.epoch = 0
        self.swap_left_right = swap_left_right

        self.self_speech_dir = self_speech_dir
        self.near_samples = [] #{"audio": []}
        self.far_samples = [] #{"audio": [], "rir": []}
        self.samples_per_epoch = samples_per_epoch
        self.split = split
        self.far_sources_range = far_sources_range

        self.target_rooms = []
        self.interf_rooms = []

        self.room2near = {}
        self.room2far = {}
        self.path2already_compressed = {}

        self.compression_aug_ratio = compression_aug_ratio
        self.interleaving_prob = interleaving_prob
        ## noise dir to convolute with RIR
        self.noise_prob = noise_prob
        if noise_prob > 0:
            self.wham_dir=noise_dir_config.get('wham_dir', None) 
            self.musan_dir=noise_dir_config.get('musan_dir', None)
            self.musan_noise_types=noise_dir_config.get('musan_noise_types', None)
            self.babble_dir=noise_dir_config.get('babble_dir', None)
            self.musdb18_dir=noise_dir_config.get('musdb18_dir', None)
            self.fsd50k_dir = noise_dir_config.get('fsd50k_dir', None)
            self.noise_samples = self.get_noise_samples() 
        else:
            self.noise_samples = []
        
        for dataset_dir, sampling_prob in self_speech_dir.items():
            near_room_dict = {"audio": [], "prob": sampling_prob}
            rooms = get_subdirs(dataset_dir)
            self.target_rooms.extend(rooms)

            for room in rooms:
                # Check all distances
                self.room2near[room] = {'audio': []}

                room_dir = os.path.join(dataset_dir, room)
                
                metadata_path = os.path.join(room_dir, 'metadata.csv')
                df = pd.read_csv(metadata_path)

                for index, data in df.iterrows():
                    if data['audio_type'] == 'a':
                        audio_path = os.path.join(room_dir, os.path.basename(data['path']))
                        already_compressed = bool(data['already_compressed']) if 'already_compressed' in data else False
                        self.path2already_compressed[audio_path] = already_compressed
                        self.room2near[room]['audio'].append(audio_path)
                        near_room_dict['audio'].append(audio_path)
            
            self.near_samples.append(near_room_dict)

        for dataset_dir, sampling_prob in interference_dir.items():
            far_room_dict = {"audio": [], "rir": [], "prob": sampling_prob}
            rooms = get_subdirs(dataset_dir)
            self.interf_rooms.extend(rooms)
            for room in rooms:
                self.room2far[room] = {'audio': [], 'rir': []}
                room_dir = os.path.join(dataset_dir, room)
                metadata_path = os.path.join(room_dir, 'metadata.csv')
                df = pd.read_csv(metadata_path)
                for index, data in df.iterrows():
                    if data['audio_type'] == 'a' or data['audio_type'] == 'interference':
                        audio_path = os.path.join(room_dir, os.path.basename(data['path']))
                        # self.far_samples['audio'].append(audio_path)
                        self.room2far[room]['audio'].append(audio_path)
                        far_room_dict['audio'].append(audio_path)
                    elif data['audio_type'] == 'c':
                        distance = data['distance_cm']
                        if distance < 60:
                            print(f"Skipping RIR at distance {distance} cm because distance is too close for an interference source", data['path'])
                            continue
                        rir_path = os.path.join(room_dir, os.path.basename(data['path']))
                        # self.far_samples['rir'].append(rir_path)
                        self.room2far[room]['rir'].append(rir_path)
                        far_room_dict['rir'].append(rir_path)
            
            self.far_samples.append(far_room_dict)

        print("*"*25)
        # print(f"Room Near {len(self.near_samples['audio'])}")
        for near_dataset in self.near_samples:
            print("Dataset1: ", near_dataset['prob'], len(near_dataset['audio']))
        for i in self.room2near.keys():
            print(f"room: {i} audio: {len(self.room2near[i]['audio'])}")
        print("*"*25)
        for far_dataset in self.far_samples:
            print("Dataset2: ", far_dataset['prob'], len(far_dataset['audio']), len(far_dataset['rir']))
        # # print(f"Room Far {len(self.far_samples['audio'])}")
        for i in self.room2far.keys():
            print(f"room: {i} rirs: {len(self.room2far[i]['rir'])}, audio: {len(self.room2far[i]['audio'])}")

        print("="*25)
        # Data augmentation
        self.augmentations = AudioAugmentations(augmentations)
                
    def get_noise_samples(self):
        # Get WHAM! files from this split
        self.wham_samples = []
        if self.noise_prob > 0 and self.wham_dir:
            if self.split == 'train':
                wham_split = 'tr'
            elif self.split == 'test':
                wham_split = 'tt'
            elif self.split == 'val':
                wham_split = 'cv'
            else:
                assert 0, f"split must be one of [\'train\', \'test\', \'val\'], found {self.split}"
            #wham_samples = wham_meta[wham_meta['WHAM! Split'] == wham_split]['Filename']
            #self.wham_samples = [os.path.join(self.wham_dir, 'audio', x) for x in wham_samples]
            self.wham_samples = glob.glob(os.path.join(self.wham_dir, wham_split)+"/*.wav")
            
         # Get MUSAN files from this split
        self.musan_samples = []
        if self.noise_prob > 0 and self.musan_dir:
            if self.split == 'train':
                musan_split = 'train'
            elif self.split == 'test':
                musan_split = 'test'
            elif self.split == 'val':
                musan_split = 'val'
            else:
                assert 0, f"split must be one of [\'train\', \'test\', \'val\'], found {self.split}"
            musan_samples = []
            musan_df = pd.read_csv(os.path.join(self.musan_dir, "metadata.csv"))
            
            
            for m_n_t in self.musan_noise_types:
                musan_files = musan_df[musan_df["filetype"] == f"{musan_split}_{m_n_t}"]["filename"].tolist()
                musan_samples.extend(musan_files)
            self.musan_samples = [os.path.join(self.musan_dir, f) for f in musan_samples]   
        

        # Get babble files from this split
        self.babble_samples = []
        if self.babble_dir is not None:
            if self.split == 'train':
                babble_split = 'train'
            elif self.split == 'test':
                babble_split = 'test'
            elif self.split == 'val':
                babble_split = 'val'
            self.babble_samples = glob.glob(os.path.join(self.babble_dir, babble_split)+"/*.wav")
        
        
        self.musdb18_samples = []
        if self.musdb18_dir is not None:
            if self.split == 'train':
                musdb18_split = 'train'
            elif self.split == 'test':
                musdb18_split = 'test'
            elif self.split == 'val':
                musdb18_split = 'test'
            self.musdb18_samples = glob.glob(os.path.join(self.musdb18_dir, musdb18_split)+"/*.wav")
        
        self.fsd50k_samples = []
        if self.fsd50k_dir is not None:
            if self.split == 'train':
                fsd50k_split = 'FSD50K.dev_audio'
            elif self.split == 'test':
                fsd50k_split = 'FSD50K.eval_audio'
            elif self.split == 'val':
                fsd50k_split = 'FSD50K.eval_audio'
            self.fsd50k_samples = glob.glob(os.path.join(self.fsd50k_dir, fsd50k_split)+"/*.wav")
        
        noise_samples = self.wham_samples + self.musan_samples + self.babble_samples + self.musdb18_samples + self.fsd50k_samples
        
        print("="*25)
        print("Converting to equal weightage ...")
        rng_for_noise = np.random.RandomState(42)
        max_noise_samples_per_dataset = max(len(self.wham_samples), len(self.musan_samples), len(self.babble_samples), len(self.musdb18_samples), len(self.fsd50k_samples))
        
        if self.wham_dir and len(self.wham_samples) < max_noise_samples_per_dataset :
            self.wham_samples = self.wham_samples + rng_for_noise.choice(self.wham_samples, max_noise_samples_per_dataset - len(self.wham_samples), replace=True).tolist()
        
        if self.musan_dir and len(self.musan_samples) < max_noise_samples_per_dataset :
            self.musan_samples = self.musan_samples + rng_for_noise.choice(self.musan_samples, max_noise_samples_per_dataset - len(self.musan_samples), replace=True).tolist()

        if self.babble_dir and len(self.babble_samples) < max_noise_samples_per_dataset :
            self.babble_samples = self.babble_samples + rng_for_noise.choice(self.babble_samples, max_noise_samples_per_dataset - len(self.babble_samples), replace=True).tolist()
        
        if self.musdb18_dir and len(self.musdb18_samples) < max_noise_samples_per_dataset :
            self.musdb18_samples = self.musdb18_samples + rng_for_noise.choice(self.musdb18_samples, max_noise_samples_per_dataset - len(self.musdb18_samples), replace=True).tolist()
        
        if self.fsd50k_dir and len(self.fsd50k_samples) < max_noise_samples_per_dataset :
            self.fsd50k_samples = self.fsd50k_samples + rng_for_noise.choice(self.fsd50k_samples, max_noise_samples_per_dataset - len(self.fsd50k_samples), replace=True).tolist()
        
        noise_samples = self.wham_samples + self.musan_samples + self.babble_samples + self.musdb18_samples + self.fsd50k_samples
        
        print(f"Number of wham samples: {len(self.wham_samples)}")
        print(f"Number of MUSAN samples: {len(self.musan_samples)}")
        print(f"Number of babble samples: {len(self.babble_samples)}")
        print(f"Number of musdb18 samples: {len(self.musdb18_samples)}")
        print(f"Number of fsd50k samples: {len(self.fsd50k_samples)}")
        print(f"Number of Noise samples: {len(noise_samples)}")

        return noise_samples


    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def load_audio(self, chosen_path, rng: np.random.RandomState):  
        if "/parsed_earbud_human_data/" in chosen_path:
            ### bugs for the old datset left and right are swapped
            left_path = os.path.join(chosen_path, 'right.wav')
            right_path = os.path.join(chosen_path, 'left.wav')
        else:
            left_path = os.path.join(chosen_path, 'left.wav')
            right_path = os.path.join(chosen_path, 'right.wav')
        
        left_audio, sr = librosa.load(left_path, sr = None, mono=False)
        right_audio, sr = librosa.load(right_path, sr = None, mono=False)
        assert sr == self.sr
        
        assert left_audio.shape[-1] == right_audio.shape[-1]
        assert len(left_audio.shape) == 2, f"weird audio shapoe {left_audio.shape} at {chosen_path}"
        assert len(right_audio.shape) == 2, f"weird audio shapoe {right_audio.shape} at {chosen_path}"

        audio = np.concatenate([left_audio, right_audio], axis=0)
        return audio
    
    def load_noise(self, chosen_path, rng: np.random.RandomState):      
        # print(f"Loading noise from {chosen_path}")
        left_audio, sr = librosa.load(chosen_path, sr = self.sr, mono=False)
        assert sr == self.sr
        
        if len(left_audio.shape) == 1:
            left_audio = left_audio[np.newaxis, :]
        assert len(left_audio.shape) == 2, f"weird audio shapoe {left_audio.shape} at {chosen_path}"
 
        if left_audio.shape[-1] < self.duration:
            zero_pad = self.duration - left_audio.shape[-1]
            front_pad = rng.randint(0, zero_pad)
            back_pad = zero_pad - front_pad
            left_audio = np.pad(left_audio, ((0, 0), (front_pad, back_pad)))
        
        elif left_audio.shape[-1] > self.duration:
            delta = left_audio.shape[-1] - self.duration
            start_id = rng.randint(0, delta)
            left_audio = left_audio[:, start_id : start_id+self.duration]


        audio = left_audio[0]
        return audio


    def load_audio_from_rir(self, chosen_path, rng: np.random.RandomState):
        '''
        Load audio from RIR and convolve it with the noise randomly samples from self.noise_samples
        '''
        noise_sample = rng.choice(self.noise_samples, size=1, replace=True).tolist()
        single_ch_noise = self.load_noise(noise_sample[0], rng)

        # print(f"Loading RIR from {chosen_path}")
        if "/parsed_earbud_human_data/" in chosen_path:
            ### bugs for the old datset left and right are swapped
            left_rir_path = os.path.join(chosen_path, 'right.wav')
            right_rir_path = os.path.join(chosen_path, 'left.wav')
        else:
            left_rir_path = os.path.join(chosen_path, 'left.wav')
            right_rir_path = os.path.join(chosen_path, 'right.wav')

        left_rir, sr = librosa.load(left_rir_path, sr = None, mono=False)
        right_rir, sr = librosa.load(right_rir_path, sr = None, mono=False)
        assert sr == self.sr
        
        assert left_rir.shape[-1] == right_rir.shape[-1]
        assert len(left_rir.shape) == 2, f"weird audio shapoe {left_rir.shape} at {chosen_path}"
        assert len(right_rir.shape) == 2, f"weird audio shapoe {right_rir.shape} at {chosen_path}"

        audio = []
        rir = np.concatenate([left_rir, right_rir], axis=0)
        num_channels = rir.shape[0]
        for i in range(num_channels):
            RIR = rir[i]
            spatial_audio = signal.convolve(single_ch_noise, RIR, mode='full')
            audio.append(spatial_audio[: -(len(RIR) - 1)])
        
        audio = np.stack(audio, axis=0)
        assert audio.shape[-1] == self.duration, f"expected duration {self.duration}, found {audio.shape[-1]} at {chosen_path}"

        return audio


    def pad_chunk_audio(self, audio, duration, rng: np.random.RandomState):
        if audio.shape[-1] < duration:
            zero_pad = duration - audio.shape[-1]
            front_pad = rng.randint(0, zero_pad)
            back_pad = zero_pad - front_pad
            audio = torch.nn.functional.pad(audio, (front_pad, back_pad) )
        else:
            delta = audio.shape[-1] - duration
            start_id = rng.randint(0, delta) if delta > 0 else 0
            audio = audio[:, start_id : start_id+duration]
        return audio


    def mixing_near_and_far_with_interleaving(self, near_audio, far_audio, rng: np.random.RandomState):
        '''
        Mixing near and far audio with interleaving probability
        '''
        assert len(near_audio) > 0, f"near audio is empty"
        assert len(far_audio) > 0, f"far audio is empty"

        if rng.uniform(0,1) < self.interleaving_prob:
            # interleaving the near audio and far audio
            if rng.uniform(0, 1) < 0.3:
                # target is always zeros
                # print("[Data Type]target audio set to zeros! ")
                near_audio = torch.zeros(near_audio[0].shape[0], self.duration)
                fars = []
                for far in far_audio:
                    far = self.pad_chunk_audio(far, self.duration, rng)
                    fars.append(far) 
                far_audio = sum(fars)
                return near_audio, far_audio

            else:
                # print("[Data Type]target audio interleaving with noise! ")
                interval = rng.uniform(-1, 1) # silence between target and noise
                interval = int(interval * self.sr)
                min_active_duration = self.sr * 1

                high = self.duration - interval - min_active_duration
                if high <= min_active_duration:
                    high = min_active_duration + 1
                targe_duration = rng.randint(min_active_duration, high)
                noise_duration = self.duration - targe_duration - interval

                nears = []
                for near in near_audio:
                    near = self.pad_chunk_audio(near, targe_duration, rng)
                    nears.append(near)
                near_audio = sum(nears)
                fars = []
                for far in far_audio:
                    far = self.pad_chunk_audio(far, noise_duration, rng)
                    fars.append(far)
                far_audio = sum(fars)

                tgt = torch.zeros(near_audio.shape[0], self.duration)
                noise = torch.zeros(near_audio.shape[0], self.duration)
                order_prob = rng.uniform(0,1)

                if order_prob < 0.5:
                    tgt[:, :targe_duration] = near_audio
                    noise[:, -noise_duration:] = far_audio
                else:
                    tgt[:, -targe_duration:] = near_audio
                    noise[:, :noise_duration] = far_audio

                return tgt, noise
        else:
            # overlap the near audio and far audio
            # print("[Data Type]target audio overlap with noise! ")
            nears = []
            for near in near_audio:
                near = self.pad_chunk_audio(near, self.duration, rng)
                nears.append(near)
            near_audio = sum(nears)

            fars = []
            for far in far_audio:
                far = self.pad_chunk_audio(far, self.duration, rng)
                fars.append(far)
            far_audio = sum(fars)
            return near_audio, far_audio


    def sample_near_audio(self, num_near, num_far, rng: np.random.RandomState):
        '''
        Sample near audio from the near samples
        '''
        near_probs = [near_dataset['prob'] for near_dataset in self.near_samples]
        # normalize the probabilities
        near_probs = np.array(near_probs) / np.sum(near_probs)
        near_dataset = rng.choice(self.near_samples, size=1, replace=True, p=near_probs).tolist()
        near_dataset = near_dataset[0]
        near_audio_order = rng.choice(near_dataset['audio'], size=1, replace=True).tolist()
        assert len(near_audio_order) == 1, f"near audio order length is not 1"

        far_probs = [far_dataset['prob'] for far_dataset in self.far_samples]
        far_probs = np.array(far_probs) / np.sum(far_probs)
        far_dataset = rng.choice(self.far_samples, size=num_far, replace=True, p=far_probs).tolist()
        far_audio_order = []
        for dataset in far_dataset:
            select_path = rng.choice(dataset['audio'], size=1, replace=True).tolist()
            far_audio_order.append(select_path[0])
        assert len(far_audio_order) == num_far, f"far audio order length is not {num_far}"

        return near_audio_order, far_audio_order
    
    def create_scene(self, idx: int, rng: np.random.RandomState, used_channels_final: list[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        num_near = 1
        num_far = rng.randint(self.far_sources_range[0], self.far_sources_range[1] + 1)

        near_audio_order, far_audio_order = self.sample_near_audio(num_near, num_far, rng)
        # print(f"near_audio_order: {near_audio_order}, far_audio_order: {far_audio_order}")
        near_already_compressed = self.path2already_compressed.get(near_audio_order[0], False)
        # print(f"near_already_compressed: {near_already_compressed} at {near_audio_order[0]}")
        room_name = near_audio_order[0]
        far_rir_order = []
        selected_far_audio_order = far_audio_order
        ## not support noise yet
        # if self.noise_prob > 0:
        #     far_rir_order = rng.choice(self.far_samples['rir'], size=num_far, replace=True).tolist()
        # else:
        #     far_rir_order = []
        
        audio_list = []
        for i in range(num_near + num_far):
            if i < num_near:
                # print(near_audio_order[0])
                audio = self.load_audio(near_audio_order[i], rng)
                # audio = torch.from_numpy(audio)
                audio_list.append(audio)
            else:
                if self.split == 'train':
                    rir_p = rng.uniform(0,1)
                else:
                    rir_p = 1
                if rir_p <= self.noise_prob:
                    audio = self.load_audio_from_rir(far_rir_order[i-num_near], rng)
                else:
                    audio = self.load_audio(far_audio_order[i-num_near], rng)
                # audio = torch.from_numpy(audio)
                audio_list.append(audio)
                # assert(audio.shape == audio_list[0].shape), f"shape mismatch noise {audio.shape} and target {audio_list[0].shape}"

        # Randomly scale each audio event and trim the audio
        for i in range(len(audio_list)):
            max_val = np.abs(audio_list[i]).max()
            if max_val < 1e-9:
                print(f"[Warning] audio {i} loaded power is very small and set it to all zeros")
                audio = np.zeros_like(audio_list[i])
                audio_list[i] = torch.from_numpy(audio)
            else:
                scale = rng.uniform(0.1, 1)
                audio = ( audio_list[i] /  max_val ) * scale
                # trim the audio 
                SAFE_GUARD = 160
                _, indexes = librosa.effects.trim(audio, top_db=20)
                start_i = max(0, indexes[0] - SAFE_GUARD)
                end_i = min(audio.shape[-1], indexes[1] + SAFE_GUARD)
                audio = audio[:, start_i:end_i]
                audio_list[i] = torch.from_numpy(audio)
        
        tgt, noise = self.mixing_near_and_far_with_interleaving(audio_list[:num_near], audio_list[num_near:], rng)
        assert tgt.shape == noise.shape, f"shape mismatch tgt {tgt.shape} and noise {noise.shape}"
        rms_tgt = torch.sqrt(torch.mean(tgt**2))
        rms_noise = torch.sqrt(torch.mean(noise**2))

         # Compute the amount to scale the noise by so that the SNR is uniformly distributed
        target_snr = None
        if rms_tgt > 1e-6 and rms_noise > 1e-6:
            target_snr = rng.uniform(self.snr_range[0], self.snr_range[1])
            snr_scale = get_scale_for_snr(target_snr, tgt, noise, used_channels_final)
            noise = noise * snr_scale
            

        elif rms_noise <= 1e-6 and rms_tgt > 1e-6: # noise too low
            # print(f"[Warning] {far_audio_order} noise power is very small {rms_noise}")
            num_far = 0
        elif rms_tgt <= 1e-6 and rms_noise > 1e-6: # target too low
            # print(f"[Warning] {near_audio_order[0]} No target audio found or loaded power is very small")
            tgt = torch.zeros_like(tgt)
            num_near = 0
        else:
            raise ValueError(f"Invalid state: rms_tgt {rms_tgt} and rms_noise {rms_noise}")
            
        # Get the target and mixture audio
        mixture = noise + tgt
        # Scale mixture so that peak is <= 1
        peak = torch.abs(mixture).max().item()
        if peak > 1:
            mixture /= peak
            tgt /= peak

        assert (torch.isnan(mixture).max().item() == 0), "mixture tensor is nan!"

        return mixture, tgt, num_near, num_far, target_snr, near_already_compressed, room_name, selected_far_audio_order


    def get_item_from_paths(self, near_path: str, far_paths: list, seed: int, forced_target_snr: float = None):
        """Fixed-path entry point for apple-to-apple comparison. No interleaving, negative samples disabled."""
        rng = np.random.RandomState(seed)
        used_channels_final = self.used_channels
        near_already_compressed = self.path2already_compressed.get(near_path, False)

        # Load and process near audio
        near_audio = self.load_audio(near_path, rng)
        max_val = np.abs(near_audio).max()
        num_near = 1
        if max_val < 1e-9:
            near_t = torch.zeros_like(torch.from_numpy(near_audio))
            num_near = 0
        else:
            scale = rng.uniform(0.1, 1)
            near_np = (near_audio / max_val) * scale
            SAFE_GUARD = 160
            _, indexes = librosa.effects.trim(near_np, top_db=20)
            start_i = max(0, indexes[0] - SAFE_GUARD)
            end_i = min(near_np.shape[-1], indexes[1] + SAFE_GUARD)
            near_np = near_np[:, start_i:end_i]
            near_t = torch.from_numpy(near_np)
        near_t = self.pad_chunk_audio(near_t, self.duration, rng)
        tgt = near_t

        # Load and process far audios (overlap mode, no interleaving)
        far_list = []
        for far_path in far_paths:
            audio = self.load_audio(far_path, rng)
            max_val_far = np.abs(audio).max()
            if max_val_far < 1e-9:
                far_t = torch.zeros_like(tgt)
            else:
                scale_far = rng.uniform(0.1, 1)
                audio_np = (audio / max_val_far) * scale_far
                _, idx_far = librosa.effects.trim(audio_np, top_db=20)
                si = max(0, idx_far[0] - 160)
                ei = min(audio_np.shape[-1], idx_far[1] + 160)
                audio_np = audio_np[:, si:ei]
                far_t = torch.from_numpy(audio_np)
            far_t = self.pad_chunk_audio(far_t, self.duration, rng)
            far_list.append(far_t)
        noise = sum(far_list) if far_list else torch.zeros_like(tgt)
        num_far = len(far_list)

        rms_tgt = torch.sqrt(torch.mean(tgt**2))
        rms_noise = torch.sqrt(torch.mean(noise**2))
        target_snr = forced_target_snr
        if rms_tgt > 1e-6 and rms_noise > 1e-6:
            if target_snr is None:
                target_snr = rng.uniform(self.snr_range[0], self.snr_range[1])
            snr_scale = get_scale_for_snr(target_snr, tgt, noise, used_channels_final)
            noise = noise * snr_scale
        elif rms_noise <= 1e-6:
            num_far = 0
        elif rms_tgt <= 1e-6:
            tgt = torch.zeros_like(tgt)
            num_near = 0

        mixture = noise + tgt
        peak = torch.abs(mixture).max().item()
        if peak > 1:
            mixture /= peak
            tgt /= peak

        if len(used_channels_final) > 1:
            x_11 = mixture[used_channels_final[0], :].unsqueeze(0)
            x_12 = mixture[used_channels_final[1], :].unsqueeze(0)
            gt1 = tgt[used_channels_final[0], :].unsqueeze(0)
            gt2 = tgt[used_channels_final[1], :].unsqueeze(0)
            mixture_out = torch.cat([x_11, x_12], dim=0)
            target_out = torch.cat([gt1, gt2], dim=0)
        else:
            mixture_out = mixture[used_channels_final[0], :].unsqueeze(0)
            target_out = tgt[used_channels_final[0], :].unsqueeze(0)

        inputs = {'mixture': mixture_out}
        targets_out = {
            'target': target_out,
            'num_target_speakers': num_near,
            'num_speakers': num_near + num_far,
            'target_snr': target_snr if target_snr is not None else 0,
            'room_name': near_path,
            'near_path': near_path,
            'far_paths': far_paths,
        }
        return inputs, targets_out

    def swap_mic_channels(self, used_channels):
        used_channels_swap = []
        for c in used_channels:
            if c < 4:
                used_channels_swap.append(c + 4)
            else:
                used_channels_swap.append(c - 4)
        return used_channels_swap

    def __getitem__(self, idx: int, seed=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if seed is None:
            if self.split == 'train':
                # IT IS ACTUALLY **** EXTREMELY **** IMPORTANT TO ADD IDX, ESPECIALLY IF WE ARE FIXING THE WORKERS SEEDS
                # OTHERWISE ALL WORKERS WILL HAVE THE SAME SEED!!!
                seed = idx + self.epoch * len(self)
            else:
                seed = idx
        rng = np.random.RandomState(seed)
        
        if self.swap_left_right:
            if rng.uniform(0, 1) < 0.4:
                used_channels_final = self.swap_mic_channels(self.used_channels)
            else:
                used_channels_final = self.used_channels
        else:
            used_channels_final = self.used_channels
        mixture, tgt, num_near, num_far, target_snr, near_already_compressed, room_name, far_paths = self.create_scene(idx, rng, used_channels_final)
        # Sanity check
        if num_near > 0:
            assert torch.abs(tgt).max().item() > 0


        if len(used_channels_final) > 1:
            x_11 = mixture[used_channels_final[0], :].unsqueeze(0)
            x_12 = mixture[used_channels_final[1], :].unsqueeze(0)
            gt1 = tgt[used_channels_final[0], :].unsqueeze(0)
            gt2 = tgt[used_channels_final[1], :].unsqueeze(0)

            if self.split == 'train':
                # Apply aug on dev 1
                mixture = torch.cat([x_11, x_12], dim=0)
                num_ch = x_11.shape[0]
                x_1, gt1 = self.augmentations.apply_random_augmentations(mixture, gt1, rng)
                x_11 = x_1[:num_ch]
                x_12 = x_1[num_ch:]
            # Apply DRC
            if self.use_drc:
                avg_sign = (x_11[0] + x_12[0]) / 2
                g1, _ = drc(avg_sign.numpy(), 0.5)
                g1 = torch.from_numpy(g1).unsqueeze(0) * 0.8
                x_11 = x_11 * g1
                x_12 = x_12 * g1
                gt1 = gt1 * g1
                gt2 = gt2 * g1
            mixture = torch.cat([x_11, x_12], dim=0)
            target = torch.cat([gt1, gt2], dim=0)

        else:
            x_11 = mixture[used_channels_final[0], :].unsqueeze(0)
            gt1 = tgt[used_channels_final[0], :].unsqueeze(0)
            if self.split == 'train':
                # Apply aug on dev 1
                x_11, gt1 = self.augmentations.apply_random_augmentations(x_11, gt1, rng)
            # Apply DRC
            if self.use_drc:
                avg_sign = x_11[0]
                g1, _ = drc(avg_sign.numpy(), 0.5)
                g1 = torch.from_numpy(g1).unsqueeze(0) * 0.8
                x_11 = x_11 * g1
                gt1 = gt1 * g1

            mixture = x_11
            target = gt1    

        # print(f"target_snr: {target_snr}, actual_snr: {get_snr(target, mixture)}")

        if (target_snr is not None) and (len(self.augmentations) == 0) and (not self.use_drc):
            
            actual_snr = get_snr(target, mixture)
            assert torch.abs(actual_snr - target_snr) < 0.1
        else:
            target_snr = 0

        if self.compression_aug_ratio > 0 and not near_already_compressed:
            mixture = apply_codec_artifacts_multichannel(mixture.numpy(), self.sr)
            target = apply_codec_artifacts_multichannel(target.numpy(), self.sr)
            mixture = torch.from_numpy(mixture)
            target = torch.from_numpy(target)

        inputs = {
            'mixture': mixture,
            "g1": g1,
        }

        targets = {
            'target': target,
            "num_target_speakers": num_near,
            "num_speakers": num_near + num_far,
            "target_snr": target_snr,
            "room_name": room_name,
            "near_path": room_name,
            "far_paths": far_paths,
        }

        return inputs, targets


