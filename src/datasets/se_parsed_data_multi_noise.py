"""
Torch dataset object for synthetically rendered
spatial data
"""
from typing import Tuple

import torch
import numpy as np
import os

from src.datasets.augmentations.audio_augmentations import AudioAugmentations
import glob
import soundfile as sf
import pandas as pd
from torchmetrics.functional.audio import signal_noise_ratio as snr 

import scipy.signal as signal
import pandas as pd
import librosa

from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
import torchaudio.functional as AF
from scipy.signal import butter, filtfilt

def brickwall_lowpass_stft(
    audio,
    sr=16000,
    cutoff=800,
    n_fft=4096,
    hop_length=None,
    window=None,
):
    """
    audio: Tensor [C, T]
    returns: filtered Tensor [C, T]
    """

    C, T = audio.shape
    device = audio.device

    if hop_length is None:
        hop_length = n_fft // 4  # good reconstruction default

    if window is None:
        window = torch.hann_window(n_fft, device=device)

    # -------------------------
    # STFT
    # -------------------------
    S = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )  # shape: [C, F, frames]

    # -------------------------
    # Compute cutoff bin
    # -------------------------
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1, device=device)
    cut_bin = torch.searchsorted(freqs, cutoff)

    # -------------------------
    # Brick-wall mask
    # -------------------------
    mask = torch.zeros_like(S)
    mask[:, :cut_bin, :] = 1.0

    S_filtered = S * mask

    # -------------------------
    # ISTFT
    # -------------------------
    filtered = torch.istft(
        S_filtered,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=T,
    )

    return filtered

def estimate_noise(mix, tgt):
    cutoff = 800  # Hz
    order = 6   # filter steepness
    nyq = 16000 / 2
    normal_cutoff = cutoff / nyq
    b_high, a_high = butter(order, normal_cutoff, btype='high')
    b_low, a_low = butter(order, normal_cutoff, btype='low')
    
    
    mix_high = torch.tensor(np.ascontiguousarray(filtfilt(b_high, a_high, mix)), dtype=mix.dtype)
    tgt_high = torch.tensor(np.ascontiguousarray(filtfilt(b_high, a_high, tgt)), dtype=tgt.dtype)

    mix_low = torch.tensor(np.ascontiguousarray(filtfilt(b_low, a_low, mix)), dtype=mix.dtype)
    tgt_low = torch.tensor(np.ascontiguousarray(filtfilt(b_low, a_low, tgt)), dtype=tgt.dtype)

    


    o_snr = snr(mix, tgt)
    h_snr = snr(mix_high, tgt_high)
    l_snr = snr(mix_low, tgt_low)
    print(f"SNR: {o_snr}, high_snr: {h_snr}, low_snr: {l_snr}")


def drc(avg_sign, prev_val):
    a = np.abs(avg_sign)
    alpha = 0.999

    b, prev_val = lfilter([1 - alpha], [1, -alpha], a, zi=[prev_val * alpha])
    
    b_dB = 20 * np.log10(b + 1e-6)

    tk = -25
    max_gain = 30
    
    gain_dB = np.clip(tk - b_dB, 0, max_gain)
    gain = 10 ** (gain_dB / 20)
    
    return gain.astype(np.float32), prev_val


def rms(audio: np.ndarray, EPS=1e-6):
    return 20 * np.log10 ( np.sqrt(np.mean(audio ** 2)) + EPS)

class sample_descriptor:
    def __init__(self, path, angle, distance, speaker_id) -> None:
        self.path = path
        self.angle = angle
        self.distance = distance
        self.speaker_id = None
        if not np.isnan(speaker_id):
            self.speaker_id = str(int(speaker_id))

def get_snr(target, mixture, EPS=1e-9):
    """
    Computes the average SNR across all channels
    """
    return snr(mixture, target).mean()

def get_scale_for_snr(target_snr, denoised_audio, num_near, reference_channels = None):
    """
    Returns the scale for a MULTICHANNEL noise signal to achieve an 
    average SNR (across all channels) equal to target snr.
    """

    denoised_tgt = sum(denoised_audio[:num_near])
    denoised_mix = sum(denoised_audio)

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


class Dataset(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).

    Each scenario is represented by a folder. Multiple datapoints are generated per
    scenario. This can be customized using the points_per_scenario parameter.
    """
    def __init__(self, dataset_dir, libritts_dir, 
                 split, samples_per_epoch,
                 use_split_for_rooms = None,
                 wham_dir=None,
                 musan_dir=None,
                 musan_noise_types=None,
                 babble_dir=None,
                 musdb18_dir=None,
                 fsd50k_dir = None,
                 DSED_dir = None,
                 dsed_selected_classes = None,
                 Noise_Equal_Weightage=False,
                 n_mics=8,
                 sr=16000,
                 min_active_duration=2, duration=5,
                 dis_threshold = 1.5,
                 near_speakers_min = 1, near_speakers_max = 1,
                 far_sources_min = 0, far_sources_max = 0,
                 noise_sources_min = 0, noise_sources_max = 0,
                 noise_probability=1, # Probability that a far source is a noise source
                 snr_min=-10, snr_max=5,
                 control_noise_sourcewise_power=False,
                 silence_prob=0,
                 silence_min = 1, silence_max = 4,
                 complete_silence_prob = 0,
                 far_noise_only = True,
                 apply_lowpass = None,
                 augmentations = [],
                 comm_delay_chunks = 6,
                 chunk_size = 96,
                 dev1_channels = [0,1,2],
                 dev1_channels_to_share = [7],
                 dev2_channels = [4,5,6],
                 dev2_channels_to_share = [3],
                 boundary_safety=25,
                 rir_ratio=0.8, # 80% RIR data
                 room_mixing = False, # Use multiple rooms to create mixtures
                 use_drc = False, # Apply DRC on mixture
                 ):
        super().__init__()
        self.epoch = 0
        self.apply_lowpass = apply_lowpass
        self.samples_per_epoch = samples_per_epoch
        self.split = split
        
        self.sr = sr
        self.dis_threshold_cm = int(dis_threshold * 100)
        self.n_mics = n_mics
        self.dataset_dir = dataset_dir
        self.far_noise_only = far_noise_only
        self.min_active_duration = min_active_duration # Time where speaker is active
        self.duration = duration
        self.wham_dir = wham_dir
        self.musan_dir = musan_dir
        self.musan_noise_types = musan_noise_types
        self.babble_dir = babble_dir
        self.musdb18_dir = musdb18_dir
        self.fsd50k_dir = fsd50k_dir
        self.DSED_dir = DSED_dir
        self.dsed_selected_classes = dsed_selected_classes
        self.Noise_Equal_Weightage = Noise_Equal_Weightage
        self.control_noise_sourcewise_power = control_noise_sourcewise_power
        
        self.rir_ratio = rir_ratio

        self.comm_delay_chunks = comm_delay_chunks
        self.comm_delay_samples = comm_delay_chunks * chunk_size

        self.reference_channels = [dev1_channels[0], dev2_channels[0]] # Reference channel to compute SNR
        self.dev1_channels = dev1_channels
        self.dev1_channels_to_share = dev1_channels_to_share
        self.dev2_channels = dev2_channels
        self.dev2_channels_to_share = dev2_channels_to_share

        self.use_drc = use_drc
        self.room_mixing = room_mixing
        
        
        self.libritts_dir = libritts_dir
        self.speaker2wavs = {}
        for spk_id in os.listdir(self.libritts_dir):
            spk_dir = os.path.join(self.libritts_dir, spk_id)
            
            wavfile_list = glob.glob(os.path.join(spk_dir, '*/*.wav'))
            
            self.speaker2wavs[spk_id] = wavfile_list
    
        self.speaker_ids = list(self.speaker2wavs.keys())
        
        self.rooms = sorted(os.listdir(dataset_dir))
        print("="*25)
        if use_split_for_rooms is not None:
            
            training_rooms, testing_rooms = train_test_split(self.rooms, test_size=use_split_for_rooms, random_state=42)
            print("using split for room splitting")
            print(f"Training rooms: {len(training_rooms)}, Testing rooms: {len(testing_rooms)}")
            if split == 'train':
                self.rooms = training_rooms
            elif split == 'test':
                self.rooms = testing_rooms
            elif split == 'val':
                self.rooms = testing_rooms
            else:
                assert 0, f"[Room] split must be one of [\'train\', \'test\', \'val\'], found {split}"
        print("Selected Rooms: ", self.rooms)
        print("="*25)
        
        self.room2near = {}
        self.room2far = {}

        if type(boundary_safety) != list:
            boundary_safety = [boundary_safety, boundary_safety]

        # Room-wide dicts
        self.near_samples = {'rir':[], 'audio':[]}
        self.far_samples = {'rir':[], 'audio':[]}
        for room in self.rooms:
            # Check all distances
            room_dir = os.path.join(self.dataset_dir, room)
            
            metadata_path = os.path.join(room_dir, 'metadata.csv')
            df = pd.read_csv(metadata_path)

            # Split samples to near and far
            near_samples = df[df['distance_cm'] <= self.dis_threshold_cm - boundary_safety[0]]
            far_samples = df[df['distance_cm'] >= self.dis_threshold_cm + boundary_safety[1]]
            
            # Add elements to near and far dicts
            self.room2near[room] = {'rir':[], 'audio':[]}
            for i, data in near_samples.iterrows():
                audio_path = audio_path = os.path.join(room_dir, os.path.basename(data['path']))
                sample = sample_descriptor(audio_path, data['angle_deg'], data['distance_cm'], data['speaker_id'])
                if data['audio_type'] == 'a':
                    self.room2near[room]['audio'].append(sample) 
                    self.near_samples['audio'].append(sample)
                elif data['audio_type'] == 'c':
                    self.room2near[room]['rir'].append(sample) 
                    self.near_samples['rir'].append(sample)
                else:
                    assert 0, f'Invalid audio type {data["audio_type"]}'

            self.room2far[room] = {'rir':[], 'audio':[]}
            for i, data in far_samples.iterrows():
                audio_path = audio_path = os.path.join(room_dir, os.path.basename(data['path']))
                sample = sample_descriptor(audio_path, data['angle_deg'], data['distance_cm'], data['speaker_id'])
                if data['audio_type'] == 'a':
                    self.room2far[room]['audio'].append(sample) 
                    self.far_samples['audio'].append(sample)
                elif data['audio_type'] == 'c':
                    self.room2far[room]['rir'].append(sample) 
                    self.far_samples['rir'].append(sample)
                else:
                    assert 0, f'Invalid audio type {data["audio_type"]}'

        print("Room Near")
        for i in self.room2near.keys():
            print(f"room: {i} -> rirs: {len(self.room2near[i]['rir'])} -> audio: {len(self.room2near[i]['audio'])}")

        print("Room Far")
        for i in self.room2far.keys():
            print(f"room: {i} -> rirs: {len(self.room2far[i]['rir'])} -> audio: {len(self.room2far[i]['audio'])}")

        self.near_speakers_range = (near_speakers_min, near_speakers_max)
        self.far_sources_range = (far_sources_min, far_sources_max)
        self.noise_sources_range = (noise_sources_min, noise_sources_max)
        self.noise_prob = noise_probability
        self.snr_range = (snr_min, snr_max)
        self.silence_prob = silence_prob
        self.silence_range = (silence_min, silence_max)
        self.complete_silence_prob = complete_silence_prob

        # Get WHAM! files from this split
        self.wham_samples = []
        if self.noise_prob > 0 and self.wham_dir:
            if split == 'train':
                wham_split = 'tr'
            elif split == 'test':
                wham_split = 'tt'
            elif split == 'val':
                wham_split = 'cv'
            else:
                assert 0, f"split must be one of [\'train\', \'test\', \'val\'], found {split}"
            #wham_samples = wham_meta[wham_meta['WHAM! Split'] == wham_split]['Filename']
            #self.wham_samples = [os.path.join(self.wham_dir, 'audio', x) for x in wham_samples]
            self.wham_samples = glob.glob(os.path.join(self.wham_dir, wham_split)+"/*.wav")
            
        
        
        # Get MUSAN files from this split
        self.musan_samples = []
        if self.noise_prob > 0 and self.musan_dir:
            if split == 'train':
                musan_split = 'train'
            elif split == 'test':
                musan_split = 'test'
            elif split == 'val':
                musan_split = 'val'
            else:
                assert 0, f"split must be one of [\'train\', \'test\', \'val\'], found {split}"
            musan_samples = []
            musan_df = pd.read_csv(os.path.join(self.musan_dir, "metadata.csv"))
            
            
            for m_n_t in self.musan_noise_types:
                musan_files = musan_df[musan_df["filetype"] == f"{musan_split}_{m_n_t}"]["filename"].tolist()
                musan_samples.extend(musan_files)
            self.musan_samples = [os.path.join(self.musan_dir, f) for f in musan_samples]   
        
        
        # Get babble files from this split
        self.babble_samples = []
        if self.noise_prob > 0 and self.babble_dir:
            if split == 'train':
                babble_split = 'train'
            elif split == 'test':
                babble_split = 'test'
            elif split == 'val':
                babble_split = 'val'
            self.babble_samples = glob.glob(os.path.join(self.babble_dir, babble_split)+"/*.wav")
        
        
        self.musdb18_samples = []
        if self.noise_prob > 0 and self.musdb18_dir:
            if split == 'train':
                musdb18_split = 'train'
            elif split == 'test':
                musdb18_split = 'test'
            elif split == 'val':
                musdb18_split = 'test'
            self.musdb18_samples = glob.glob(os.path.join(self.musdb18_dir, musdb18_split)+"/*.wav")
        
        self.fsd50k_samples = []
        if self.noise_prob > 0 and self.fsd50k_dir:
            if split == 'train':
                fsd50k_split = 'FSD50K.dev_audio'
            elif split == 'test':
                fsd50k_split = 'FSD50K.eval_audio'
            elif split == 'val':
                fsd50k_split = 'FSD50K.eval_audio'
            self.fsd50k_samples = glob.glob(os.path.join(self.fsd50k_dir, fsd50k_split)+"/*.wav")
        
        self.dsed_samples = []
        if self.noise_prob > 0 and self.DSED_dir:
            if split == 'train':
                dsed_split = 'train'
            elif split == 'test':
                dsed_split = 'eval'
            elif split == 'val':
                dsed_split = 'eval'
            for dsed_class in self.dsed_selected_classes:
                temp_files = glob.glob(os.path.join(self.DSED_dir, dsed_split, "soundbank","foreground", dsed_class, "*.wav"))
                print("[DSED] ", dsed_class, " -> ", len(temp_files))
                self.dsed_samples = self.dsed_samples + temp_files
            print("[DSED] total files: ", len(self.dsed_samples))

            
        self.noise_samples = self.wham_samples + self.musan_samples + self.babble_samples + self.musdb18_samples + self.fsd50k_samples + self.dsed_samples
        
        print("="*25)
        if self.Noise_Equal_Weightage:
            print("Converting to equal weightage ...")
            rng_for_noise = np.random.RandomState(42)
            max_noise_samples_per_dataset = max(len(self.wham_samples), len(self.musan_samples), len(self.babble_samples), len(self.musdb18_samples), len(self.fsd50k_samples), len(self.dsed_samples))
            
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
            
            if self.DSED_dir and len(self.dsed_samples) < max_noise_samples_per_dataset :
                self.dsed_samples = self.dsed_samples + rng_for_noise.choice(self.dsed_samples, max_noise_samples_per_dataset - len(self.dsed_samples), replace=True).tolist()
            
            self.noise_samples = self.wham_samples + self.musan_samples + self.babble_samples + self.musdb18_samples + self.fsd50k_samples + self.dsed_samples
        
        print(f"Number of wham samples: {len(self.wham_samples)}")
        print(f"Number of MUSAN samples: {len(self.musan_samples)}")
        print(f"Number of babble samples: {len(self.babble_samples)}")
        print(f"Number of musdb18 samples: {len(self.musdb18_samples)}")
        print(f"Number of fsd50k samples: {len(self.fsd50k_samples)}")
        print(f"Number of dsed samples: {len(self.dsed_samples)}")
        print(f"Number of Noise samples: {len(self.noise_samples)}")
        print("Apply low pass filter: ", self.apply_lowpass)
        print("snr_range: ", self.snr_range)
        print("noise_sources_range: ", self.noise_sources_range)
        print("control_noise_sourcewise_power : ", control_noise_sourcewise_power)

        print("="*25)
        
        
        # Data augmentation
        self.augmentations = AudioAugmentations(augmentations)
        
    def get_near_samples(self, room: str):
        if self.room_mixing:
            return self.near_samples
        else:
            return self.room2near[room]

    def get_far_samples(self, room: str):
        if self.room_mixing:
            return self.far_samples
        else:
            return self.room2far[room]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def read_audio_as_mono(self, sf: sf.SoundFile, num_frames: int):
        # Workaround for loading flac files
        if sf.name.endswith('.flac'):
            audio = sf.read(frames=num_frames, dtype='int32')
            audio = (audio / (2 ** (31) - 1)).astype(np.float32)
        else:
            audio = sf.read(frames=num_frames, dtype='float32')
        
        # If multichannel audio take a single channel
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        return audio
    
    def sample_snippet(self, N, audio_file, rng):
        """
        Reads up to N frames from an audio files
        """
        
        with sf.SoundFile(audio_file) as f:
            file_sr = f.samplerate
            
            # N samples in dataset sampling rate
            # N * (file_sr / self.sr) in file sampling rate
            N_file = int(N * file_sr / self.sr)
            
            num_frames = f.frames
    
            # If there aren't enough frames, then read the whole audio. No words will be split.
            if N_file > num_frames:
                # Read entire audio
                audio = self.read_audio_as_mono(f, num_frames=num_frames)
            else:                
                # Randomly choose start of snippet
                start_frame = rng.randint(0, num_frames - N_file + 1) 

                # Get end frame
                end_frame = start_frame + N_file
    
                # Move to start of snippet and read
                f.seek(start_frame)
                audio = self.read_audio_as_mono(f, num_frames=end_frame - start_frame)
        
            # Resample to target sampling rate
            if file_sr != self.sr:
                audio = librosa.resample(audio, orig_sr=f.samplerate, target_sr=self.sr, res_type='polyphase')

        assert audio.shape[-1] <= N, f"Read {audio.shape[-1]} frames, expected at most {N}"

        return audio

    def sample_noise_snippet(self, rng: np.random.RandomState):
        noise_file = rng.choice(self.noise_samples, size=1)[0]
        tgt_frames = int(round(self.sr * self.duration))
        with sf.SoundFile(noise_file) as f:
            num_frames = f.frames

            file_sr = f.samplerate
            
            # TODO: Maybe randomize this?
            N = int(round(file_sr * self.duration))
            
            if N > num_frames:
                # Read entire audio
                snippet = self.read_audio_as_mono(f, num_frames=num_frames)

                # Create a frame of all zeros
                audio = np.zeros(N)

                # Choose a point to add the snippet
                zeros_before = rng.randint(0, N - num_frames)
                audio[zeros_before:zeros_before+num_frames] = snippet
            else:
                # Randomly choose start of snippet
                start_frame = rng.randint(0, num_frames - N + 1)
    
                # Move to start of snippet and read
                f.seek(start_frame)
                audio = self.read_audio_as_mono(f, num_frames=N)
            # print(audio.shape)
            # Resample to target sampling rate
            if file_sr != self.sr:
                audio = librosa.resample(audio, orig_sr=f.samplerate, target_sr=self.sr, res_type='polyphase')

                # Pad zeros if resampling causes off-by-one errors
                if audio.shape[-1] < tgt_frames:
                    audio = np.concatenate([audio, np.zeros(tgt_frames - audio.shape[-1])], axis=-1)

        assert audio.shape[-1] == tgt_frames, f"Expected noise to have {tgt_frames} frames, found {audio.shape[-1]}"
        
        if(np.mean(np.square(audio)) < 1e-5): 
            audio = self.sample_noise_snippet(rng) 
        return audio

    def sample_long_snippet(self, speaker_id, rng: np.random.RandomState):
        num_frames = 0
        min_frames = int(round(self.sr * self.min_active_duration))
        tgt_frames = int(round(self.sr * self.duration))
        
        audio_list = []

        # Sample audio until we have at least min_active_duration amount of frames
        while num_frames < min_frames:
            audio_path = rng.choice(self.speaker2wavs[speaker_id], size=1)[0]
            audio_snippet = self.sample_snippet(tgt_frames - num_frames, audio_path, rng)

            num_frames += audio_snippet.shape[-1]
            audio_list.append(audio_snippet)
        
        audio = np.zeros(tgt_frames)
        
        # Place random amounts of zeros in between the speech segments
        remaining_zeros = tgt_frames - num_frames
        audio_write_idx = 0
        for i in range(len(audio_list)):
            segment = audio_list[i]
            segment_size = segment.shape[-1]

            # If there are no more zeros to place, just put the segment in the next avaliable spot
            if remaining_zeros >= 0:
                zeros_to_place = rng.randint(remaining_zeros, remaining_zeros + 1)
                audio_write_idx += zeros_to_place

                audio[audio_write_idx:audio_write_idx+segment_size] = segment
                audio_write_idx += segment_size
                
                remaining_zeros -= zeros_to_place
            else:
                assert 0, "Code should not go here!"

        assert audio.shape[-1] == tgt_frames, f"Expected final length of audio to be {tgt_frames}, found {audio.shape[-1]}"
        assert remaining_zeros == tgt_frames - audio_write_idx, "There are some zeros that weren't placed"

        return audio
        
    def sample_random_snippet(self, spk_id, rir_dir, rng: np.random.RandomState, is_noise: bool):
        # assert sr == self.sr, f"RIR sampling rate doesn't match. Found {sr}, expected {self.sr}"
        # assert rir.shape[0] == self.n_mics, f"Number of channels in RIR doesn't match. Found {rir.shape[0]}, expected {self.n_mics}"
        
        num_channels = self.n_mics
        tgt_samples = int(round(self.sr * self.duration))
        audio = np.zeros((num_channels, tgt_samples), dtype=np.float32)
        
        # Sample speaker id
        if is_noise:
            single_ch_audio = self.sample_noise_snippet(rng)
            
        else:
            #spk_id = rng.choice(self.speaker_ids, size=1)[0]
            single_ch_audio = self.sample_long_snippet(spk_id, rng)

        left_rir_path = os.path.join(rir_dir, 'left.wav')
        right_rir_path = os.path.join(rir_dir, 'right.wav')
        
        # Read RIRs
        left_rir, sr = sf.read(left_rir_path)
        assert sr == self.sr
        left_rir = left_rir.T
        
        right_rir, sr = sf.read(right_rir_path)
        assert sr == self.sr
        right_rir = right_rir.T

        rir = np.concatenate([left_rir, right_rir], axis=0)

        # TODO: Make it faster
        for i in range(num_channels):
            # print(f"RIR {i}")
            # print(rir[i].shape, single_ch_audio.shape)
            
            RIR = rir[i]
            spatial_audio = signal.convolve(single_ch_audio, RIR, mode='full')
            audio[i] = spatial_audio[: -(len(RIR) - 1)]
        
        assert audio.shape[-1] == tgt_samples

        return audio
    
    def sample_random_snippet_no_spatialization(self, chosen_path, rng: np.random.RandomState):
        left_path = os.path.join(chosen_path, 'left.wav')
        right_path = os.path.join(chosen_path, 'right.wav')
        
        tgt_duration = rng.uniform(self.min_active_duration, self.duration)
        tgt_frames = int(round(self.sr * tgt_duration))
        complete_frames = int(round(self.sr * self.duration))
        
        again = False
        with sf.SoundFile(left_path) as fleft:
            assert fleft.samplerate == self.sr
            
            with sf.SoundFile(right_path) as fright:
                assert fright.samplerate == self.sr

                start = rng.randint(0, fleft.frames - tgt_frames)
                fleft.seek(start)
                fright.seek(start)

                left = fleft.read(tgt_frames).T
                right = fright.read(tgt_frames).T

                snippet = np.concatenate([left, right], axis=0).astype(np.float32)
                
                if complete_frames == tgt_frames:
                    audio = snippet
                else:
                    zeros_before = rng.randint(0, complete_frames - tgt_frames)
                    audio = np.zeros((snippet.shape[0], complete_frames), dtype=np.float32)
                    audio[:, zeros_before:zeros_before+tgt_frames] = snippet

                assert audio.shape[-1] == complete_frames

                # Make sure average power is high enough
                # Average channels
                rms_left = rms(audio[self.reference_channels[0]])
                rms_right = rms(audio[self.reference_channels[0]])
                if min(rms_left, rms_right) < -100:
                    print()
                    print('[WARNING]')
                    print("RMS value is too low, sampling another section")
                    print('Chosen audio', chosen_path)
                    print('RMS', rms(audio[0]))
                    print()
                    again = True

        # Sample another audio if it isn't loud enough
        if again:
            return self.sample_random_snippet_no_spatialization(chosen_path, rng)
        else:
            return audio

    def create_scene(self, sample_idx: int, room: str, rng: np.random.RandomState):
        # Choose inner speakers
        num_near = rng.randint(self.near_speakers_range[0], self.near_speakers_range[1] + 1)
        is_complete_silence = False
        # Choose outer speakers
        num_far = rng.randint(self.far_sources_range[0], self.far_sources_range[1] + 1)


        num_noises = rng.randint(self.noise_sources_range[0], self.noise_sources_range[1] + 1)
        
        if rng.rand() < self.noise_prob:
            # Use noise source
            num_noises += 1

        audio_list = []
        chosen_paths = []

        # Get near & far dicts
        near_dict = self.get_near_samples(room)
        far_dict = self.get_far_samples(room)

        # With replacement
        # if self.split == 'train':
        near_rir_order = rng.choice(near_dict['rir'], size=num_near, replace=True).tolist()
        far_rir_order = rng.choice(far_dict['rir'], size=num_far + num_noises, replace=True).tolist()
        # else:
        #     near_rir_order = []
        #     far_rir_order = rng.choice(far_dict['rir'], size=num_far + num_noises, replace=True).tolist()
        
        rir_ratio = self.rir_ratio
        if len(near_dict['audio']) > 0:
            near_audio_order = rng.choice(near_dict['audio'], size=num_near, replace=True).tolist()
            far_audio_order = rng.choice(far_dict['audio'], size=num_far, replace=True).tolist()
        else:
            near_audio_order = []
            far_audio_order = []
            rir_ratio = 1
        # print("rir order: ", len(near_rir_order), len(far_rir_order))
        # print("audio order: ", len(near_audio_order), len(far_audio_order))
        # print(num_near, num_far, num_noises)
        # Load near, far and noise into target audio
        angles = []
        distances = []
        selected_speaker_ids = rng.choice(self.speaker_ids, size=(num_near + num_far + num_noises))
        for i in range(num_near + num_far + num_noises):
            audio_dict = None

            is_noise = False
            if i in range(num_near):
                # near
                if self.split == 'train':
                    rir_p = rng.uniform(0,1)
                else:
                    rir_p = 1
                audio_order = near_audio_order
                rir_order = near_rir_order
            elif i in range(num_near, num_near + num_far):
                # far
                if self.split == 'train':
                    rir_p = rng.uniform(0,1)
                else:
                    rir_p = 1
                audio_order = far_audio_order
                rir_order = far_rir_order
            else:
                # Noise
                is_noise = True
                rir_p = 0
                audio_order = far_audio_order
                rir_order = far_rir_order
            
            # Choose sample
            chosen_sample: sample_descriptor
            # print(i, self.split, rir_p, len(rir_order), self.rir_ratio, num_near , num_far , num_noises)
            
            if rir_p <= rir_ratio:
                # Choose unique RIR
                
                chosen_sample = rir_order[-1]
                rir_order.pop()

                
                spatialized_audio = self.sample_random_snippet(selected_speaker_ids[i], chosen_sample.path, rng, is_noise=is_noise)
                #print("training, id:", target_speaker_id)
            else:
                
                # if (i < num_near) and (num_near==2):
                #     print('Near', i, near_audio_order)
                # Choose unique audio
                chosen_sample = audio_order[-1]
                audio_order.pop()
                
                spatialized_audio = self.sample_random_snippet_no_spatialization(chosen_sample.path, rng)
            
            if i < (num_near + num_far) and (rng.rand() < self.silence_prob):
                selected_silence_duration = rng.uniform(self.silence_range[0], self.silence_range[1])
                selected_silence_frames = int(round(selected_silence_duration * self.sr))
                total_frames = int(round(self.sr * self.duration))
                starting_silence_frame = rng.randint(0, total_frames - selected_silence_frames+1)
                temp_spatialized_audio = spatialized_audio.copy()
                temp_spatialized_audio[:,starting_silence_frame:starting_silence_frame+selected_silence_frames] = 0
                if np.mean((temp_spatialized_audio)**2) > 10e-6:
                    #print("added silence")
                    spatialized_audio = temp_spatialized_audio
                else:
                    #print("------ not adding silence")
                    spatialized_audio = spatialized_audio
            
            if (i < num_near) and (rng.rand() < self.complete_silence_prob):
                spatialized_audio = 0*spatialized_audio.copy()
                is_complete_silence = True
            else: 
                # Store metadata
                distances.append(chosen_sample.distance)
                angles.append(chosen_sample.angle)
                chosen_paths.append(chosen_sample.path)
                
                audio_list.append(torch.from_numpy(spatialized_audio))
                assert (torch.isnan(torch.from_numpy(spatialized_audio)).max() == 0)
        
        if is_complete_silence:
            num_near = 0
        
        # Randomly scale each audio event
        for i in range(len(audio_list)):
            scale = rng.uniform(0.1, 1)
            audio_list[i] = (audio_list[i] / np.abs(audio_list[i]).max()) * scale

        if self.control_noise_sourcewise_power:
            ScalingForNoise = [0.8, 0.7, 0.1, 0.05, 0.01, 0.005, 0.001, 0.001]
            ccc = 0
            for i in range(num_near, len(audio_list)):
                audio_list[i] = (audio_list[i] / np.abs(audio_list[i]).max()) * ScalingForNoise[ccc]
                ccc+=1



        noise = sum(audio_list[num_near:])
        tgt = torch.zeros_like(noise)

        
        
        
        
        # Compute the amount to scale the noise by so that the SNR is uniformly distributed
        target_snr = None
        if num_near > 0:
            target_snr = rng.uniform(self.snr_range[0], self.snr_range[1])
            snr_scale = get_scale_for_snr(target_snr, audio_list, num_near, self.reference_channels)
            
            noise = noise * snr_scale

            tgt = sum(audio_list[:num_near])
        
        # Get the target and mixture audio
        #print("target snr: ", target_snr)
        if (self.apply_lowpass is not None) and (rng.rand() < self.apply_lowpass["prob"]):
            filtered_tgt = tgt.clone()
            selected_cutoff_freq = rng.uniform(self.apply_lowpass["min_freq"], self.apply_lowpass["max_freq"])
            filtered_tgt = brickwall_lowpass_stft(filtered_tgt, sr=self.sr, cutoff=selected_cutoff_freq)   
            mixture = noise + filtered_tgt
        else:
            #print("without filter")
            mixture = noise + tgt
        
        #print("tgt:", tgt.shape)
        
        # Scale mixture so that peak is <= 1
        peak = torch.abs(mixture).max()
        if peak > 1:
            mixture /= peak
            tgt /= peak
        
        assert (torch.isnan(mixture).max() == 0), "gt tensor has nan!"
        
        return mixture, tgt, num_near, num_far, num_noises,\
                audio_list, target_snr, chosen_paths, angles, distances

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def split_and_share(self, mixture: torch.Tensor):
        comm_delay_samples = self.comm_delay_samples

        # Split audio into device 1 and device 2 audio
        x_11 = mixture[self.dev1_channels]
        x_12 = mixture[self.dev2_channels_to_share]
        x_12 = torch.roll(x_12, comm_delay_samples, dims=-1)
        x_12[..., :comm_delay_samples] = 0 # Set wrapped samples to zero
        
        x_22 = mixture[self.dev2_channels]
        x_21 = mixture[self.dev1_channels_to_share]
        x_21 = torch.roll(x_21, comm_delay_samples, dims=-1)
        x_21[..., :comm_delay_samples] = 0 # Set wrapped samples to zero        

        return x_11, x_12, x_21, x_22 


    def __getitem__(self, idx: int, seed=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if seed is None:
            if self.split == 'train':
                # IT IS ACTUALLY **** EXTREMELY **** IMPORTANT TO ADD IDX, ESPECIALLY IF WE ARE FIXING THE WORKERS SEEDS
                # OTHERWISE ALL WORKERS WILL HAVE THE SAME SEED!!!
                seed = idx + self.epoch * len(self)
            else:
                seed = idx
        rng = np.random.RandomState(seed)

        # Choose room
        room = rng.choice(self.rooms, size=1)[0]

        # Create scene
        mixture: torch.Tensor
        gt: torch.Tensor
        mixture, gt, num_tgt_speakers, num_far_speakers, num_noises, audio_list, target_snr, chosen_paths, speaker_angles, speaker_distances = \
            self.create_scene(idx, room, rng)
        assert len(audio_list) == num_tgt_speakers + num_far_speakers + num_noises

        # Sanity check
        if num_tgt_speakers > 0:
            assert torch.abs(gt).max() > 0

        # Share audio between devices
        x_11, x_12, x_21, x_22 = self.split_and_share(mixture)
        gt1 = gt[self.reference_channels[0]].unsqueeze(0)
        gt2 = gt[self.reference_channels[1]].unsqueeze(0)

        # Add dummy sample discriptors to get fixed size tensors
        for i in range(self.near_speakers_range[1] + self.far_sources_range[1] - len(chosen_paths)):
            chosen_paths.append('None')
            speaker_distances.append(-1)
            speaker_angles.append(-1)
        speaker_distances = torch.Tensor(speaker_distances)
        speaker_angles = torch.Tensor(speaker_angles)

        # Apply perturbations to entire audio
        if self.split == 'train':
            # Apply aug on dev 1
            mixture = torch.cat([x_11, x_12], dim=0)
            num_ch = x_11.shape[0]
            x_1, gt1 = self.augmentations.apply_random_augmentations(mixture, gt1, rng)
            x_11 = x_1[:num_ch]
            x_12 = x_1[num_ch:]

            # Apply aug on dev 2
            mixture = torch.cat([x_22, x_21], dim=0)
            x_2, gt2 = self.augmentations.apply_random_augmentations(mixture, gt2, rng)
            num_ch = x_22.shape[0]
            x_22 = x_2[:num_ch]
            x_21 = x_2[num_ch:]

        # Apply DRC
        if self.use_drc:
            avg_sign = (x_11[0] + x_12[0]) / 2
            g1, _ = drc(avg_sign.numpy(), 0.5)
            g1 = torch.from_numpy(g1).unsqueeze(0)
            # x_11 = g1 * x_11
            # x_12 = g1 * x_12
            # gt1 = g1 * gt1

            avg_sign = (x_22[0] + x_21[0]) / 2
            g2, _ = drc(avg_sign.numpy(), 0.5)
            g2 = torch.from_numpy(g2).unsqueeze(0)

            # x_22 = g2 * x_22
            # x_21 = g2 * x_21
            # gt2 = g2 * gt2
        else:
            g1 = torch.ones((1, x_11.shape[-1]))
            g2 = torch.ones((1, x_22.shape[-1]))
        
        if (target_snr is not None) and (len(self.augmentations) == 0) and (not self.use_drc):
            actual_snr = get_snr(torch.cat([gt1, gt2]), torch.stack([x_11[0], x_22[0]]))
            assert torch.abs(actual_snr - target_snr) < 0.1
        
        # Define inputs
        inputs = {
            'original_mixture':mixture,
            'audio_at_dev1_from_dev1':x_11,
            'audio_at_dev1_from_dev2':x_12,
            'audio_at_dev2_from_dev1':x_21,
            'audio_at_dev2_from_dev2':x_22,
            'comm_delay_chunks': self.comm_delay_chunks,


            'drc_gain1':g1,
            'drc_gain2':g2,

            'seed':seed,
        }
        # print(num_tgt_speakers, num_noises, self.comm_delay_chunks, num_tgt_speakers, num_far_speakers)
        # Define targets
        targets = {
            'target_at_dev1':gt1,
            'target_at_dev2':gt2,
            # 'sin_basis_vectors':sin_basis_vectors,
            # 'sin_split':[self.near_speakers_range[1], self.far_speakers_range[1] , self.noises_range[1]],
            'num_target_speakers':num_tgt_speakers,
            'num_interfering_speakers':num_far_speakers,
            'num_noises':num_noises,
            #'chosen_paths': chosen_paths,
            #'angles':speaker_angles,
            #'distances':speaker_distances,
            #'room':room,
        }
        
        return inputs, targets
    

def save_wav(outdir, inputs, targets, input_keys, target_keys, sr):
    os.makedirs(outdir, exist_ok=True)
    
    
    for k in input_keys:
        outfile = f"{outdir}/{k}.wav"
        x = inputs[k]
        x = x.T
        x = x.numpy()
        print(k, " => ", x.shape)
        sf.write(outfile, x, sr)
    
    for k in target_keys:
        outfile = f"{outdir}/{k}.wav"
        x = targets[k]
        x = x.T
        x = x.numpy()
        print(k, " => ", x.shape)
        sf.write(outfile, x, sr)



if __name__ == "__main__":
    train_data_args = {
        "dataset_dir":"/home/ubuntu/Hearvana/datasets/parsed_earbud_human_data/parsed_earbud_human_data/train", 
        "libritts_dir":"/home/ubuntu/Hearvana/datasets/TfmlpnetData/LibriTTS/train-clean-360",
        "use_split_for_rooms": 0.15,
        "wham_dir": "/home/ubuntu/Hearvana/datasets/NoiseDataSets/wham_noise",
        "musan_dir": "/home/ubuntu/Hearvana/datasets/NoiseDataSets/musan",
        "musan_noise_types":["noise", "music"],
        "babble_dir": "/home/ubuntu/Hearvana/datasets/NoiseDataSets/babble",
        "musdb18_dir":"/home/ubuntu/Hearvana/datasets/NoiseDataSets/musdb18_data",
        "fsd50k_dir": "/home/ubuntu/Hearvana/datasets/NoiseDataSets/FSD50K",
        "DSED_dir" : "/home/ubuntu/Hearvana/datasets/NoiseDataSets/DSED/synthetic/audio",
        "dsed_selected_classes" : ["Alarm_bell_ringing", "Blender", "Electric_shaver_toothbrush", "Vacuum_cleaner"],
        "Noise_Equal_Weightage":True,
        "samples_per_epoch":20000,
        "n_mics":8,
        "sr":16000,
        "min_active_duration":2,
        "duration":5,
        "dis_threshold":1.5,
        "near_speakers_min":1, 
        "near_speakers_max":1,
        "far_sources_min":0,
        "far_sources_max":0,
        "noise_sources_min":1,
        "noise_sources_max":6,
        "noise_probability":0.0001, # Probability that a far source is a noise source
        "snr_min":-10, 
        "snr_max":5,
        "silence_prob": 0,
        "silence_min": 1, 
        "silence_max": 4,
        "complete_silence_prob": 0.1,
        "far_noise_only" : True,
        "augmentations" : [],
        "comm_delay_chunks" : 6,
        "chunk_size" : 96,
        "dev1_channels" : [0,1,2],
        "dev1_channels_to_share" : [7],
        "dev2_channels" : [4,5,6],
        "dev2_channels_to_share" : [3],
        "boundary_safety":-25000000,
        "rir_ratio":1, # 80% RIR data
        "room_mixing" : False, # Use multiple rooms to create mixtures
        "use_drc" : True, # Apply DRC on mixture
    }

    val_data_args = {
        "dataset_dir":"/home/ubuntu/Hearvana/datasets/parsed_earbud_human_data/parsed_earbud_human_data/train", 
        "libritts_dir":"/home/ubuntu/Hearvana/datasets/TfmlpnetData/LibriTTS/train-clean-360",
        "use_split_for_rooms": 0.15,
        "wham_dir": "/home/ubuntu/Hearvana/datasets/NoiseDataSets/wham_noise",
        "musan_dir": "/home/ubuntu/Hearvana/datasets/NoiseDataSets/musan",
        "musan_noise_types":["noise", "music"],
        "babble_dir": "/home/ubuntu/Hearvana/datasets/NoiseDataSets/babble",
        "musdb18_dir":"/home/ubuntu/Hearvana/datasets/NoiseDataSets/musdb18_data",
        "fsd50k_dir": "/home/ubuntu/Hearvana/datasets/NoiseDataSets/FSD50K",
        "DSED_dir" : "/home/ubuntu/Hearvana/datasets/NoiseDataSets/DSED/synthetic/audio",
        "dsed_selected_classes" : ["Alarm_bell_ringing", "Blender", "Electric_shaver_toothbrush", "Vacuum_cleaner"],
        "Noise_Equal_Weightage":True,
        "samples_per_epoch":2000,
        "n_mics":8,
        "sr":16000,
        "min_active_duration":2,
        "duration":5,
        "dis_threshold":1.5,
        "near_speakers_min":1, 
        "near_speakers_max":1,
        "far_sources_min":0,
        "far_sources_max":0,
        "noise_sources_min":1,
        "noise_sources_max":6,
        "noise_probability":0.0001, # Probability that a far source is a noise source
        "snr_min":-5, 
        "snr_max":5,
        "silence_prob": 0,
        "silence_min": 1, 
        "silence_max": 4,
        "complete_silence_prob": 0,
        "far_noise_only" : True,
        "augmentations" : [],
        "comm_delay_chunks" : 6,
        "chunk_size" : 96,
        "dev1_channels" : [0,1,2],
        "dev1_channels_to_share" : [7],
        "dev2_channels" : [4,5,6],
        "dev2_channels_to_share" : [3],
        "boundary_safety":-25000000,
        "rir_ratio":0, # 80% RIR data
        "room_mixing" : False, # Use multiple rooms to create mixtures
        "use_drc" : True, # Apply DRC on mixture
    }
    
    # data_loader = Dataset(**train_data_args, split='train')
    # outdir = "/home/ubuntu/Hearvana/datasets/TestingSE_Dataset_New/train"

    data_loader = Dataset(**val_data_args, split='val')
    #outdir = "/home/ubuntu/Hearvana/datasets/TestingSE_Dataset_New/val"
    # data_loader = torch.utils.data.DataLoader(temp_loader,
    #                                            batch_size=4,
    #                                            shuffle=False)
    
    
    
    print("creation complete")
    ii = 0
    print("length: ", len(data_loader))

    # input_keys = ['audio_at_dev1_from_dev1',
    #         'audio_at_dev1_from_dev2',
    #         'audio_at_dev2_from_dev1',
    #         'audio_at_dev2_from_dev2']
    # target_keys = ['target_at_dev1',
    #         'target_at_dev2']
    
    for batch_idx, b in enumerate(data_loader):
        inputs, targets = b
        '''
        'original_mixture':mixture,
            'audio_at_dev1_from_dev1':x_11,
            'audio_at_dev1_from_dev2':x_12,
            'audio_at_dev2_from_dev1':x_21,
            'audio_at_dev2_from_dev2':x_22,
            'comm_delay_chunks': self.comm_delay_chunks,
            'target_at_dev1':gt1,
            'target_at_dev2':gt2,
        '''
        
        #save_wav(os.path.join(outdir, f"sample{batch_idx}"), inputs, targets, input_keys, target_keys, sr=16000)
        
        for i in inputs.keys():
            if i not in ["comm_delay_chunks", "seed"]:
                print(f"{i} -> {inputs[i].shape}, -> {inputs[i].dtype}")
        for i in targets.keys():
            if i not in ["comm_delay_chunks", "seed"]:
                print(f"{i} -> {targets[i].shape}, -> {targets[i].dtype}")
        print(batch_idx)
        print("="*25)
        break
        if(batch_idx > 10):
            break