"""
Torch dataset object for synthetically rendered spatial data - ASR focused version.
Loads full near audio utterances (resampled if longer than max_duration),
with transcript support. No interleaving or left/right swap augmentation.
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
from torchmetrics.functional.audio import signal_noise_ratio as snr

import scipy.signal as signal
import pandas as pd
import librosa


def compute_left_right_power_difference(audio, reference_channels=-1):
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
    return snr(mixture, target).mean()

def get_scale_for_snr(target_snr, tgt, noise, reference_channels=None):
    denoised_tgt = tgt
    denoised_mix = tgt + noise
    if reference_channels is not None:
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
    frame_duration = 10  # ms
    nchannels = 1
    bitrate = 16000  # 16kbps
    bitdepth = None  # Float32
    encoder = lc3.Encoder(int(frame_duration)*1000, sr, nchannels)
    frame_size = encoder.get_frame_bytes(bitrate)
    frame_length = encoder.get_frame_samples()
    bitrate = encoder.resolve_bitrate(frame_size)

    # Pad to multiple of frame_length so the last frame is always full
    original_len = audio.shape[-1]
    remainder = original_len % frame_length
    if remainder != 0:
        audio = np.pad(audio, (0, frame_length - remainder))

    encoded = []
    for i in range(0, audio.shape[-1], frame_length):
        chunk = audio[..., i:i+frame_length]
        encoded_bytes = encoder.encode(chunk, frame_size)
        encoded.append(encoded_bytes)
    decoder = lc3.Decoder(int(frame_duration)*1000, sr, nchannels)
    decoded_audio_chunks = []
    for enc_b in encoded:
        chunk = decoder.decode(enc_b, bitdepth)
        decoded_audio_chunks.append(chunk)
    decoded_audio = np.concatenate(decoded_audio_chunks)
    return decoded_audio[:original_len]

def apply_codec_artifacts_multichannel(audio, sr):
    assert len(audio.shape) == 2, "Multichannel audio only"
    num_channels = audio.shape[0]
    outs = []
    for i in range(num_channels):
        output_audio = apply_codec_artifacts(audio[i], sr)
        outs.append(output_audio)
    outs = np.stack(outs, axis=0)
    return outs


class Dataset(torch.utils.data.Dataset):
    """
    Dataset for ASR self-speech separation.
    Near audio is loaded as a full utterance (resampled to max_duration if too long).
    Interference is chunked if longer than near audio, zero-padded if shorter.
    Transcript is loaded from transcript.txt in the sample folder.
    """
    def __init__(self,
            self_speech_dir: Dict[str, float],
            interference_dir: Dict[str, float],
            noise_dir_config,
            split,
            samples_per_epoch,
            sr=16000,
            max_duration=40,
            augmentations=[],
            used_channels=[0, 3],
            far_sources_range=[0, 4],
            snr_min=-10, snr_max=5,
            use_drc=False,
            noise_prob=0.0,
            compression_aug_ratio=0.0,
            negative_ratio=0.2,
            interleaving_prob=0.0,
            **args,
        ):
        super().__init__()

        self.max_duration = int(max_duration * sr)
        self.sr = sr
        self.snr_range = (snr_min, snr_max)
        self.used_channels = used_channels
        self.use_drc = use_drc
        self.epoch = 0

        self.self_speech_dir = self_speech_dir
        self.near_samples = []
        self.far_samples = []
        self.samples_per_epoch = samples_per_epoch
        self.split = split
        self.far_sources_range = far_sources_range

        self.target_rooms = []
        self.interf_rooms = []

        self.room2near = {}
        self.room2far = {}
        self.path2already_compressed = {}

        self.compression_aug_ratio = compression_aug_ratio
        self.noise_prob = noise_prob
        self.negative_ratio = negative_ratio
        self.interleaving_prob = interleaving_prob
        print("self.interleaving_prob - ", self.interleaving_prob)

        if noise_prob > 0:
            self.wham_dir = noise_dir_config.get('wham_dir', None)
            self.musan_dir = noise_dir_config.get('musan_dir', None)
            self.musan_noise_types = noise_dir_config.get('musan_noise_types', None)
            self.babble_dir = noise_dir_config.get('babble_dir', None)
            self.musdb18_dir = noise_dir_config.get('musdb18_dir', None)
            self.fsd50k_dir = noise_dir_config.get('fsd50k_dir', None)
            self.noise_samples = self.get_noise_samples()
        else:
            self.noise_samples = []

        for dataset_dir, sampling_prob in self_speech_dir.items():
            near_room_dict = {"audio": [], "prob": sampling_prob}
            rooms = get_subdirs(dataset_dir)
            self.target_rooms.extend(rooms)

            for room in rooms:
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
                        self.room2far[room]['audio'].append(audio_path)
                        far_room_dict['audio'].append(audio_path)
                    elif data['audio_type'] == 'c':
                        distance = data['distance_cm']
                        if distance < 60:
                            print(f"Skipping RIR at distance {distance} cm because distance is too close for an interference source", data['path'])
                            continue
                        rir_path = os.path.join(room_dir, os.path.basename(data['path']))
                        self.room2far[room]['rir'].append(rir_path)
                        far_room_dict['rir'].append(rir_path)

            self.far_samples.append(far_room_dict)

        print("*"*25)
        for near_dataset in self.near_samples:
            print("Dataset1: ", near_dataset['prob'], len(near_dataset['audio']))
        for i in self.room2near.keys():
            print(f"room: {i} audio: {len(self.room2near[i]['audio'])}")
        print("*"*25)
        for far_dataset in self.far_samples:
            print("Dataset2: ", far_dataset['prob'], len(far_dataset['audio']), len(far_dataset['rir']))
        for i in self.room2far.keys():
            print(f"room: {i} rirs: {len(self.room2far[i]['rir'])}, audio: {len(self.room2far[i]['audio'])}")
        print("="*25)

        self.augmentations = AudioAugmentations(augmentations)

    def get_noise_samples(self):
        self.wham_samples = []
        if self.noise_prob > 0 and self.wham_dir:
            if self.split == 'train':
                wham_split = 'tr'
            elif self.split == 'test':
                wham_split = 'tt'
            elif self.split == 'val':
                wham_split = 'cv'
            else:
                assert 0, f"split must be one of ['train', 'test', 'val'], found {self.split}"
            self.wham_samples = glob.glob(os.path.join(self.wham_dir, wham_split)+"/*.wav")

        self.musan_samples = []
        if self.noise_prob > 0 and self.musan_dir:
            if self.split == 'train':
                musan_split = 'train'
            elif self.split == 'test':
                musan_split = 'test'
            elif self.split == 'val':
                musan_split = 'val'
            else:
                assert 0, f"split must be one of ['train', 'test', 'val'], found {self.split}"
            musan_samples = []
            musan_df = pd.read_csv(os.path.join(self.musan_dir, "metadata.csv"))
            for m_n_t in self.musan_noise_types:
                musan_files = musan_df[musan_df["filetype"] == f"{musan_split}_{m_n_t}"]["filename"].tolist()
                musan_samples.extend(musan_files)
            self.musan_samples = [os.path.join(self.musan_dir, f) for f in musan_samples]

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

        print("="*25)
        print("Converting to equal weightage ...")
        rng_for_noise = np.random.RandomState(42)
        max_noise_samples_per_dataset = max(
            len(self.wham_samples), len(self.musan_samples),
            len(self.babble_samples), len(self.musdb18_samples), len(self.fsd50k_samples)
        )

        if self.wham_dir and len(self.wham_samples) < max_noise_samples_per_dataset:
            self.wham_samples = self.wham_samples + rng_for_noise.choice(self.wham_samples, max_noise_samples_per_dataset - len(self.wham_samples), replace=True).tolist()
        if self.musan_dir and len(self.musan_samples) < max_noise_samples_per_dataset:
            self.musan_samples = self.musan_samples + rng_for_noise.choice(self.musan_samples, max_noise_samples_per_dataset - len(self.musan_samples), replace=True).tolist()
        if self.babble_dir and len(self.babble_samples) < max_noise_samples_per_dataset:
            self.babble_samples = self.babble_samples + rng_for_noise.choice(self.babble_samples, max_noise_samples_per_dataset - len(self.babble_samples), replace=True).tolist()
        if self.musdb18_dir and len(self.musdb18_samples) < max_noise_samples_per_dataset:
            self.musdb18_samples = self.musdb18_samples + rng_for_noise.choice(self.musdb18_samples, max_noise_samples_per_dataset - len(self.musdb18_samples), replace=True).tolist()
        if self.fsd50k_dir and len(self.fsd50k_samples) < max_noise_samples_per_dataset:
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
            left_path = os.path.join(chosen_path, 'right.wav')
            right_path = os.path.join(chosen_path, 'left.wav')
        else:
            left_path = os.path.join(chosen_path, 'left.wav')
            right_path = os.path.join(chosen_path, 'right.wav')

        left_audio, sr = librosa.load(left_path, sr=None, mono=False)
        right_audio, sr = librosa.load(right_path, sr=None, mono=False)
        assert sr == self.sr

        assert left_audio.shape[-1] == right_audio.shape[-1]
        assert len(left_audio.shape) == 2, f"weird audio shape {left_audio.shape} at {chosen_path}"
        assert len(right_audio.shape) == 2, f"weird audio shape {right_audio.shape} at {chosen_path}"

        audio = np.concatenate([left_audio, right_audio], axis=0)
        # Pad silence at the beginning (0.1s–1s) so the model sees a quiet lead-in
        SAFE_GUARD_AUDIO = rng.randint(3200, 16000)
        SAFE_GUARD_AUDIO2 = rng.randint(1600, 10000)
        audio = np.pad(audio, ((0, 0), (SAFE_GUARD_AUDIO, SAFE_GUARD_AUDIO2)))

        return audio

    def load_noise(self, chosen_path, target_len: int, rng: np.random.RandomState):
        left_audio, sr = librosa.load(chosen_path, sr=self.sr, mono=False)
        assert sr == self.sr

        if len(left_audio.shape) == 1:
            left_audio = left_audio[np.newaxis, :]
        assert len(left_audio.shape) == 2, f"weird audio shape {left_audio.shape} at {chosen_path}"

        if left_audio.shape[-1] < target_len:
            zero_pad = target_len - left_audio.shape[-1]
            front_pad = rng.randint(0, zero_pad + 1)
            back_pad = zero_pad - front_pad
            left_audio = np.pad(left_audio, ((0, 0), (front_pad, back_pad)))
        elif left_audio.shape[-1] > target_len:
            delta = left_audio.shape[-1] - target_len
            start_id = rng.randint(0, delta)
            left_audio = left_audio[:, start_id:start_id + target_len]

        return left_audio[0]

    def load_audio_from_rir(self, chosen_path, target_len: int, rng: np.random.RandomState):
        noise_sample = rng.choice(self.noise_samples, size=1, replace=True).tolist()
        single_ch_noise = self.load_noise(noise_sample[0], target_len, rng)

        if "/parsed_earbud_human_data/" in chosen_path:
            left_rir_path = os.path.join(chosen_path, 'right.wav')
            right_rir_path = os.path.join(chosen_path, 'left.wav')
        else:
            left_rir_path = os.path.join(chosen_path, 'left.wav')
            right_rir_path = os.path.join(chosen_path, 'right.wav')

        left_rir, sr = librosa.load(left_rir_path, sr=None, mono=False)
        right_rir, sr = librosa.load(right_rir_path, sr=None, mono=False)
        assert sr == self.sr

        assert left_rir.shape[-1] == right_rir.shape[-1]
        assert len(left_rir.shape) == 2, f"weird audio shape {left_rir.shape} at {chosen_path}"
        assert len(right_rir.shape) == 2, f"weird audio shape {right_rir.shape} at {chosen_path}"

        rir = np.concatenate([left_rir, right_rir], axis=0)
        num_channels = rir.shape[0]
        audio = []
        for i in range(num_channels):
            RIR = rir[i]
            spatial_audio = signal.convolve(single_ch_noise, RIR, mode='full')
            audio.append(spatial_audio[:-(len(RIR) - 1)])
        audio = np.stack(audio, axis=0)

        # Trim or pad convolution output to target_len
        if audio.shape[-1] > target_len:
            audio = audio[:, :target_len]
        elif audio.shape[-1] < target_len:
            audio = np.pad(audio, ((0, 0), (0, target_len - audio.shape[-1])))

        return audio

    def adjust_far_audio_length(self, audio: torch.Tensor, target_len: int, rng: np.random.RandomState) -> torch.Tensor:
        """Chunk far audio if longer than target, zero-pad if shorter."""
        if audio.shape[-1] > target_len:
            delta = audio.shape[-1] - target_len
            start_id = rng.randint(0, delta)
            audio = audio[:, start_id:start_id + target_len]
        elif audio.shape[-1] < target_len:
            pad_amount = target_len - audio.shape[-1]
            front_pad = rng.randint(0, pad_amount + 1)
            back_pad = pad_amount - front_pad
            audio = torch.nn.functional.pad(audio, (front_pad, back_pad))
        return audio

    def sample_near_audio(self, num_near, num_far, rng: np.random.RandomState):
        near_probs = np.array([d['prob'] for d in self.near_samples])
        near_probs = near_probs / near_probs.sum()
        near_dataset = rng.choice(self.near_samples, size=1, replace=True, p=near_probs).tolist()[0]
        near_audio_order = rng.choice(near_dataset['audio'], size=1, replace=True).tolist()

        far_probs = np.array([d['prob'] for d in self.far_samples])
        far_probs = far_probs / far_probs.sum()
        far_dataset = rng.choice(self.far_samples, size=num_far, replace=True, p=far_probs).tolist()
        far_audio_order = []
        for dataset in far_dataset:
            select_path = rng.choice(dataset['audio'], size=1, replace=True).tolist()
            far_audio_order.append(select_path[0])

        return near_audio_order, far_audio_order

    def interleave_near_and_far(self, tgt: torch.Tensor, far_raw_list: list, rng: np.random.RandomState):
        """
        Interleave near audio and interference sequentially within max_duration,
        mirroring the 70% sequential branch of mixing_near_and_far_with_interleaving
        in _new dataset. Near audio is kept at its full utterance length (no chunking).
        Returns (tgt_buf, noise_buf) of shape [channels, total_duration] where
        total_duration is randomized in [near_len+1s, max_duration],
        or (None, None) if there is insufficient space for min 1s of interference.
        """
        num_channels = tgt.shape[0]
        targe_duration = tgt.shape[-1]
        min_active_duration = int(self.sr * 1)  # 1 second minimum for noise

        # Randomize total duration: near_len + 1s  →  max_duration
        min_total = targe_duration + min_active_duration
        if min_total > self.max_duration:
            return None, None  # near audio leaves no room for 1s of noise

        total_duration = rng.randint(min_total, self.max_duration + 1)

        interval = int(rng.uniform(-1, 1) * self.sr)  # gap in samples (can be negative)
        noise_duration = total_duration - targe_duration - interval

        if noise_duration < min_active_duration:
            return None, None  # interval pushed noise too short; caller falls back to overlap

        far_adjusted = [self.adjust_far_audio_length(f, noise_duration, rng) for f in far_raw_list]
        noise_chunk = sum(far_adjusted)

        tgt_buf   = torch.zeros(num_channels, total_duration)
        noise_buf = torch.zeros(num_channels, total_duration)

        order_prob = rng.uniform(0, 1)
        if order_prob < 0.5:
            # near first, noise at the far end (same placement as _new)
            tgt_buf[:, :targe_duration] = tgt
            noise_buf[:, -noise_duration:] = noise_chunk
        else:
            # noise first, near at the far end
            noise_buf[:, :noise_duration] = noise_chunk
            tgt_buf[:, -targe_duration:] = tgt

        return tgt_buf, noise_buf

    def create_scene(self, idx: int, rng: np.random.RandomState, used_channels_final: list) -> Tuple:
        num_near = 1
        num_far = rng.randint(self.far_sources_range[0], self.far_sources_range[1] + 1)

        near_audio_order, far_audio_order = self.sample_near_audio(num_near, num_far, rng)

        # Negative sample: replace near audio with 5s of silence
        near_already_compressed = False
        room_name = near_audio_order[0]
        transcript = ""

        if rng.uniform(0, 1) < self.negative_ratio:
            num_channels = 8
            near_audio_tensor = torch.zeros(num_channels, int(5 * self.sr))
            num_near = 0
        else:
            near_audio_tensor = None

        # Load full near audio; retry with a new sample if duration exceeds max_duration
        MAX_RETRIES = 5
        for attempt in range(MAX_RETRIES):
            if near_audio_tensor is not None:
                break
            near_path = near_audio_order[0]
            near_already_compressed = self.path2already_compressed.get(near_path, False)
            room_name = near_path

            transcript_path = os.path.join(near_path, 'transcript.txt')
            transcript = ""
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r') as f:
                    transcript = f.read().strip()

            near_audio = self.load_audio(near_path, rng)  # [channels, samples]
            max_val = np.abs(near_audio).max()

            if max_val < 1e-9:
                print(f"[Warning] near audio loaded power is very small, setting to zeros")
                near_audio_tensor = torch.from_numpy(np.zeros_like(near_audio))
                num_near = 0
                transcript = ""
                break

            scale = rng.uniform(0.1, 1)
            near_audio = (near_audio / max_val) * scale

            # SAFE_GUARD = 160
            # _, indexes = librosa.effects.trim(near_audio, top_db=20)
            # start_i = max(0, indexes[0] - SAFE_GUARD)
            # end_i = min(near_audio.shape[-1], indexes[1] + SAFE_GUARD)
            # near_audio = near_audio[:, start_i:end_i]

            if near_audio.shape[-1] > self.max_duration:
                print(f"[Warning] near audio duration {near_audio.shape[-1] / self.sr:.2f}s exceeds max_duration, retrying with next sample")
                near_audio_order, _ = self.sample_near_audio(num_near, 0, rng)
                continue

            near_audio_tensor = torch.from_numpy(near_audio)
            break

        if near_audio_tensor is None:
            print(f"[Warning] could not find near audio within max_duration after {MAX_RETRIES} attempts, using all zeros")
            num_channels = near_audio.shape[0]
            near_audio_tensor = torch.zeros(num_channels, self.max_duration)
            num_near = 0
            transcript = ""

        tgt = near_audio_tensor
        target_len = tgt.shape[-1]
        assert tgt.shape[0] == 8, "the channel numbder of channels should be 8"

        # Load interference audio (trimmed but not yet length-adjusted)
        far_raw_list = []
        far_rir_order = []
        # RIR-based interference: not yet supported (far_rir_order always empty)

        for i in range(num_far):
            if self.split == 'train':
                rir_p = rng.uniform(0, 1)
            else:
                rir_p = 1

            if rir_p <= self.noise_prob and len(far_rir_order) > 0:
                audio = self.load_audio_from_rir(far_rir_order[i], target_len, rng)
                far = torch.from_numpy(audio)
            else:
                audio = self.load_audio(far_audio_order[i], rng)
                max_val_far = np.abs(audio).max()
                if max_val_far < 1e-9:
                    print(f"[Warning] far audio {i} loaded power is very small, setting to zeros")
                    audio = np.zeros_like(audio)
                else:
                    scale_far = rng.uniform(0.1, 1)
                    audio = (audio / max_val_far) * scale_far
                    _, indexes_far = librosa.effects.trim(audio, top_db=20)
                    start_i_far = max(0, indexes_far[0] - 160)
                    end_i_far = min(audio.shape[-1], indexes_far[1] + 160)
                    audio = audio[:, start_i_far:end_i_far]
                far = torch.from_numpy(audio)

            far_raw_list.append(far)  # keep raw; length adjustment deferred

        # Decide: interleave (sequential, no near-audio chunking) or overlap
        use_interleave = (
            self.interleaving_prob > 0
            and num_near > 0
            and len(far_raw_list) > 0
            and rng.uniform(0, 1) < self.interleaving_prob
        )

        if use_interleave:
            tgt_interleaved, noise_interleaved = self.interleave_near_and_far(tgt, far_raw_list, rng)
            if tgt_interleaved is not None:
                tgt = tgt_interleaved
                noise = noise_interleaved
            else:
                # Not enough space for interference; fall back to overlap
                far_audio_list = [self.adjust_far_audio_length(f, target_len, rng) for f in far_raw_list]
                noise = sum(far_audio_list)
        else:
            far_audio_list = [self.adjust_far_audio_length(f, target_len, rng) for f in far_raw_list]
            noise = sum(far_audio_list) if len(far_audio_list) > 0 else torch.zeros_like(tgt)

        rms_tgt = torch.sqrt(torch.mean(tgt**2))
        rms_noise = torch.sqrt(torch.mean(noise**2))

        target_snr = None
        if rms_tgt > 1e-6 and rms_noise > 1e-6:
            target_snr = rng.uniform(self.snr_range[0], self.snr_range[1])
            snr_scale = get_scale_for_snr(target_snr, tgt, noise, used_channels_final)
            noise = noise * snr_scale


        elif rms_noise <= 1e-6 and rms_tgt > 1e-6:
            # the power of noise is almost zeros
            num_far = 0
        elif rms_tgt <= 1e-6 and rms_noise > 1e-6:
            # the power of tgt is almost zeros
            tgt = torch.zeros_like(tgt)
            num_near = 0
            transcript = ""
        else:
            raise ValueError(f"Invalid state: rms_tgt {rms_tgt} and rms_noise {rms_noise}")
        mixture = noise + tgt
        peak = torch.abs(mixture).max().item()
        if peak > 1:
            mixture /= peak
            tgt /= peak

        assert (torch.isnan(mixture).max().item() == 0), "mixture tensor is nan!"

        return mixture, tgt, num_near, num_far, target_snr, near_already_compressed, room_name, transcript

    def get_item_from_paths(self, near_path: str, far_paths: list, seed: int, forced_target_snr: float = None):
        """Fixed-path entry point for apple-to-apple comparison. Negative samples and negative_ratio disabled."""
        rng = np.random.RandomState(seed)
        used_channels_final = self.used_channels
        near_already_compressed = self.path2already_compressed.get(near_path, False)

        transcript = ""
        transcript_path = os.path.join(near_path, 'transcript.txt')
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                transcript = f.read().strip()

        # Load full near audio
        near_audio = self.load_audio(near_path, rng)
        max_val = np.abs(near_audio).max()
        num_near = 1
        if max_val < 1e-9:
            near_t = torch.zeros_like(torch.from_numpy(near_audio))
            num_near = 0
        else:
            scale = rng.uniform(0.1, 1)
            near_np = (near_audio / max_val) * scale
            near_t = torch.from_numpy(near_np)
            if near_t.shape[-1] > self.max_duration:
                near_t = near_t[:, :self.max_duration]

        tgt = near_t
        target_len = tgt.shape[-1]

        # Load far audio (raw, no length adjustment yet) — mirrors create_scene
        far_raw_list = []
        for far_path in far_paths:
            audio = self.load_audio(far_path, rng)
            max_val_far = np.abs(audio).max()
            if max_val_far < 1e-9:
                audio = np.zeros_like(audio)
            else:
                scale_far = rng.uniform(0.1, 1)
                audio = (audio / max_val_far) * scale_far
                _, idx_far = librosa.effects.trim(audio, top_db=20)
                si = max(0, idx_far[0] - 160)
                ei = min(audio.shape[-1], idx_far[1] + 160)
                audio = audio[:, si:ei]
            far_raw_list.append(torch.from_numpy(audio))
        num_far = len(far_raw_list)

        # Interleave vs overlap — same logic as create_scene
        use_interleave = (
            self.interleaving_prob > 0
            and num_near > 0
            and len(far_raw_list) > 0
            and rng.uniform(0, 1) < self.interleaving_prob
        )

        if use_interleave:
            tgt_interleaved, noise_interleaved = self.interleave_near_and_far(tgt, far_raw_list, rng)
            if tgt_interleaved is not None:
                tgt = tgt_interleaved
                noise = noise_interleaved
            else:
                far_audio_list = [self.adjust_far_audio_length(f, target_len, rng) for f in far_raw_list]
                noise = sum(far_audio_list)
        else:
            far_audio_list = [self.adjust_far_audio_length(f, target_len, rng) for f in far_raw_list]
            noise = sum(far_audio_list) if far_audio_list else torch.zeros_like(tgt)

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
            transcript = ""

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
            'transcript': transcript,
            'num_target_speakers': num_near,
            'num_speakers': num_near + num_far,
            'target_snr': target_snr if target_snr is not None else 0,
            'room_name': near_path,
            'near_path': near_path,
            'far_paths': far_paths,
        }
        return inputs, targets_out

    def __getitem__(self, idx: int, seed=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is None:
            if self.split == 'train':
                seed = idx + self.epoch * len(self)
            else:
                seed = idx
        rng = np.random.RandomState(seed)

        used_channels_final = self.used_channels
        mixture, tgt, num_near, num_far, target_snr, near_already_compressed, room_name, transcript = \
            self.create_scene(idx, rng, used_channels_final)

        if num_near > 0:
            assert torch.abs(tgt).max().item() > 0

        g1 = None

        if len(used_channels_final) > 1:
            x_11 = mixture[used_channels_final[0], :].unsqueeze(0)
            x_12 = mixture[used_channels_final[1], :].unsqueeze(0)
            gt1 = tgt[used_channels_final[0], :].unsqueeze(0)
            gt2 = tgt[used_channels_final[1], :].unsqueeze(0)

            if self.split == 'train':
                mixture_cat = torch.cat([x_11, x_12], dim=0)
                num_ch = x_11.shape[0]
                x_1, gt1 = self.augmentations.apply_random_augmentations(mixture_cat, gt1, rng)
                x_11 = x_1[:num_ch]
                x_12 = x_1[num_ch:]

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
                x_11, gt1 = self.augmentations.apply_random_augmentations(x_11, gt1, rng)

            if self.use_drc:
                avg_sign = x_11[0]
                g1, _ = drc(avg_sign.numpy(), 0.5)
                g1 = torch.from_numpy(g1).unsqueeze(0) * 0.8
                x_11 = x_11 * g1
                gt1 = gt1 * g1

            mixture = x_11
            target = gt1

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
        }
        if g1 is not None:
            inputs['g1'] = g1

        targets = {
            'target': target,
            'transcript': transcript,
            'num_target_speakers': num_near,
            'num_speakers': num_near + num_far,
            'target_snr': target_snr,
            'room_name': room_name,
        }

        return inputs, targets
