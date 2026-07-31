"""
Torch dataset object for synthetically rendered
spatial data, with full (unchunked) LibriTTS utterances
and their transcripts attached to the near/target speaker.
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

    Unlike se_parsed_data_multi_noise.Dataset, the near/target speaker's LibriTTS
    utterance is never chunked/concatenated to a fixed duration: the FULL utterance
    (and its transcript) is loaded whenever the near speaker is synthesized via
    RIR + LibriTTS, and that utterance's own length becomes the scene's duration for
    that sample. All other sources (far speakers, noise, extra near speakers) are then
    fit to that length (cropped if longer; else 50% zero-padded / 50% tile-repeated).
    When there is no such reference utterance (e.g. no near speaker, or the near
    speaker is drawn from the real pre-recorded "audio" pool instead), behavior falls
    back to the fixed `duration` config value and `transcript` is None.

    `__getitem__`'s `idx` walks a flat, duration-filtered list of every LibriTTS
    utterance under `libritts_dir` in a fixed (speaker_id, path) order (see
    `_libritts_flat`) -- so one epoch (`samples_per_epoch`, or its default of
    `len(_libritts_flat)`) covers every usable utterance once. The near speaker's
    room and RIR/recording position are instead drawn randomly per sample.
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
                 min_utterance_duration=2.0, max_utterance_duration=30.0,
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
        self.min_utterance_duration = min_utterance_duration
        self.max_utterance_duration = max_utterance_duration
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

        # Deterministic (speaker_id, wav_path) enumeration of every LibriTTS
        # utterance whose duration falls in [min_utterance_duration,
        # max_utterance_duration]. idx walks this flat list in order (see
        # __getitem__) so that one epoch corresponds to one pass over every
        # usable LibriTTS utterance; the near speaker's room/RIR position is
        # instead drawn randomly by the seeded RNG in __getitem__.
        self._libritts_flat = []
        for spk_id in sorted(self.speaker2wavs.keys()):
            for wav_path in sorted(self.speaker2wavs[spk_id]):
                info = sf.info(wav_path)
                duration_sec = info.frames / info.samplerate
                if min_utterance_duration <= duration_sec <= max_utterance_duration:
                    self._libritts_flat.append((spk_id, wav_path))
        if not self._libritts_flat:
            raise ValueError(
                "No LibriTTS utterances were found with duration in "
                f"[{min_utterance_duration}, {max_utterance_duration}]s."
            )
        print(f"[libritts_flat] {len(self._libritts_flat)} utterances in duration range "
              f"out of {sum(len(v) for v in self.speaker2wavs.values())} total")

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

        # Exactly one near/target speaker per scene, always.
        if near_speakers_min != 1 or near_speakers_max != 1:
            print(f"[near_speakers] near_speakers_min/max={near_speakers_min}/{near_speakers_max} "
                  f"passed but ignored -- always exactly 1 near speaker per scene.")
        if self.room_mixing:
            raise NotImplementedError(
                "room_mixing is not supported: far/noise sources are drawn per-room "
                "(see get_far_samples), and room_mixing would pool them globally "
                "across rooms instead."
            )
        # Flat (room, near-position) pool: rooms in sorted order, each room's
        # RIR-type near positions (sorted by path) before its
        # pre-recorded-audio-type near positions (sorted by path). idx does
        # NOT walk this list -- __getitem__ draws a random entry from it via
        # the seeded RNG, since idx instead walks self._libritts_flat.
        self._near_flat = []
        for room in sorted(self.rooms):
            room_near = self.room2near[room]
            for kind in ("rir", "audio"):
                for sample in sorted(room_near[kind], key=lambda s: s.path):
                    self._near_flat.append((room, sample, kind))
        if not self._near_flat:
            raise ValueError("No near samples were found across any room.")
        print(f"[near_flat] {len(self._near_flat)} unique near positions "
              f"across {len(self.rooms)} rooms")

        if self.samples_per_epoch is None:
            self.samples_per_epoch = len(self._libritts_flat)

        self.near_speakers_range = (1, 1)
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
        print("utterance duration range (s): ", (self.min_utterance_duration, self.max_utterance_duration))

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

    def read_transcript(self, wav_path):
        """
        Reads the LibriTTS transcript that accompanies `wav_path`, preferring the
        normalized transcript. Returns None if neither transcript file exists.
        """
        base_path = wav_path[:-len('.wav')] if wav_path.endswith('.wav') else wav_path
        for suffix in ('.normalized.txt', '.original.txt'):
            transcript_path = base_path + suffix
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r') as f:
                    return f.read().strip()
        return None

    def sample_full_speaker_utterance(self, speaker_id, rng: np.random.RandomState,
                                       max_attempts=50, max_speaker_switches=20):
        """
        Loads an ENTIRE (unchunked) LibriTTS utterance for `speaker_id`, together
        with its transcript. Utterances whose duration falls outside
        [min_utterance_duration, max_utterance_duration] are rejected and another
        utterance is sampled instead. If `speaker_id` has no utterance in range
        after `max_attempts` tries, a different random speaker is substituted
        instead of raising (up to `max_speaker_switches` times, as a last-resort
        safety net against an unusable config looping forever).
        """
        for _switch in range(max_speaker_switches + 1):
            for _ in range(max_attempts):
                audio_path = rng.choice(self.speaker2wavs[speaker_id], size=1)[0]

                with sf.SoundFile(audio_path) as f:
                    file_sr = f.samplerate
                    num_frames = f.frames
                    duration_sec = num_frames / file_sr

                    if duration_sec < self.min_utterance_duration or duration_sec > self.max_utterance_duration:
                        continue

                    audio = self.read_audio_as_mono(f, num_frames=num_frames)

                if file_sr != self.sr:
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr, res_type='polyphase')

                transcript = self.read_transcript(audio_path)

                return audio, transcript

            # This speaker has no utterance in range -- skip it, try another speaker.
            speaker_id = rng.choice(self.speaker_ids, size=1)[0]

        raise RuntimeError(
            f"Could not find any utterance with duration in "
            f"[{self.min_utterance_duration}, {self.max_utterance_duration}]s after "
            f"trying {max_speaker_switches + 1} speakers x {max_attempts} attempts each."
        )

    def load_utterance(self, wav_path):
        """
        Loads one specific, already duration-filtered LibriTTS utterance (see
        `_libritts_flat`) together with its transcript. Unlike
        `sample_full_speaker_utterance`, this is deterministic: no RNG, no
        retries, since the caller already knows `wav_path` is usable.
        """
        with sf.SoundFile(wav_path) as f:
            file_sr = f.samplerate
            num_frames = f.frames
            audio = self.read_audio_as_mono(f, num_frames=num_frames)

        if file_sr != self.sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr, res_type='polyphase')

        transcript = self.read_transcript(wav_path)
        return audio, transcript

    def fit_audio_to_length(self, audio: np.ndarray, tgt_frames: int, rng: np.random.RandomState):
        """
        Crops `audio` to `tgt_frames` if it is longer. If it is shorter, either
        zero-pads it (50%) or tile-repeats it (50%) to reach `tgt_frames`.
        """
        cur_frames = audio.shape[-1]
        if cur_frames == tgt_frames:
            return audio

        if cur_frames > tgt_frames:
            start = rng.randint(0, cur_frames - tgt_frames + 1)
            return audio[start:start + tgt_frames]

        if rng.rand() < 0.5:
            out = np.zeros(tgt_frames, dtype=audio.dtype)
            start = rng.randint(0, tgt_frames - cur_frames + 1)
            out[start:start + cur_frames] = audio
            return out
        else:
            n_repeats = int(np.ceil(tgt_frames / cur_frames))
            return np.tile(audio, n_repeats)[:tgt_frames]

    def sample_noise_snippet(self, rng: np.random.RandomState, tgt_frames: int):
        noise_file = rng.choice(self.noise_samples, size=1)[0]
        with sf.SoundFile(noise_file) as f:
            num_frames = f.frames
            file_sr = f.samplerate

            N = int(round(file_sr * tgt_frames / self.sr))

            if N > num_frames:
                # Read entire audio; fit_audio_to_length below will pad/repeat it
                audio = self.read_audio_as_mono(f, num_frames=num_frames)
            else:
                # Randomly choose start of snippet
                start_frame = rng.randint(0, num_frames - N + 1)

                # Move to start of snippet and read
                f.seek(start_frame)
                audio = self.read_audio_as_mono(f, num_frames=N)

            # Resample to target sampling rate
            if file_sr != self.sr:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr, res_type='polyphase')

        audio = self.fit_audio_to_length(audio, tgt_frames, rng)

        if(np.mean(np.square(audio)) < 1e-5):
            audio = self.sample_noise_snippet(rng, tgt_frames)
        return audio

    def sample_random_snippet(self, spk_id, rir_dir, rng: np.random.RandomState, is_noise: bool,
                               tgt_frames=None, forced_utterance_path=None):
        """
        If is_noise, `tgt_frames` is required and the noise is fit to it.
        If not is_noise and `forced_utterance_path` is given, that exact LibriTTS
        utterance is loaded (used for the near/reference speaker, whose utterance
        is walked deterministically by idx -- see `_libritts_flat`). Otherwise the
        FULL LibriTTS utterance (+ transcript) is loaded for a randomly chosen
        `spk_id`; when `tgt_frames` is given (i.e. this is not the reference/target
        speaker), the utterance is fit to it, and the transcript is only
        meaningful for the reference speaker.
        """
        transcript = None

        if is_noise:
            assert tgt_frames is not None, "Noise sources require a target scene length"
            single_ch_audio = self.sample_noise_snippet(rng, tgt_frames)
        else:
            if forced_utterance_path is not None:
                single_ch_audio, transcript = self.load_utterance(forced_utterance_path)
            else:
                single_ch_audio, transcript = self.sample_full_speaker_utterance(spk_id, rng)
            if tgt_frames is not None:
                single_ch_audio = self.fit_audio_to_length(single_ch_audio, tgt_frames, rng)

        num_channels = self.n_mics
        tgt_samples = single_ch_audio.shape[-1]
        audio = np.zeros((num_channels, tgt_samples), dtype=np.float32)

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
            RIR = rir[i]
            spatial_audio = signal.convolve(single_ch_audio, RIR, mode='full')
            audio[i] = spatial_audio[: -(len(RIR) - 1)]

        return audio, transcript

    def sample_random_snippet_no_spatialization(self, chosen_path, rng: np.random.RandomState, total_duration=None):
        if total_duration is None:
            total_duration = self.duration

        left_path = os.path.join(chosen_path, 'left.wav')
        right_path = os.path.join(chosen_path, 'right.wav')

        min_active_duration = min(self.min_active_duration, total_duration)
        tgt_duration = rng.uniform(min_active_duration, total_duration)
        tgt_frames = int(round(self.sr * tgt_duration))
        complete_frames = int(round(self.sr * total_duration))

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
            return self.sample_random_snippet_no_spatialization(chosen_path, rng, total_duration=total_duration)
        else:
            return audio

    def create_scene(self, sample_idx: int, room: str, rng: np.random.RandomState,
                      forced_near_sample: sample_descriptor, forced_near_kind: str,
                      forced_utterance_path=None):
        # Exactly one near/target speaker per scene, always. `room` and
        # forced_near_sample/forced_near_kind (the RIR/recording position) are
        # drawn randomly by __getitem__; when forced_near_kind == 'rir',
        # forced_utterance_path is the LibriTTS utterance walked deterministically
        # by idx (see `_libritts_flat`) that gets synthesized through that RIR.
        num_near = 1
        is_complete_silence = False
        # Choose outer speakers
        num_far = rng.randint(self.far_sources_range[0], self.far_sources_range[1] + 1)


        num_noises = rng.randint(self.noise_sources_range[0], self.noise_sources_range[1] + 1)

        if rng.rand() < self.noise_prob:
            # Use noise source
            num_noises += 1

        audio_list = []
        chosen_paths = []

        # Get near & far dicts. `near_dict['audio']` availability still gates
        # whether far sources may use the pre-recorded-audio path below (a
        # pre-existing coupling, unrelated to the near speaker itself, which
        # is always `forced_near_sample`/`forced_near_kind`).
        near_dict = self.get_near_samples(room)
        far_dict = self.get_far_samples(room)

        # With replacement
        far_rir_order = rng.choice(far_dict['rir'], size=num_far + num_noises, replace=True).tolist()

        rir_ratio = self.rir_ratio
        if len(near_dict['audio']) > 0:
            far_audio_order = rng.choice(far_dict['audio'], size=num_far, replace=True).tolist()
        else:
            far_audio_order = []
            rir_ratio = 1
        # Load near, far and noise into target audio
        angles = []
        distances = []
        selected_speaker_ids = rng.choice(self.speaker_ids, size=(num_near + num_far + num_noises))

        tgt_frames = None
        transcript = None
        near_libritts_path = None
        for i in range(num_near + num_far + num_noises):
            audio_dict = None
            is_noise = False

            if i == 0:
                # Near/target speaker: the LibriTTS utterance synthesized here is
                # fixed by idx via `forced_utterance_path` (see `_libritts_flat`);
                # the (room, position) it's rendered through is drawn from `rng`.
                is_reference = True
                current_tgt_frames = None
                chosen_sample: sample_descriptor = forced_near_sample

                if forced_near_kind == 'rir':
                    spatialized_audio, seg_transcript = self.sample_random_snippet(
                        selected_speaker_ids[i], chosen_sample.path, rng,
                        is_noise=False, tgt_frames=current_tgt_frames,
                        forced_utterance_path=forced_utterance_path)
                    near_libritts_path = forced_utterance_path
                else:
                    spatialized_audio = self.sample_random_snippet_no_spatialization(
                        chosen_sample.path, rng, total_duration=self.duration)
                    seg_transcript = None

                tgt_frames = spatialized_audio.shape[-1]
                transcript = seg_transcript
            else:
                is_reference = False
                current_tgt_frames = tgt_frames

                if i < num_near + num_far:
                    # far
                    rir_p = rng.uniform(0, 1) if self.split == 'train' else 1
                else:
                    # noise
                    is_noise = True
                    rir_p = 0
                audio_order = far_audio_order
                rir_order = far_rir_order

                chosen_sample: sample_descriptor
                if rir_p <= rir_ratio:
                    # Choose unique RIR
                    chosen_sample = rir_order[-1]
                    rir_order.pop()

                    spatialized_audio, seg_transcript = self.sample_random_snippet(
                        selected_speaker_ids[i], chosen_sample.path, rng,
                        is_noise=is_noise, tgt_frames=current_tgt_frames)
                else:
                    # Choose unique audio (real, pre-recorded; no transcript available)
                    chosen_sample = audio_order[-1]
                    audio_order.pop()

                    total_duration = (current_tgt_frames / self.sr) if current_tgt_frames is not None else self.duration
                    spatialized_audio = self.sample_random_snippet_no_spatialization(
                        chosen_sample.path, rng, total_duration=total_duration)
                    seg_transcript = None

            if i < (num_near + num_far) and (rng.rand() < self.silence_prob):
                selected_silence_duration = rng.uniform(self.silence_range[0], self.silence_range[1])
                selected_silence_frames = int(round(selected_silence_duration * self.sr))
                total_frames = spatialized_audio.shape[-1]
                starting_silence_frame = rng.randint(0, max(total_frames - selected_silence_frames, 0) + 1)
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

        if tgt_frames is None:
            tgt_frames = int(round(self.sr * self.duration))

        if is_complete_silence:
            num_near = 0
            transcript = None
            near_libritts_path = None

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
                audio_list, target_snr, chosen_paths, angles, distances, transcript, near_libritts_path

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

        # Walk the flat LibriTTS utterance list in order instead of randomly
        # choosing one; mirrors the seed formula above so a second training
        # epoch keeps advancing through the pool instead of replaying the same
        # utterances. The near speaker's room/RIR position is instead drawn
        # randomly (via `rng`) from `_near_flat`.
        libritts_pos = idx + self.epoch * len(self) if self.split == 'train' else idx
        _forced_speaker_id, forced_utterance_path = self._libritts_flat[libritts_pos % len(self._libritts_flat)]

        near_idx = rng.randint(0, len(self._near_flat))
        room, forced_near_sample, forced_near_kind = self._near_flat[near_idx]

        # Create scene
        mixture: torch.Tensor
        gt: torch.Tensor
        mixture, gt, num_tgt_speakers, num_far_speakers, num_noises, audio_list, target_snr, chosen_paths, speaker_angles, speaker_distances, transcript, near_libritts_path = \
            self.create_scene(idx, room, rng, forced_near_sample=forced_near_sample,
                               forced_near_kind=forced_near_kind,
                               forced_utterance_path=forced_utterance_path)
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

        # if (target_snr is not None) and (len(self.augmentations) == 0) and (not self.use_drc):
        # actual_snr = get_snr(gt1, x_11[0].unsqueeze(0))
        # actual_snr2 = get_snr(gt2, x_22[0].unsqueeze(0))
        # print(actual_snr, actual_snr2, target_snr)
        # assert torch.abs(actual_snr - target_snr) < 0.1

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
        # Define targets
        targets = {
            'target_at_dev1':gt1,
            'target_at_dev2':gt2,
            'num_target_speakers':num_tgt_speakers,
            'num_interfering_speakers':num_far_speakers,
            'num_noises':num_noises,
            'transcript': transcript,
            'libritts_path': near_libritts_path,
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
        "Noise_Equal_Weightage":True,
        "samples_per_epoch":2000,
        "n_mics":8,
        "sr":16000,
        "min_active_duration":2,
        "duration":5,
        "min_utterance_duration": 2.0,
        "max_utterance_duration": 30.0,
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
        "rir_ratio":1, # Use the synthetic RIR+LibriTTS path so a transcript is produced
        "room_mixing" : False, # Use multiple rooms to create mixtures
        "use_drc" : True, # Apply DRC on mixture
    }

    data_loader = Dataset(**val_data_args, split='train')

    print("creation complete")
    print("length: ", len(data_loader))

    for batch_idx, b in enumerate(data_loader):
        inputs, targets = b

        for i in inputs.keys():
            if i not in ["comm_delay_chunks", "seed"]:
                print(f"{i} -> {inputs[i].shape}, -> {inputs[i].dtype}")
        for i in targets.keys():
            if i not in ["comm_delay_chunks", "seed"]:
                print(f"{i} -> {targets[i]}")
        print(batch_idx)
        print("="*25)
        if(batch_idx > 10):
            break
