"""
Torch dataset for VAD finetuning.

Same data pipeline as the separation dataset, with one addition:
    - loads left_vad.npy / right_vad.npy from each near-speech sample folder
    - returns vad_gt as chunk-level binary labels (1 label per VAD_CHUNK_SIZE samples)
      aligned with the output audio

left_vad.npy / right_vad.npy convention:
    float32 binary label arrays (0 or 1), one value per 512-sample chunk at 16 kHz.
    Shape: [N_c] (single-channel) or [n_ch, N_c] (multi-channel, one row per audio channel).
    Multi-channel arrays are reduced to 1D by taking the per-chunk max across channels.

All random crop/pad operations use chunk-aligned offsets so that vad labels remain
exactly in sync with the audio without any sub-chunk misalignment.
"""
from typing import Tuple, Dict

import torch
import numpy as np
import os
import glob
import traceback

import pandas as pd
import librosa
import scipy.signal as signal

from torchmetrics.functional.audio import signal_noise_ratio as snr

import src.utils as utils
from src.datasets.augmentations.audio_augmentations import AudioAugmentations

try:
    import lc3
    _HAS_LC3 = True
except ImportError:
    _HAS_LC3 = False

VAD_CHUNK_SIZE = 512   # samples at 16 kHz → 32 ms per label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drc(avg_sign, prev_val):
    a = np.abs(avg_sign)
    alpha = 0.999
    b, prev_val = signal.lfilter([1 - alpha], [1, -alpha], a, zi=[prev_val * alpha])
    b_dB = 20 * np.log10(b + 1e-6)
    gain_dB = np.clip(-25 - b_dB, 0, 30)
    return (10 ** (gain_dB / 20)).astype(np.float32), prev_val


def get_snr(target, mixture):
    return snr(mixture, target).mean()


def get_scale_for_snr(target_snr, tgt, noise, reference_channels=None):
    dt = tgt  if reference_channels is None else tgt[reference_channels]
    dm = tgt + noise if reference_channels is None else (tgt + noise)[reference_channels]
    current_snr = get_snr(dt, dm)
    return 10 ** ((current_snr - target_snr) / 20)


def get_subdirs(path):
    return sorted(s for s in os.listdir(path) if not s.startswith('.'))


def apply_codec_artifacts(audio, sr):
    if not _HAS_LC3:
        return audio
    assert len(audio.shape) == 1
    fd, nc, br = 10, 1, 16000
    enc = lc3.Encoder(fd * 1000, sr, nc)
    fs, fl = enc.get_frame_bytes(br), enc.get_frame_samples()
    encoded = [enc.encode(audio[i:i + fl], fs) for i in range(0, len(audio), fl)]
    dec = lc3.Decoder(fd * 1000, sr, nc)
    return np.concatenate([dec.decode(b, None) for b in encoded])


def apply_codec_artifacts_multichannel(audio, sr):
    return np.stack([apply_codec_artifacts(audio[i], sr) for i in range(audio.shape[0])])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    """
    Same mixing pipeline as the separation dataset, extended with VAD GT.

    targets dict gains one extra key:
        'vad_gt': FloatTensor [n_chunks]  — chunk-level speech activity labels,
                  n_chunks = duration * sr // VAD_CHUNK_SIZE
    """

    def __init__(self,
                 self_speech_dir: Dict[str, float],
                 interference_dir: Dict[str, float],
                 noise_dir_config,
                 split,
                 samples_per_epoch,
                 sr=16000,
                 duration=5,
                 augmentations=[],
                 used_channels=[0, 3],
                 far_sources_range=[0, 4],
                 snr_min=-10, snr_max=5,
                 use_drc=False,
                 noise_prob=0.0,
                 compression_aug_ratio=0.0,
                 interleaving_prob=0.0,
                 swap_left_right=False):
        super().__init__()

        self.duration            = int(duration * sr)
        self.sr                  = sr
        self.snr_range           = (snr_min, snr_max)
        self.used_channels       = used_channels
        self.use_drc             = use_drc
        self.epoch               = 0
        self.swap_left_right     = swap_left_right
        self.self_speech_dir     = self_speech_dir
        self.near_samples        = []
        self.far_samples         = []
        self.samples_per_epoch   = samples_per_epoch
        self.split               = split
        self.far_sources_range   = far_sources_range
        self.target_rooms        = []
        self.interf_rooms        = []
        self.room2near           = {}
        self.room2far            = {}
        self.compression_aug_ratio = compression_aug_ratio
        self.interleaving_prob   = interleaving_prob
        self.noise_prob          = noise_prob
        # chunks in one output clip
        self.n_out_chunks        = self.duration // VAD_CHUNK_SIZE

        if noise_prob > 0:
            self.wham_dir          = noise_dir_config.get('wham_dir', None)
            self.musan_dir         = noise_dir_config.get('musan_dir', None)
            self.musan_noise_types = noise_dir_config.get('musan_noise_types', None)
            self.babble_dir        = noise_dir_config.get('babble_dir', None)
            self.musdb18_dir       = noise_dir_config.get('musdb18_dir', None)
            self.fsd50k_dir        = noise_dir_config.get('fsd50k_dir', None)
            self.noise_samples     = self._get_noise_samples()
        else:
            self.noise_samples = []

        for dataset_dir, prob in self_speech_dir.items():
            nd = {"audio": [], "prob": prob}
            for room in get_subdirs(dataset_dir):
                self.target_rooms.append(room)
                self.room2near[room] = {'audio': []}
                room_dir = os.path.join(dataset_dir, room)
                df = pd.read_csv(os.path.join(room_dir, 'metadata.csv'))
                for _, row in df.iterrows():
                    if row['audio_type'] == 'a':
                        p = os.path.join(room_dir, os.path.basename(row['path']))
                        self.room2near[room]['audio'].append(p)
                        nd['audio'].append(p)
            self.near_samples.append(nd)

        for dataset_dir, prob in interference_dir.items():
            fd = {"audio": [], "rir": [], "prob": prob}
            for room in get_subdirs(dataset_dir):
                self.interf_rooms.append(room)
                self.room2far[room] = {'audio': [], 'rir': []}
                room_dir = os.path.join(dataset_dir, room)
                df = pd.read_csv(os.path.join(room_dir, 'metadata.csv'))
                for _, row in df.iterrows():
                    if row['audio_type'] == 'a':
                        p = os.path.join(room_dir, os.path.basename(row['path']))
                        self.room2far[room]['audio'].append(p)
                        fd['audio'].append(p)
                    elif row['audio_type'] == 'c':
                        if row['distance_cm'] < 60:
                            continue
                        p = os.path.join(room_dir, os.path.basename(row['path']))
                        self.room2far[room]['rir'].append(p)
                        fd['rir'].append(p)
            self.far_samples.append(fd)

        print("*" * 25)
        for nd in self.near_samples:
            print("Near dataset:", nd['prob'], len(nd['audio']))
        for fd in self.far_samples:
            print("Far dataset:", fd['prob'], len(fd['audio']), len(fd['rir']))
        print("=" * 25)

        self.augmentations = AudioAugmentations(augmentations)

    # ------------------------------------------------------------------
    # Noise helpers
    # ------------------------------------------------------------------

    def _get_noise_samples(self):
        wham = musan = babble = musdb = fsd = []
        if getattr(self, 'wham_dir', None):
            split_map = {'train': 'tr', 'test': 'tt', 'val': 'cv'}
            wham = glob.glob(os.path.join(self.wham_dir, split_map[self.split]) + "/*.wav")
        if getattr(self, 'musan_dir', None):
            df = pd.read_csv(os.path.join(self.musan_dir, "metadata.csv"))
            files = []
            for t in self.musan_noise_types:
                files += df[df["filetype"] == f"{self.split}_{t}"]["filename"].tolist()
            musan = [os.path.join(self.musan_dir, f) for f in files]
        if getattr(self, 'babble_dir', None):
            babble = glob.glob(os.path.join(self.babble_dir, self.split) + "/*.wav")
        if getattr(self, 'musdb18_dir', None):
            s = 'test' if self.split == 'val' else self.split
            musdb = glob.glob(os.path.join(self.musdb18_dir, s) + "/*.wav")
        if getattr(self, 'fsd50k_dir', None):
            s = 'FSD50K.dev_audio' if self.split == 'train' else 'FSD50K.eval_audio'
            fsd = glob.glob(os.path.join(self.fsd50k_dir, s) + "/*.wav")

        rng = np.random.RandomState(42)
        all_lists = [wham, musan, babble, musdb, fsd]
        max_n = max((len(x) for x in all_lists), default=0)
        balanced = []
        for lst in all_lists:
            if 0 < len(lst) < max_n:
                lst = lst + rng.choice(lst, max_n - len(lst), replace=True).tolist()
            balanced.extend(lst)
        return balanced

    # ------------------------------------------------------------------
    # Path helpers (handles left/right swap bug in old dataset)
    # ------------------------------------------------------------------

    def _audio_paths(self, sample_dir):
        """Return (left_audio_path, right_audio_path) with L/R swap correction."""
        if "/parsed_earbud_human_data/" in sample_dir:
            return (os.path.join(sample_dir, 'right.wav'),
                    os.path.join(sample_dir, 'left.wav'))
        return (os.path.join(sample_dir, 'left.wav'),
                os.path.join(sample_dir, 'right.wav'))

    def _vad_paths(self, sample_dir):
        """Return (left_vad_path, right_vad_path) with the same L/R swap correction."""
        if "/parsed_earbud_human_data/" in sample_dir:
            return (os.path.join(sample_dir, 'right_vad.npy'),
                    os.path.join(sample_dir, 'left_vad.npy'))
        return (os.path.join(sample_dir, 'left_vad.npy'),
                os.path.join(sample_dir, 'right_vad.npy'))

    # ------------------------------------------------------------------
    # Audio / VAD loading
    # ------------------------------------------------------------------

    def load_audio(self, sample_dir, rng):
        lp, rp = self._audio_paths(sample_dir)
        left,  sr = librosa.load(lp, sr=None, mono=False)
        right, sr = librosa.load(rp, sr=None, mono=False)
        assert sr == self.sr
        assert left.shape[-1] == right.shape[-1]
        return np.concatenate([left, right], axis=0)   # [C, N]

    def load_audio_with_vad(self, sample_dir, rng):
        """
        Returns:
            audio  : np.ndarray [C, N]
            vad_gt : np.ndarray [C, N_c]  per-channel chunk-level labels,
                     concatenated from left_vad and right_vad along the channel axis,
                     mirroring how load_audio concatenates left.wav and right.wav.
        """
        audio = self.load_audio(sample_dir, rng)       # [C, N]
        N = audio.shape[-1]
        N_c = N // VAD_CHUNK_SIZE

        lvp, rvp = self._vad_paths(sample_dir)

        def _load_vad_2d(path, n_c):
            """Load VAD npy → always [n_ch, n_c] float32."""
            v = np.load(path).astype(np.float32) if os.path.exists(path) \
                else np.zeros((1, n_c), dtype=np.float32)
            if v.ndim == 1:
                v = v[np.newaxis, :]          # [1, N_c]
            return _fit_vad(v, n_c)           # [n_ch, N_c]

        left_vad  = _load_vad_2d(lvp, N_c)   # [C_left, N_c]
        right_vad = _load_vad_2d(rvp, N_c)   # [C_right, N_c]

        # Mirror audio concatenation: left channels first, then right
        vad_gt = np.concatenate([left_vad, right_vad], axis=0)   # [C, N_c]
        return audio, vad_gt

    def load_noise(self, chosen_path, rng):
        audio, sr = librosa.load(chosen_path, sr=self.sr, mono=False)
        assert sr == self.sr
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        D = self.duration
        if audio.shape[-1] < D:
            z = D - audio.shape[-1]
            fp = rng.randint(0, max(z, 1))
            audio = np.pad(audio, ((0, 0), (fp, z - fp)))
        elif audio.shape[-1] > D:
            delta = audio.shape[-1] - D
            s = rng.randint(0, max(delta, 1))
            audio = audio[:, s:s + D]
        return audio[0]

    def load_audio_from_rir(self, chosen_path, rng):
        noise = self.load_noise(rng.choice(self.noise_samples), rng)
        lp, rp = self._audio_paths(chosen_path)
        left_rir,  sr = librosa.load(lp, sr=None, mono=False)
        right_rir, sr = librosa.load(rp, sr=None, mono=False)
        assert sr == self.sr
        rir = np.concatenate([left_rir, right_rir], axis=0)
        result = []
        for i in range(rir.shape[0]):
            s = signal.convolve(noise, rir[i], mode='full')
            result.append(s[:-(len(rir[i]) - 1)])
        audio = np.stack(result)
        assert audio.shape[-1] == self.duration
        return audio

    # ------------------------------------------------------------------
    # Chunk-aligned pad / crop
    # ------------------------------------------------------------------

    def pad_chunk_audio(self, audio, duration, rng):
        """Pad/crop audio [C, T] → [C, duration] (sample-level, arbitrary alignment)."""
        T = audio.shape[-1]
        if T < duration:
            z = duration - T
            fp = rng.randint(0, max(z, 1))
            audio = torch.nn.functional.pad(audio, (fp, z - fp))
        elif T > duration:
            delta = T - duration
            s = rng.randint(0, max(delta, 1))
            audio = audio[:, s:s + duration]
        return audio

    def pad_chunk_audio_and_vad(self, audio, vad, duration, rng):
        """
        Pad/crop audio [C, T] and chunk-level vad [C, T_c] together.

        Random offsets are always multiples of VAD_CHUNK_SIZE so that
        sample positions and vad chunk indices stay exactly in sync.
        vad operations use the last axis so the channel dimension is preserved.

        Returns:
            audio [C, duration], vad [C, duration // VAD_CHUNK_SIZE]
        """
        T     = audio.shape[-1]
        n_out = duration // VAD_CHUNK_SIZE    # desired vad length
        C_v   = vad.shape[0]                  # vad channel count

        if T < duration:
            zero_pad = duration - T
            max_fp_c = zero_pad // VAD_CHUNK_SIZE
            fp_c     = rng.randint(0, max(max_fp_c + 1, 1))
            fp       = fp_c * VAD_CHUNK_SIZE
            bp       = zero_pad - fp
            audio = torch.nn.functional.pad(audio, (fp, bp))
            # Build vad: [C_v, fp_c zeros] + [existing vad] → fit to n_out
            front = np.zeros((C_v, fp_c), dtype=np.float32)
            vad   = np.concatenate([front, vad], axis=-1)
            vad   = _fit_vad(vad, n_out)

        elif T > duration:
            delta   = T - duration
            max_s_c = delta // VAD_CHUNK_SIZE
            s_c     = rng.randint(0, max(max_s_c + 1, 1))
            s       = s_c * VAD_CHUNK_SIZE
            audio = audio[:, s:s + duration]
            vad   = vad[..., s_c:]
            vad   = _fit_vad(vad, n_out)

        else:
            vad = _fit_vad(vad, n_out)

        return audio, vad    # audio [C, duration], vad np.ndarray [C_v, n_out]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_near_audio(self, num_near, num_far, rng):
        near_probs = np.array([d['prob'] for d in self.near_samples])
        near_probs /= near_probs.sum()
        nd = rng.choice(self.near_samples, p=near_probs)
        near_audio_order = rng.choice(nd['audio'], size=1, replace=True).tolist()

        far_probs = np.array([d['prob'] for d in self.far_samples])
        far_probs /= far_probs.sum()
        far_datasets = rng.choice(self.far_samples, size=num_far, replace=True, p=far_probs)
        far_audio_order = [rng.choice(d['audio']) for d in far_datasets]
        return near_audio_order, far_audio_order

    # ------------------------------------------------------------------
    # Mixing with chunk-level VAD tracking
    # ------------------------------------------------------------------

    def mixing_near_and_far_with_interleaving(
        self, near_audio_list, far_audio_list, near_vad_list, rng
    ):
        """
        Returns (tgt [C, D], noise [C, D], vad_gt np.ndarray [C_v, n_out_chunks]).
        near_vad_list: list of np.ndarray [C_v, T_c] — per-channel labels for each near source.
        """
        D     = self.duration
        N_OUT = self.n_out_chunks
        C     = near_audio_list[0].shape[0] if near_audio_list \
                else far_audio_list[0].shape[0]
        C_v   = near_vad_list[0].shape[0]   # VAD channel count

        if rng.uniform(0, 1) < self.interleaving_prob:
            # ---- interleaving branch ----
            # if rng.uniform(0, 1) < 0.3:
            #     # Silent target
            #     near_audio = torch.zeros(C, D)
            #     vad_gt     = np.zeros((C_v, N_OUT), dtype=np.float32)
            #     fars = [self.pad_chunk_audio(f, D, rng) for f in far_audio_list]
            #     noise = sum(fars)
            #     return near_audio, noise, vad_gt

            interval       = int(rng.uniform(-1, 1) * self.sr)
            min_active     = self.sr
            targe_duration = rng.randint(min_active,
                                         D - max(interval, 0) - min_active + 1)
            noise_duration = D - targe_duration - interval

            nears, near_vads = [], []
            for na, nv in zip(near_audio_list, near_vad_list):
                na, nv = self.pad_chunk_audio_and_vad(na, nv, targe_duration, rng)
                nears.append(na)
                near_vads.append(nv)
            near_audio = sum(nears)
            # Combine near VADs (clamp to binary) — [C_v, n_near_chunks]
            near_vad_combined = np.clip(sum(near_vads), 0, 1)
            n_near_chunks = targe_duration // VAD_CHUNK_SIZE

            fars  = [self.pad_chunk_audio(f, noise_duration, rng) for f in far_audio_list]
            noise = sum(fars)

            tgt        = torch.zeros(C, D)
            noise_full = torch.zeros(C, D)
            vad_gt     = np.zeros((C_v, N_OUT), dtype=np.float32)
            if rng.uniform(0, 1) < 0.5:
                # Near at beginning
                tgt[:, :targe_duration]         = near_audio
                noise_full[:, -noise_duration:] = noise
                vad_gt[:, :n_near_chunks]        = near_vad_combined[:, :n_near_chunks]
            else:
                # Near at end
                tgt[:, -targe_duration:]        = near_audio
                noise_full[:, :noise_duration]  = noise
                start_c = (D - targe_duration) // VAD_CHUNK_SIZE
                vad_gt[:, start_c:start_c + n_near_chunks] = \
                    near_vad_combined[:, :min(n_near_chunks, N_OUT - start_c)]

            return tgt, noise_full, vad_gt

        else:
            # ---- overlap branch ----
            nears, near_vads = [], []
            for na, nv in zip(near_audio_list, near_vad_list):
                na, nv = self.pad_chunk_audio_and_vad(na, nv, D, rng)
                nears.append(na)
                near_vads.append(nv)
            near_audio = sum(nears)
            vad_gt     = np.clip(sum(near_vads), 0, 1)   # [C_v, N_OUT]

            fars  = [self.pad_chunk_audio(f, D, rng) for f in far_audio_list]
            noise = sum(fars)
            return near_audio, noise, vad_gt

    # ------------------------------------------------------------------
    # Scene creation
    # ------------------------------------------------------------------

    SAFE_GUARD = 160

    def create_scene(self, idx, rng, used_channels_final):
        num_near = 1
        num_far  = rng.randint(self.far_sources_range[0],
                               self.far_sources_range[1] + 1)

        near_audio_order, far_audio_order = self.sample_near_audio(num_near, num_far, rng)
        
        audio_list  = []
        vad_gt_list = []
        
        for i in range(num_near + num_far):
            if i < num_near:
                raw_audio, raw_vad = self.load_audio_with_vad(near_audio_order[i], rng)
            else:
                if self.split == 'train' and rng.uniform(0, 1) <= self.noise_prob \
                        and self.noise_samples:
                    # sample a far rir from the room of the far audio (best effort)
                    room_key = os.path.basename(
                        os.path.dirname(far_audio_order[i - num_near]))
                    rir_list = self.room2far.get(room_key, {}).get('rir', [])
                    rir_path = rng.choice(rir_list) if rir_list \
                               else far_audio_order[i - num_near]
                    raw_audio = self.load_audio_from_rir(rir_path, rng)
                else:
                    raw_audio = self.load_audio(far_audio_order[i - num_near], rng)
                raw_vad = None

            max_val = np.abs(raw_audio).max()
            if max_val < 1e-9:
                print(f"[Warning] audio {i} power very small — zeroed")
                audio_list.append(torch.zeros(raw_audio.shape[0], 1))
                if i < num_near:
                    vad_gt_list.append(np.zeros((raw_audio.shape[0], 1), dtype=np.float32))
                continue

            scale = rng.uniform(0.1, 1)
            raw_audio = (raw_audio / max_val) * scale

            # Trim silence
            _, idx_bounds = librosa.effects.trim(raw_audio, top_db=20)
            s_samp = max(0, idx_bounds[0] - self.SAFE_GUARD)
            e_samp = min(raw_audio.shape[-1], idx_bounds[1] + self.SAFE_GUARD)
            raw_audio = raw_audio[:, s_samp:e_samp]

            audio_list.append(torch.from_numpy(raw_audio))

            if i < num_near and raw_vad is not None:
                # Trim vad along time axis at chunk resolution (same bounds as audio trim)
                s_c = s_samp // VAD_CHUNK_SIZE
                e_c = e_samp // VAD_CHUNK_SIZE
                s_c = max(0, s_c)
                e_c = min(raw_vad.shape[-1], e_c)
                vad_gt_list.append(raw_vad[..., s_c:e_c])

        if not vad_gt_list:
            vad_gt_list = [np.zeros((1, 1), dtype=np.float32)]

        # Dummy zero noise when num_far == 0
        if num_far == 0:
            C_dummy = audio_list[0].shape[0]
            far_list = [torch.zeros(C_dummy, self.duration)]
        else:
            far_list = audio_list[num_near:]

        tgt, noise, vad_gt = self.mixing_near_and_far_with_interleaving(
            audio_list[:num_near], far_list, vad_gt_list, rng)

        assert tgt.shape == noise.shape

        rms_tgt   = torch.sqrt(torch.mean(tgt ** 2))
        rms_noise = torch.sqrt(torch.mean(noise ** 2))

        target_snr = None
        if rms_tgt > 1e-6 and rms_noise > 1e-6:
            target_snr = rng.uniform(self.snr_range[0], self.snr_range[1])
            noise = noise * get_scale_for_snr(target_snr, tgt, noise, used_channels_final)
        elif rms_noise <= 1e-6 and rms_tgt > 1e-6:
            num_far = 0
        elif rms_tgt <= 1e-6 and rms_noise > 1e-6:
            tgt    = torch.zeros_like(tgt)
            vad_gt = np.zeros((vad_gt.shape[0], self.n_out_chunks), dtype=np.float32)
            num_near = 0
        else:
            raise ValueError(f"rms_tgt={rms_tgt}, rms_noise={rms_noise}")

        mixture = noise + tgt
        peak    = torch.abs(mixture).max().item()
        if peak > 1:
            mixture /= peak
            tgt     /= peak

        assert not torch.isnan(mixture).any(), "NaN in mixture"

        # Ensure exactly n_out_chunks labels
        vad_gt = _fit_vad(vad_gt, self.n_out_chunks)

        return mixture, tgt, num_near, num_far, target_snr, vad_gt

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def swap_mic_channels(self, used_channels):
        return [c + 4 if c < 4 else c - 4 for c in used_channels]

    def __getitem__(self, idx, seed=None):
        if seed is None:
            seed = idx + self.epoch * len(self) if self.split == 'train' else idx
        rng = np.random.RandomState(seed)

        if self.swap_left_right and rng.uniform(0, 1) < 0.4:
            used_channels_final = self.swap_mic_channels(self.used_channels)
        else:
            used_channels_final = self.used_channels

        mixture, tgt, num_near, num_far, target_snr, vad_gt = self.create_scene(
            idx, rng, used_channels_final)

        if num_near > 0:
            assert torch.abs(tgt).max().item() > 0

        g1 = None
        if len(used_channels_final) > 1:
            x_11 = mixture[used_channels_final[0]].unsqueeze(0)
            x_12 = mixture[used_channels_final[1]].unsqueeze(0)
            gt1  = tgt[used_channels_final[0]].unsqueeze(0)
            gt2  = tgt[used_channels_final[1]].unsqueeze(0)
            # Reduce per-channel VAD to 1D using the same channel indices as audio
            vad_gt =vad_gt[used_channels_final[0]]

            if self.split == 'train':
                mix_cat = torch.cat([x_11, x_12], dim=0)
                num_ch  = x_11.shape[0]
                x_1, gt1 = self.augmentations.apply_random_augmentations(
                    mix_cat, gt1, rng)
                x_11 = x_1[:num_ch]
                x_12 = x_1[num_ch:]

            if self.use_drc:
                avg_sign = (x_11[0] + x_12[0]) / 2
                g1, _ = drc(avg_sign.numpy(), 0.5)
                g1 = torch.from_numpy(g1).unsqueeze(0) * 0.8
                x_11 = x_11 * g1
                x_12 = x_12 * g1
                gt1  = gt1  * g1
                gt2  = gt2  * g1

            mixture = torch.cat([x_11, x_12], dim=0)
            target  = torch.cat([gt1, gt2], dim=0)
        else:
            x_11 = mixture[used_channels_final[0]].unsqueeze(0)
            gt1  = tgt[used_channels_final[0]].unsqueeze(0)
            vad_gt = vad_gt[used_channels_final[0]]   # [N_c]

            if self.split == 'train':
                x_11, gt1 = self.augmentations.apply_random_augmentations(
                    x_11, gt1, rng)

            if self.use_drc:
                avg_sign = x_11[0]
                g1, _ = drc(avg_sign.numpy(), 0.5)
                g1 = torch.from_numpy(g1).unsqueeze(0) * 0.8
                x_11 = x_11 * g1
                gt1  = gt1  * g1

            mixture = x_11
            target  = gt1

        if rng.uniform(0, 1) < self.compression_aug_ratio:
            mixture = torch.from_numpy(
                apply_codec_artifacts_multichannel(mixture.numpy(), self.sr))
            target  = torch.from_numpy(
                apply_codec_artifacts_multichannel(target.numpy(), self.sr))

        inputs = {'mixture': mixture}
        if g1 is not None:
            inputs['g1'] = g1

        targets = {
            'target': target,
            'vad_gt': torch.from_numpy(vad_gt).float(),   # [n_out_chunks]
            'num_target_speakers': num_near,
            'num_speakers': num_near + num_far,
            'target_snr': target_snr if target_snr is not None else 0.0,
        }

        return inputs, targets


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _fit_vad(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Clip or zero-pad a vad array to exactly target_len along the last (time) axis.

    Accepts 1D [N_c] or multi-channel 2D [n_ch, N_c].
    """
    if arr.ndim == 1:
        if len(arr) >= target_len:
            return arr[:target_len]
        return np.pad(arr, (0, target_len - len(arr)))
    # 2D: [n_ch, N_c] — operate on time axis
    if arr.shape[-1] >= target_len:
        return arr[..., :target_len]
    return np.pad(arr, ((0, 0), (0, target_len - arr.shape[-1])))
