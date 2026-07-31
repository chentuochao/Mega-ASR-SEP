import torch
import numpy as np


# Empirical distribution of high-frequency suppression magnitudes (from separation model analysis)
_SUPPRESSION_BUCKETS = [
    {"lo_db": 0,  "hi_db": 1,  "pct": 9.59},
    {"lo_db": 1,  "hi_db": 2,  "pct": 9.21},
    {"lo_db": 2,  "hi_db": 3,  "pct": 9.77},
    {"lo_db": 3,  "hi_db": 4,  "pct": 9.22},
    {"lo_db": 4,  "hi_db": 5,  "pct": 8.66},
    {"lo_db": 5,  "hi_db": 6,  "pct": 7.44},
    {"lo_db": 6,  "hi_db": 7,  "pct": 6.44},
    {"lo_db": 7,  "hi_db": 8,  "pct": 5.70},
    {"lo_db": 8,  "hi_db": 9,  "pct": 5.26},
    {"lo_db": 9,  "hi_db": 10, "pct": 4.29},
    {"lo_db": 10, "hi_db": 11, "pct": 4.00},
    {"lo_db": 11, "hi_db": 12, "pct": 3.89},
    {"lo_db": 12, "hi_db": 13, "pct": 3.59},
    {"lo_db": 13, "hi_db": 14, "pct": 2.46},
    {"lo_db": 14, "hi_db": 15, "pct": 1.96},
    {"lo_db": 15, "hi_db": 16, "pct": 1.59},
    {"lo_db": 16, "hi_db": 17, "pct": 1.11},
    {"lo_db": 17, "hi_db": 18, "pct": 1.33},
    {"lo_db": 18, "hi_db": 19, "pct": 1.37},
    {"lo_db": 19, "hi_db": 20, "pct": 0.93},
    {"lo_db": 20, "hi_db": 21, "pct": 0.74},
    {"lo_db": 21, "hi_db": 22, "pct": 0.74},
    {"lo_db": 22, "hi_db": 23, "pct": 0.81},
    {"lo_db": 23, "hi_db": 24, "pct": 0.81},
    {"lo_db": 24, "hi_db": 25, "pct": 0.74},
    {"lo_db": 25, "hi_db": 26, "pct": 0.48},
    {"lo_db": 26, "hi_db": 27, "pct": 0.59},
    {"lo_db": 27, "hi_db": 28, "pct": 0.19},
    {"lo_db": 28, "hi_db": 29, "pct": 0.37},
    {"lo_db": 29, "hi_db": 30, "pct": 0.33},
    {"lo_db": 30, "hi_db": 31, "pct": 0.25},
    {"lo_db": 31, "hi_db": 32, "pct": 0.19},
    {"lo_db": 32, "hi_db": 33, "pct": 0.15},
    {"lo_db": 33, "hi_db": 34, "pct": 0.17},
    {"lo_db": 34, "hi_db": 35, "pct": 0.17},
    {"lo_db": 35, "hi_db": 36, "pct": 0.15},
    {"lo_db": 36, "hi_db": 37, "pct": 0.14},
    {"lo_db": 37, "hi_db": 100, "pct": 0.17},
]

_BUCKET_PROBS = None


def _get_bucket_probs():
    global _BUCKET_PROBS
    if _BUCKET_PROBS is None:
        raw = np.array([b["pct"] for b in _SUPPRESSION_BUCKETS])
        _BUCKET_PROBS = raw / raw.sum()
    return _BUCKET_PROBS


def _sample_suppression_db(rng: np.random.RandomState) -> float:
    probs = _get_bucket_probs()
    idx = rng.choice(len(_SUPPRESSION_BUCKETS), p=probs)
    bucket = _SUPPRESSION_BUCKETS[idx]
    return rng.uniform(bucket["lo_db"], bucket["hi_db"])


class HighFrequencySuppressionAugmentation:
    """
    Simulates the high-frequency over-suppression artifact introduced by a
    separation model.  For every 0.5 s chunk of the spectrogram that contains
    active speech, a random high-frequency roll-off (sampled from the empirical
    bucket distribution) is applied with probability p_per_chunk.

    Follows the same (audio_data, gt_audio, rng) call interface as other
    augmentations in this package.
    """

    def __init__(
        self,
        p_per_chunk: float = 0.5,
        chunk_duration_s: float = 0.5,
        sample_rate: int = 16000,
        nfft: int = 512,
        hop_length: int = None,
        cutoff_freq_hz: float = 2000.0,
        silence_threshold_db: float = -20.0,
        reference_channels=None,
        aggressive_factor: float = 1.0
    ):
        """
        Args:
            p_per_chunk: Probability of applying suppression to each active chunk.
            chunk_duration_s: Duration of each analysis chunk in seconds.
            sample_rate: Audio sample rate in Hz.
            nfft: FFT size for STFT.
            hop_length: STFT hop; defaults to nfft // 4.
            cutoff_freq_hz: Frequencies above this value (Hz) are attenuated.
            silence_threshold_db: A chunk whose power is this many dB below the
                mean power of the whole signal is treated as silence and skipped.
            reference_channels: Indices of audio_data channels whose corresponding
                gt_audio channel should receive the same suppression.  Defaults to
                [0, 1] to mirror the convention in other augmentations.
        """
        self.p_per_chunk = p_per_chunk
        self.chunk_duration_s = chunk_duration_s
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.hop_length = hop_length if hop_length is not None else nfft // 4
        self.cutoff_freq_hz = cutoff_freq_hz
        self.silence_threshold_db = silence_threshold_db
        self.reference_channels = reference_channels if reference_channels is not None else [0, 1]
        self.aggressive_factor = aggressive_factor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cutoff_bin(self, n_freqs: int) -> int:
        bin_idx = int(self.cutoff_freq_hz / (self.sample_rate / 2.0) * (n_freqs - 1))
        return min(max(bin_idx, 0), n_freqs - 1)

    def _process_channel(self, audio_1d: torch.Tensor, rng: np.random.RandomState) -> torch.Tensor:
        T = audio_1d.shape[-1]
        window = torch.hamming_window(self.nfft, device=audio_1d.device)

        S = torch.stft(
            audio_1d,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            return_complex=True,
            window=window,
        )
        # S: (n_freqs, n_frames)

        n_freqs, n_frames = S.shape
        cutoff_bin = self._cutoff_bin(n_freqs)

        # Reference power for silence detection (use full spectrogram)
        mean_power = (S.abs() ** 2).mean().item()
        mean_power = max(mean_power, 1e-10)

        frames_per_chunk = max(1, int(self.chunk_duration_s * self.sample_rate / self.hop_length))

        # Pass 1: collect the (start, end) ranges of active chunks that fire on p_per_chunk.
        # We defer sampling so we can sort the dB values before applying them, producing a
        # smooth monotonic ramp rather than random jumps between adjacent chunks.
        to_suppress = []
        for start in range(0, n_frames, frames_per_chunk):
            end = min(start + frames_per_chunk, n_frames)
            chunk_power = (S[:, start:end].abs() ** 2).mean().item()
            chunk_db_rel = 10.0 * np.log10(max(chunk_power, 1e-10) / mean_power)
            if chunk_db_rel < self.silence_threshold_db:
                continue
            if rng.random() < self.p_per_chunk:
                to_suppress.append((start, end))
        if to_suppress:
            # Sample all dB values at once, sort for smooth variation, then roll
            # to randomise which part of the ramp aligns with the start of speech.
            suppression_dbs = np.sort([_sample_suppression_db(rng) for _ in to_suppress]) * self.aggressive_factor
            if rng.random() < 0.5:
                suppression_dbs = suppression_dbs[::-1].copy()
            shift = rng.randint(0, len(suppression_dbs))
            suppression_dbs = np.concatenate([suppression_dbs[shift:], suppression_dbs[:shift]])

            # Pass 2: apply each sorted value to its chunk in temporal order.
            for (start, end), db in zip(to_suppress, suppression_dbs):
                scale = 10.0 ** (-db / 20.0)
                S[cutoff_bin:, start:end] *= scale

        audio_out = torch.istft(
            S,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            length=T,
            window=window,
        )
        return audio_out

    def remove_all(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Zero out all STFT bins above cutoff_freq_hz for every channel."""
        C = audio_data.shape[0]
        for i in range(C):
            T = audio_data[i].shape[-1]
            window = torch.hamming_window(self.nfft, device=audio_data.device)
            S = torch.stft(
                audio_data[i],
                n_fft=self.nfft,
                hop_length=self.hop_length,
                return_complex=True,
                window=window,
            )
            S[self._cutoff_bin(S.shape[0]):] = 0.0
            audio_data[i] = torch.istft(
                S,
                n_fft=self.nfft,
                hop_length=self.hop_length,
                length=T,
                window=window,
            )
        return audio_data

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        audio_data: torch.Tensor,
        rng: np.random.RandomState,
    ):
        C = audio_data.shape[0]

        for i in range(C):
            audio_data[i] = self._process_channel(audio_data[i], rng)

        return audio_data,
