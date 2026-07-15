# coding=utf-8
import zlib
from dataclasses import dataclass, fields
from typing import Any, Dict, List

import librosa
import numpy as np
import torch
from datasets import load_dataset


def read_audio(path: str, sr: int = 16000):
    return librosa.load(path, sr=sr, mono=True)[0]


def audio_messages(prompt: str):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]


def _stable_seed(path: str, base_seed: int) -> int:
    """Deterministic per-path seed (Python's str hash() is randomized per
    process, so it can't be used here) -- lets the same corruption pattern
    reproduce across separate eval runs, e.g. to compare fusion types on
    identical corrupted audio."""
    return (zlib.crc32(path.encode("utf-8")) ^ base_seed) & 0xFFFFFFFF


def _add_extreme_white_noise(
    chunk: np.ndarray,
    rng: np.random.Generator,
    noise_ratio: float,
    max_amplitude: float,
) -> np.ndarray:
    """Overwhelm `chunk` with additive white noise at `noise_ratio` times the
    chunk's own RMS, so it's unintelligible regardless of the source's
    absolute loudness. Clipped to +/-max_amplitude afterward: float PCM audio
    is conventionally in [-1, 1], and unclipped noise at a large noise_ratio
    can blow well past that and saturate on write/playback."""
    rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) + 1e-8)
    noise = rng.normal(0.0, rms * noise_ratio, size=chunk.shape)
    noisy = np.clip(chunk.astype(np.float64) + noise, -max_amplitude, max_amplitude)
    return noisy.astype(chunk.dtype)


def _apply_disjoint_noise_masks(
    chan_a: np.ndarray,
    chan_b: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    chunk_sec: float,
    mask_prob: float,
    noise_ratio: float,
    max_amplitude: float,
) -> None:
    """Corrupt `chan_a`/`chan_b` in place with additive white noise, chunked
    into `chunk_sec`-second windows. Each chunk independently masks channel A,
    channel B, or neither -- NEVER both -- so every instant has clean signal
    in at least one channel. This is the invariant the whole diagnostic
    depends on: if fusion can't recover near-clean WER here, it isn't
    combining information across the two streams at all."""
    n = min(len(chan_a), len(chan_b))
    chunk_len = max(1, int(round(chunk_sec * sr)))
    for start in range(0, n, chunk_len):
        end = min(start + chunk_len, n)
        r = rng.random()
        if r < mask_prob / 2:
            chan_a[start:end] = _add_extreme_white_noise(chan_a[start:end], rng, noise_ratio, max_amplitude)
        elif r < mask_prob:
            chan_b[start:end] = _add_extreme_white_noise(chan_b[start:end], rng, noise_ratio, max_amplitude)
        # else: leave this chunk clean in both channels



@dataclass
class Qwen3ASRCollator:
    processor: Any
    sampling_rate: int = 16000
    # Forced language tag, mirroring qwen_asr's transcribe(..., language=...):
    # appended to the prefix (not the target) so the frozen LLM decoder sees
    # the same "language X<asr_text>" forcing condition at train and eval
    # time, and the target stays plain text. "" disables forcing (auto mode).
    language: str = "English"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [x.get("prompt", "") for x in features]
        targets = [x["text"] for x in features]
        # Accept "audio_sep" as an alias for "audio" so manifests written for
        # the mix collator also load here. Uses truthy `or` (not dict.get's
        # key-presence default) so "audio": null still falls through.
        sep_paths = [x.get("audio") or x.get("audio_sep") for x in features]
        audios = [read_audio(p, self.sampling_rate) for p in sep_paths]

        lang_suffix = f"language {self.language}<asr_text>" if self.language else ""
        prefixes = [
            self.processor.apply_chat_template(
                [audio_messages(p)],
                add_generation_prompt=True,
                tokenize=False,
            )[0] + lang_suffix
            for p in prompts
        ]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefixes, targets)]

        batch = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_batch = self.processor(
            text=prefixes,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = batch["input_ids"].clone()
        prefix_lens = prefix_batch["attention_mask"].sum(dim=1)
        full_lens = batch["attention_mask"].sum(dim=1)

        seq_len = labels.size(1)
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")

        for i, prefix_len in enumerate(prefix_lens):
            start = seq_len - int(full_lens[i]) if padding_side == "left" else 0
            labels[i, start:start + int(prefix_len)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch["labels"] = labels
        return batch
    
@dataclass
class Qwen3ASRCollatorMix:
    processor: Any
    sampling_rate: int = 16000
    # Forced language tag, mirroring qwen_asr's transcribe(..., language=...):
    # appended to the prefix (not the target) so the frozen LLM decoder sees
    # the same "language X<asr_text>" forcing condition at train and eval
    # time, and the target stays plain text. "" disables forcing (auto mode).
    language: str = "English"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [x.get("prompt", "") for x in features]
        targets = [x["text"] for x in features]

        # Prefer the legacy single-stream key "audio" when present; fall back to
        # "audio_sep" otherwise. Uses truthy `or` (not dict.get's key-presence
        # default) so a row with "audio": null still falls through to "audio_sep".
        sep_paths = [x.get("audio_sep") for x in features]
        audios = [read_audio(p, self.sampling_rate) for p in sep_paths]

        # Mixture (pre-separation) stream, required by this collator -- fail
        # loudly on a bad row instead of silently degrading the whole batch
        # to single-stream (which is what used to happen here).
        missing = [i for i, x in enumerate(features) if not x.get("audio_mix")]
        assert not missing, (
            f"Qwen3ASRCollatorMix requires 'audio_mix' on every row; "
            f"batch indices missing it: {missing}"
        )
        audios_mix = [read_audio(x["audio_mix"], self.sampling_rate) for x in features]

        lang_suffix = f"language {self.language}<asr_text>" if self.language else ""
        prefixes = [
            self.processor.apply_chat_template(
                [audio_messages(p)],
                add_generation_prompt=True,
                tokenize=False,
            )[0] + lang_suffix
            for p in prompts
        ]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefixes, targets)]

        batch = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_batch = self.processor(
            text=prefixes,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = batch["input_ids"].clone()
        prefix_lens = prefix_batch["attention_mask"].sum(dim=1)
        full_lens = batch["attention_mask"].sum(dim=1)

        seq_len = labels.size(1)
        # NOTE: qwen_asr's Qwen3ASRProcessorKwargs hardcodes padding_side="left"
        # for the combined audio+text call unconditionally (see
        # processing_qwen3_asr.py), so this must be "left" to match reality.
        # --padding_side left (arguments.py's default) sets this attribute via
        # finetune.py; keep the two in sync if either side ever changes.
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")

        for i, prefix_len in enumerate(prefix_lens):
            start = seq_len - int(full_lens[i]) if padding_side == "left" else 0
            labels[i, start:start + int(prefix_len)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch["labels"] = labels

        # Feature-extract the mixture stream through the *same* processor so
        # the mel features are formatted identically to the separated stream.
        # Only the audio features are kept; the mixture's text ids/mask are
        # discarded (the LLM never sees the mixture as a separate span).
        mix_batch = self.processor(
            text=prefixes,
            audio=audios_mix,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        batch["mix_input_features"] = mix_batch["input_features"]
        batch["mix_feature_attention_mask"] = mix_batch["feature_attention_mask"]

        return batch


@dataclass
class Qwen3ASRCollator_WhiteNoise_Test:
    """Diagnostic probe, NOT a training collator: both streams start as the
    SAME clean reference audio ("libritts_path"), then get corrupted with
    disjoint extreme-white-noise chunks -- channel A (sep) and channel B
    (mix) are never masked at the same instant, so the clean word is always
    recoverable from at least one stream. If fusion can't reach near-clean
    WER here, it isn't actually combining information across the two
    channels; if it does, the real-world bottleneck is separation quality /
    gate capacity, not the fusion mechanism's ability to use two streams."""
    # Fixed diagnostic-design constants -- not CLI-tunable, deliberately: these
    # define what the probe measures, not a training hyperparameter to sweep.
    CHUNK_SEC = 1.0      # window size for the disjoint noise-masking grid
    MASK_PROB = 0.64      # P(a given chunk is corrupted in exactly ONE channel)
    NOISE_RATIO = 15.0   # noise RMS = NOISE_RATIO * chunk's own RMS ("extremely high")
    MAX_AMPLITUDE = 0.8  # clip masked samples to +/-this, avoiding saturation

    processor: Any
    sampling_rate: int = 16000
    # Forced language tag, mirroring qwen_asr's transcribe(..., language=...):
    # appended to the prefix (not the target) so the frozen LLM decoder sees
    # the same "language X<asr_text>" forcing condition at train and eval
    # time, and the target stays plain text. "" disables forcing (auto mode).
    language: str = "English"
    seed: int = 0  # base seed; combined with a stable per-path hash (see _stable_seed)

    def _load_masked_channels(self, features: List[Dict[str, Any]]):
        """Load the clean reference audio for each row and corrupt the sep/mix
        channels with disjoint white-noise masks. Split out from __call__ so
        it can be exercised directly -- e.g. to dump the masked audio for
        manual listening -- without needing a real processor."""
        clean_paths = [x["libritts_path"] for x in features]
        audios = [read_audio(p, self.sampling_rate) for p in clean_paths]
        audios_mix = [read_audio(p, self.sampling_rate) for p in clean_paths]

        for sep_chunk, mix_chunk, path in zip(audios, audios_mix, clean_paths):
            rng = np.random.default_rng(_stable_seed(path, self.seed))
            _apply_disjoint_noise_masks(
                sep_chunk, mix_chunk, self.sampling_rate, rng,
                chunk_sec=self.CHUNK_SEC, mask_prob=self.MASK_PROB,
                noise_ratio=self.NOISE_RATIO, max_amplitude=self.MAX_AMPLITUDE,
            )
        return audios, audios_mix, clean_paths

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [x.get("prompt", "") for x in features]
        targets = [x["text"] for x in features]

        audios, audios_mix, clean_paths = self._load_masked_channels(features)

        lang_suffix = f"language {self.language}<asr_text>" if self.language else ""
        prefixes = [
            self.processor.apply_chat_template(
                [audio_messages(p)],
                add_generation_prompt=True,
                tokenize=False,
            )[0] + lang_suffix
            for p in prompts
        ]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefixes, targets)]

        batch = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_batch = self.processor(
            text=prefixes,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = batch["input_ids"].clone()
        prefix_lens = prefix_batch["attention_mask"].sum(dim=1)
        full_lens = batch["attention_mask"].sum(dim=1)

        seq_len = labels.size(1)
        # NOTE: qwen_asr's Qwen3ASRProcessorKwargs hardcodes padding_side="left"
        # for the combined audio+text call unconditionally (see
        # processing_qwen3_asr.py), so this must be "left" to match reality.
        # --padding_side left (arguments.py's default) sets this attribute via
        # finetune.py; keep the two in sync if either side ever changes.
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")

        for i, prefix_len in enumerate(prefix_lens):
            start = seq_len - int(full_lens[i]) if padding_side == "left" else 0
            labels[i, start:start + int(prefix_len)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch["labels"] = labels

        # Feature-extract the mixture stream through the *same* processor so
        # the mel features are formatted identically to the separated stream.
        # Only the audio features are kept; the mixture's text ids/mask are
        # discarded (the LLM never sees the mixture as a separate span).
        mix_batch = self.processor(
            text=prefixes,
            audio=audios_mix,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        batch["mix_input_features"] = mix_batch["input_features"]
        batch["mix_feature_attention_mask"] = mix_batch["feature_attention_mask"]

        return batch



COLLATORS = {
    "none": Qwen3ASRCollator,
    "mix": Qwen3ASRCollatorMix,
    "white_noise_test": Qwen3ASRCollator_WhiteNoise_Test,
}

# Collators that emit mix_input_features/mix_feature_attention_mask -- i.e.
# require a fusion-capable model (use_fusion=1). Keep in sync with COLLATORS
# when adding a new collator: a fusion-emitting collator paired with
# use_fusion=0 (plain model, no fusion_gate/conv2d1_mix) would build tensors
# the model never consumes; a non-fusion collator paired with use_fusion=1
# would silently run the fusion model as single-stream for the whole run --
# both are exactly the kind of desync build_collator() rejects below.
FUSION_COLLATORS = {"mix", "white_noise_test"}


def build_collator(name: str, use_fusion: bool, **kwargs) -> Any:
    """Look up a collator class by config name and construct it, dropping any
    kwarg the target class doesn't declare (so callers can pass one shared
    kwargs dict -- e.g. `language=...` -- across collators with different
    fields). Add a new collator by defining the dataclass above, adding one
    entry to COLLATORS, and (if it emits a mixture stream) to FUSION_COLLATORS.
    """
    try:
        cls = COLLATORS[name]
    except KeyError:
        raise ValueError(f"Unknown collator '{name}'; choices: {sorted(COLLATORS)}")

    is_fusion_collator = name in FUSION_COLLATORS
    if use_fusion and not is_fusion_collator:
        raise ValueError(
            f"--use_fusion 1 requires a fusion-capable collator (one of "
            f"{sorted(FUSION_COLLATORS)}), got collator={name!r}. That "
            f"collator never emits mix_input_features, so the fusion-capable "
            f"model would silently train/eval as single-stream the whole run."
        )
    if not use_fusion and is_fusion_collator:
        raise ValueError(
            f"--use_fusion 0 loads the plain upstream model (no fusion_gate/"
            f"conv2d1_mix params), so collator={name!r} would build "
            f"mix_input_features/mix_feature_attention_mask the model can't "
            f"consume. Use collator='none' (or --collator auto) with "
            f"--use_fusion 0."
        )

    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in kwargs.items() if k in valid})


def build_datasets(train_file: str, eval_file: str = ""):
    files = {"train": train_file}
    if eval_file:
        files["validation"] = eval_file
    return load_dataset("json", data_files=files)
