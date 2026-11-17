# coding=utf-8
import zlib
import json
from dataclasses import dataclass, fields
from typing import Any, Dict, List

import librosa
import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets, interleave_datasets, load_dataset

from action_format import (
    ACTION_FIRST,
    ACTION_SPECIAL_V1,
    ACTION_JUDGE_V1,
    QUERY,
    build_legacy_target_parts,
    build_transcript_target,
    check_reserved_text,
    resolve_protocol_token_ids,
    validate_format,
)


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


def build_quality_prefix(value: Any, mode: str = "none") -> str:
    """Format optional quality features as input context excluded from loss.

    ``none`` supplies no quality text. ``manifest`` serializes the supplied
    value under the QUALITY_CONTEXT prefix without changing target labels.
    """
    if mode == "none" or value in (None, "", {}):
        return ""
    if mode != "manifest":
        raise ValueError(f"Unknown quality feature mode {mode!r}")
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return f"QUALITY_CONTEXT: {value}\n"


def _validate_actions(features, reject_weight):
    actions = [str(x["action"]).upper() for x in features]
    invalid = [(i, value) for i, value in enumerate(actions) if value not in {"RESPOND", "REJECT"}]
    if invalid:
        raise ValueError(f"Invalid action labels (expected RESPOND or REJECT): {invalid}")
    if reject_weight <= 0:
        raise ValueError(f"reject_weight must be positive, got {reject_weight}")
    return actions


def _model_prefix(processor, row, language, quality_feature_mode):
    lang_suffix = f"language {language}<asr_text>" if language else ""
    return processor.apply_chat_template(
        [audio_messages(row.get("prompt", ""))],
        add_generation_prompt=True,
        tokenize=False,
    )[0] + lang_suffix + build_quality_prefix(row.get("quality_context", ""), quality_feature_mode)


def _row_span(attention_mask, i, padding_side):
    """(start, end) of row i's real tokens, verified against attention_mask."""
    seq_len = attention_mask.size(1)
    full_len = int(attention_mask[i].sum())
    start = seq_len - full_len if padding_side == "left" else 0
    end = start + full_len
    if not bool(attention_mask[i, start:end].all()):
        raise ValueError(f"Batch row {i}: attention_mask does not match padding_side={padding_side!r}")
    return start, end


def _build_action_batch(
    processor: Any,
    features: List[Dict[str, Any]],
    audios: List[np.ndarray],
    language: str,
    quality_feature_mode: str,
    reject_weight: float,
    action_format: str = ACTION_FIRST,
) -> Dict[str, torch.Tensor]:
    validate_format(action_format)
    if action_format == ACTION_JUDGE_V1:
        return _build_judge_batch(processor, features, audios, language,
                                  quality_feature_mode, reject_weight)
    if action_format == ACTION_SPECIAL_V1:
        return _build_special_batch(
            processor, features, audios, language, quality_feature_mode, reject_weight
        )
    actions = _validate_actions(features, reject_weight)

    action_texts = []
    targets = []
    full_texts = []
    eos = processor.tokenizer.eos_token or ""
    for row, action in zip(features, actions):
        model_prefix = _model_prefix(processor, row, language, quality_feature_mode)
        action_text, transcript_text = build_legacy_target_parts(action, row["text"], eos)
        target = action_text + transcript_text
        action_texts.append(action_text)
        targets.append(target)
        full_texts.append(model_prefix + target)

    # Process the audio once and locate target spans by token-suffix matching.
    # Validate boundaries against each expanded multimodal sequence.
    batch = processor(
        text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False
    )
    target_token_ids = processor.tokenizer(targets, add_special_tokens=False)["input_ids"]
    action_token_ids = processor.tokenizer(action_texts, add_special_tokens=False)["input_ids"]

    labels = batch["input_ids"].clone()
    attention_mask = batch["attention_mask"]
    action_loss_mask = torch.zeros_like(labels, dtype=torch.bool)
    transcript_loss_mask = torch.zeros_like(labels, dtype=torch.bool)
    padding_side = getattr(processor.tokenizer, "padding_side", "right")

    for i in range(len(features)):
        start, end = _row_span(attention_mask, i, padding_side)
        full_len = end - start
        target_len = len(target_token_ids[i])
        action_len = len(action_token_ids[i])
        if not 0 < action_len < target_len < full_len:
            raise ValueError(
                "Invalid action target boundaries for batch row "
                f"{i}: action={action_len}, target={target_len}, full={full_len}"
            )
        action_start = end - target_len
        transcript_start = action_start + action_len
        actual_target_ids = labels[i, action_start:end].tolist()
        if actual_target_ids != target_token_ids[i]:
            raise ValueError(
                "Action target is not an exact tokenizer suffix for batch row "
                f"{i}; refusing to construct approximate loss boundaries"
            )
        if target_token_ids[i][:action_len] != action_token_ids[i]:
            raise ValueError(
                f"Action text is not an exact target-token prefix for batch row {i}"
            )
        labels[i, start:action_start] = -100
        action_loss_mask[i, action_start:transcript_start] = True
        transcript_loss_mask[i, transcript_start:end] = True

    # Padding is identified from attention_mask, never by token value: a
    # tokenizer with pad_token_id == eos_token_id would otherwise lose its
    # EOS supervision.
    labels[attention_mask == 0] = -100
    valid = labels.ne(-100)
    action_loss_mask &= valid
    transcript_loss_mask &= valid
    if torch.any(action_loss_mask & transcript_loss_mask):
        raise AssertionError("Action and transcript loss masks overlap")
    if not torch.all((action_loss_mask | transcript_loss_mask) == valid):
        raise AssertionError("Every active label must belong to exactly one loss span")

    batch["labels"] = labels
    batch["action_loss_mask"] = action_loss_mask
    batch["transcript_loss_mask"] = transcript_loss_mask
    batch["action_sample_weight"] = torch.tensor(
        [reject_weight if action == "REJECT" else 1.0 for action in actions],
        dtype=torch.float32,
    )
    return batch


def _build_judge_batch(processor, features, audios, language,
                       quality_feature_mode, reject_weight):
    """Build audio/candidate/EOS/query inputs with action-only supervision.

    Candidate text comes from generated_transcription. Reference text, stored
    WER, and label_hypothesis are excluded from the model input sequence.
    """
    if quality_feature_mode != "none":
        raise ValueError("Judge inputs must not contain quality/label hints")
    actions = _validate_actions(features, reject_weight)
    tokenizer = processor.tokenizer
    if tokenizer.padding_side != "left":
        raise ValueError("Judge batches require left padding")
    ids = resolve_protocol_token_ids(tokenizer)
    texts = []
    for i, row in enumerate(features):
        candidate = row.get("generated_transcription")
        if not isinstance(candidate, str):
            raise ValueError(f"Row {i} requires generated_transcription (empty string is allowed)")
        # Text may not introduce structural/audio placeholders into this sequence.
        for token in tokenizer.all_special_tokens:
            if token and token in candidate:
                raise ValueError(f"Row {i}: candidate contains reserved token {token!r}")
        if row.get("prompt") or row.get("quality_context"):
            raise ValueError("Judge v1 uses a fixed prompt; manifest hints are forbidden")
        texts.append(_model_prefix(processor, {}, language, "none") + candidate)
    batch = processor(text=texts, audio=audios, return_tensors="pt",
                      padding=True, truncation=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Judge format requires a transcript boundary EOS token")
    suffix = torch.tensor([[tokenizer.eos_token_id, ids[QUERY]]] * len(features),
                          dtype=batch["input_ids"].dtype)
    batch["input_ids"] = torch.cat([batch["input_ids"], suffix], dim=1)
    batch["attention_mask"] = torch.cat([
        batch["attention_mask"], torch.ones_like(suffix)], dim=1)
    batch["labels"] = torch.tensor([ids[a] for a in actions], dtype=torch.long)
    batch["action_sample_weight"] = torch.tensor([
        reject_weight if a == "REJECT" else 1.0 for a in actions], dtype=torch.float32)
    return batch


def _build_special_batch(
    processor: Any,
    features: List[Dict[str, Any]],
    audios: List[np.ndarray],
    language: str,
    quality_feature_mode: str,
    reject_weight: float,
) -> Dict[str, torch.Tensor]:
    """action_special_v1: two decoder passes over the same audio.

    Transcript pass (input_ids/labels): [prompt] transcript<EOS> -- exactly the
    ordinary ASR target, with prompt and padding masked, so the transcript
    never sees an action token.

    Action pass (action_input_ids/action_attention_mask): [prompt]
    <|action_query|>, ALWAYS left-padded so the query is the last column; the
    target is action_labels[i] at that position. The prompt token IDs are
    copied verbatim from the transcript pass, so both passes carry identical
    audio placeholders and the audio features are computed once.
    """
    actions = _validate_actions(features, reject_weight)
    tokenizer = processor.tokenizer
    # Never registers: the processor must already carry the tokens.
    protocol_ids = resolve_protocol_token_ids(tokenizer)
    for i, row in enumerate(features):
        for field_name in ("prompt", "quality_context", "text"):
            check_reserved_text(row.get(field_name), field_name, i)

    eos = tokenizer.eos_token or ""
    targets = [build_transcript_target(row["text"], eos) for row in features]
    full_texts = [
        _model_prefix(processor, row, language, quality_feature_mode) + target
        for row, target in zip(features, targets)
    ]
    batch = processor(
        text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False
    )
    target_token_ids = tokenizer(targets, add_special_tokens=False)["input_ids"]

    labels = batch["input_ids"].clone()
    attention_mask = batch["attention_mask"]
    padding_side = getattr(tokenizer, "padding_side", "right")
    prompts = []
    for i in range(len(features)):
        start, end = _row_span(attention_mask, i, padding_side)
        target_len = len(target_token_ids[i])
        if not 0 < target_len < end - start:
            raise ValueError(f"Invalid transcript target length for batch row {i}: "
                             f"target={target_len}, full={end - start}")
        transcript_start = end - target_len
        if labels[i, transcript_start:end].tolist() != target_token_ids[i]:
            raise ValueError(
                "Transcript target is not an exact tokenizer suffix for batch row "
                f"{i}; refusing to construct approximate loss boundaries"
            )
        prompt_ids = batch["input_ids"][i, start:transcript_start].tolist()
        if any(t in protocol_ids.values() for t in prompt_ids):
            raise AssertionError(f"Protocol token inside the prompt of batch row {i}")
        prompts.append(prompt_ids)
        labels[i, start:transcript_start] = -100
    # Padding comes from attention_mask, never token value (pad may equal EOS).
    labels[attention_mask == 0] = -100
    if any(t in protocol_ids.values() for t in labels[labels.ne(-100)].tolist()):
        raise AssertionError("A protocol token appears inside a transcript target")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    width = max(len(p) for p in prompts) + 1
    action_input_ids = torch.full((len(features), width), pad_id, dtype=batch["input_ids"].dtype)
    action_attention_mask = torch.zeros((len(features), width), dtype=attention_mask.dtype)
    for i, prompt_ids in enumerate(prompts):
        row_ids = prompt_ids + [protocol_ids[QUERY]]
        action_input_ids[i, width - len(row_ids):] = torch.tensor(row_ids)
        action_attention_mask[i, width - len(row_ids):] = 1

    batch["labels"] = labels
    batch["action_input_ids"] = action_input_ids
    batch["action_attention_mask"] = action_attention_mask
    batch["action_labels"] = torch.tensor([protocol_ids[a] for a in actions], dtype=torch.long)
    batch["action_sample_weight"] = torch.tensor(
        [reject_weight if action == "REJECT" else 1.0 for action in actions],
        dtype=torch.float32,
    )
    return batch


@dataclass
class Qwen3ASRActionCollator:
    processor: Any
    sampling_rate: int = 16000
    language: str = "English"
    quality_feature_mode: str = "none"
    reject_weight: float = 1.0
    action_format: str = ACTION_FIRST

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        sep_paths = [x.get("audio") or x.get("audio_sep") for x in features]
        audios = [read_audio(path, self.sampling_rate) for path in sep_paths]
        return _build_action_batch(
            self.processor,
            features,
            audios,
            self.language,
            self.quality_feature_mode,
            self.reject_weight,
            self.action_format,
        )


@dataclass
class Qwen3ASRActionCollatorMix:
    processor: Any
    sampling_rate: int = 16000
    language: str = "English"
    quality_feature_mode: str = "none"
    reject_weight: float = 1.0
    action_format: str = ACTION_FIRST

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        missing_sep = [i for i, row in enumerate(features) if not row.get("audio_sep")]
        missing_mix = [i for i, row in enumerate(features) if not row.get("audio_mix")]
        if missing_sep or missing_mix:
            raise ValueError(
                "Qwen3ASRActionCollatorMix requires audio_sep and audio_mix on every row; "
                f"missing audio_sep={missing_sep}, missing audio_mix={missing_mix}"
            )
        audios = [read_audio(row["audio_sep"], self.sampling_rate) for row in features]
        audios_mix = [read_audio(row["audio_mix"], self.sampling_rate) for row in features]
        batch = _build_action_batch(
            self.processor,
            features,
            audios,
            self.language,
            self.quality_feature_mode,
            self.reject_weight,
            self.action_format,
        )

        # Mixture features share the same text shape as the separated stream;
        # the model consumes only the mixture mel tensors.
        mix_texts = [
            self.processor.apply_chat_template(
                [audio_messages(row.get("prompt", ""))],
                add_generation_prompt=True,
                tokenize=False,
            )[0]
            + (f"language {self.language}<asr_text>" if self.language else "")
            + build_quality_prefix(row.get("quality_context", ""), self.quality_feature_mode)
            for row in features
        ]
        mix_batch = self.processor(
            text=mix_texts,
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
    "action_none": Qwen3ASRActionCollator,
    "action_mix": Qwen3ASRActionCollatorMix,
    "white_noise_test": Qwen3ASRCollator_WhiteNoise_Test,
}

# Collators that emit mix_input_features/mix_feature_attention_mask -- i.e.
# require a fusion-capable model (use_fusion=1). Keep in sync with COLLATORS
# when adding a new collator: a fusion-emitting collator paired with
# use_fusion=0 (plain model, no fusion_gate/conv2d1_mix) would build tensors
# the model never consumes; a non-fusion collator paired with use_fusion=1
# would silently run the fusion model as single-stream for the whole run --
# both are exactly the kind of desync build_collator() rejects below.
FUSION_COLLATORS = {"mix", "action_mix", "white_noise_test"}


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


def _parse_data_spec(spec: str) -> List[Any]:
    """"path" or "path:weight,path2:weight2,..." -> [(path, weight), ...].
    Bare path (no ':') defaults to weight 1.0, so single-file specs are
    unaffected."""
    entries = []
    for item in spec.split(","):
        item = item.strip()
        if ":" in item:
            path, _, weight_str = item.rpartition(":")
            weight = float(weight_str)
        else:
            path, weight = item, 1.0
        if weight <= 0:
            raise ValueError(f"weight for {path!r} must be > 0, got {weight}")
        entries.append((path, weight))
    return entries


def _load_weighted(spec: str):
    """Single file -> that dataset as-is. Multiple -> probability-weighted
    interleave (stopping_strategy=all_exhausted: oversamples low-weight files
    so every file is fully seen at least once per epoch at the target
    proportions, instead of truncating the epoch at whichever file runs out
    first)."""
    entries = _parse_data_spec(spec)
    if len(entries) == 1:
        path, _ = entries[0]
        return load_dataset("json", data_files=path)["train"]
    datasets_list = [load_dataset("json", data_files=path)["train"] for path, _ in entries]
    total = sum(w for _, w in entries)
    probabilities = [w / total for _, w in entries]
    return interleave_datasets(
        datasets_list, probabilities=probabilities, seed=42,
        stopping_strategy="all_exhausted",
    )


def _load_concatenated(spec: str):
    """Ignores weights -- every listed file contributes all its rows, so eval
    always covers 100% of every validation file, deterministically."""
    entries = _parse_data_spec(spec)
    if len(entries) == 1:
        path, _ = entries[0]
        return load_dataset("json", data_files=path)["train"]
    datasets_list = [load_dataset("json", data_files=path)["train"] for path, _ in entries]
    return concatenate_datasets(datasets_list)


def build_datasets(train_file: str, eval_file: str = ""):
    splits = {"train": _load_weighted(train_file)}
    if eval_file:
        splits["validation"] = _load_concatenated(eval_file)
    return DatasetDict(splits)
