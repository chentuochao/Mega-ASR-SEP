# coding=utf-8
from dataclasses import dataclass
from typing import Any, Dict, List

import librosa
import torch
from datasets import load_dataset


def read_audio(path: str, sr: int = 16000):
    return librosa.load(path, sr=sr, mono=True)[0]


def audio_messages(prompt: str):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]


@dataclass
class Qwen3ASRCollator:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [x.get("prompt", "") for x in features]
        targets = [x["text"] for x in features]

        # Separated stream is the primary input. Accept the legacy single-stream
        # key "audio" as an alias for "audio_sep" so old manifests still load.
        sep_paths = [x.get("audio_sep", x.get("audio")) for x in features]
        audios = [read_audio(p, self.sampling_rate) for p in sep_paths]

        # Mixture (pre-separation) stream, optional. Fusion only kicks in when
        # every example in the batch carries an "audio_mix" path; otherwise we
        # fall back to the single-stream path for this batch.
        have_mix = all(x.get("audio_mix") for x in features)
        audios_mix = (
            [read_audio(x["audio_mix"], self.sampling_rate) for x in features]
            if have_mix else None
        )

        prefixes = [
            self.processor.apply_chat_template(
                [audio_messages(p)],
                add_generation_prompt=True,
                tokenize=False,
            )[0]
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

        if audios_mix is not None:
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


def build_datasets(train_file: str, eval_file: str = ""):
    files = {"train": train_file}
    if eval_file:
        files["validation"] = eval_file
    return load_dataset("json", data_files=files)
