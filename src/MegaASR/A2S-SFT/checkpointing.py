# coding=utf-8
import os
import re
import shutil
from typing import Optional

from transformers import TrainerCallback, TrainingArguments


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None

    best_step, best_path = -1, None
    for name in os.listdir(output_dir):
        match = _CKPT_RE.match(name)
        if not match:
            continue
        path = os.path.join(output_dir, name)
        step = int(match.group(1))
        if os.path.isdir(path) and step > best_step:
            best_step, best_path = step, path
    return best_path


# Processor files whose checkpoint-local copies contain the action-token mapping.
# Preserve these files when copying base-model configuration into a checkpoint.
PROCESSOR_FILES = {
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "chat_template.json",
    "merges.txt",
    "vocab.json",
    "added_tokens.json",
}


def copy_hf_files(src_dir: str, dst_dir: str, preserve_processor: bool = False):
    os.makedirs(dst_dir, exist_ok=True)
    for name in [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]:
        if preserve_processor and name in PROCESSOR_FILES:
            continue
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


class MakeCheckpointInferableCallback(TrainerCallback):
    """Copy tokenizer/config files into every adapter checkpoint."""

    def __init__(self, base_model_path: str, preserve_processor: bool = False):
        self.base_model_path = base_model_path
        self.preserve_processor = preserve_processor

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_hf_files(self.base_model_path, ckpt_dir, preserve_processor=self.preserve_processor)
        return control
