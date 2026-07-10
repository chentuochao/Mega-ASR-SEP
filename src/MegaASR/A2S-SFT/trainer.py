# coding=utf-8
import os
from typing import Optional

import torch
from safetensors.torch import load_file as safe_load_file
from transformers import Trainer


class MegaASRTrainer(Trainer):
    """Trainer for Mega-ASR LoRA SFT."""

    def __init__(self, *args, processor=None, base_model_path: str = "",
                 merged_from_lora_path: str = "", lr_encoder: float = 1e-5,
                 lr_aligner: float = 1e-5, lr_llm: float = 1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.base_model_path = base_model_path
        self.merged_from_lora_path = merged_from_lora_path
        self.lr_encoder = lr_encoder
        self.lr_aligner = lr_aligner
        self.lr_llm = lr_llm

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        dtype = getattr(self.model, "dtype", None)
        if dtype is None:
            return inputs
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dtype=dtype)
        return inputs

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(self.model.thinker, "peft_config"):
            # LoRA: PeftModel.save_pretrained writes adapter_model.safetensors
            # keyed relative to the thinker (base_model.model.*), which is what
            # PeftModel.from_pretrained expects when re-applying the adapter
            # on top of a separately-loaded base model.
            self.model.thinker.save_pretrained(output_dir, safe_serialization=True)
        else:
            # Full fine-tune: save the TOP-LEVEL wrapper, not just `.thinker`.
            # `model.thinker.state_dict()` keys have no "thinker." prefix, but
            # the copied config.json (below, via MakeCheckpointInferableCallback)
            # describes the wrapper class (Qwen3ASRForConditionalGeneration),
            # whose from_pretrained expects "thinker.*"-prefixed keys. Saving
            # only `.thinker` here would silently produce a checkpoint where
            # every key mismatches on reload -- from_pretrained then
            # re-initializes ~everything instead of raising, so this fails
            # silently rather than loudly. Save the wrapper so keys line up.
            self.model.save_pretrained(output_dir, safe_serialization=True)

        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
        self._write_text(output_dir, "base_model.txt", self.base_model_path)
        self._write_text(output_dir, "merged_from_lora.txt", self.merged_from_lora_path)

        if hasattr(self.model.thinker, "peft_config"):
            # LoRA adapter checkpoint: save_pretrained already wrote
            # adapter_model.safetensors. Strip any full-model files that may be
            # left over in this output_dir from a previous full-fine-tune run
            # (or an old checkpoint layout) -- NOT run for a full fine-tune
            # itself, where these files ARE the actual trained weights.
            for name in ["model.safetensors", "pytorch_model.bin",
                         "model.safetensors.index.json", "pytorch_model.bin.index.json"]:
                path = os.path.join(output_dir, name)
                if os.path.exists(path):
                    os.remove(path)

    @staticmethod
    def _write_text(output_dir: str, name: str, text: str):
        if text:
            with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
                f.write(text + "\n")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model or self.model
        adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
        if os.path.isfile(adapter_path):
            model.thinker.load_state_dict(safe_load_file(adapter_path), strict=False)
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    @staticmethod
    def _group_name(name: str) -> str:
        # fusion_gate / conv2d1_mix operate directly on the audio tower's input
        # or output; train them on the encoder LR schedule. Both are full-rank
        # modules (via modules_to_save), so they have no "lora_" in their name
        # and must be matched before the LoRA check below.
        if "fusion_gate" in name or "conv2d1_mix" in name:
            return "encoder"
        if any(x in name for x in ["audio_tower.conv_out", "audio_tower.proj1", "audio_tower.proj2"]):
            return "aligner"
        if "lora_" in name:
            if "audio_tower.layers." in name:
                return "encoder"
            if "model.layers." in name and "audio_tower.layers." not in name:
                return "llm"
            return "other"
        # No "lora_" in the name: either full-parameter (no-LoRA) fine-tuning,
        # or a base weight untouched by LoRA. Group by raw module path so
        # --lr_encoder/--lr_aligner/--lr_llm still apply to a full fine-tune.
        if "audio_tower." in name:
            return "encoder"  # transformer layers, stem convs, ln_post, etc.
        if name.startswith("model.") or name.startswith("lm_head."):
            return "llm"
        return "other"

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        groups = {"encoder": [], "aligner": [], "llm": [], "other": []}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                groups[self._group_name(name)].append(param)

        lrs = {"encoder": self.lr_encoder, "aligner": self.lr_aligner,
               "llm": self.lr_llm, "other": self.args.learning_rate}
        optim_groups = [
            {"params": params, "lr": lrs[name], "weight_decay": self.args.weight_decay}
            for name, params in groups.items() if params
        ]

        if self.args.process_index == 0:
            for name, params in groups.items():
                print(f"[optimizer] {name:7s}: {sum(p.numel() for p in params)} params")

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer
