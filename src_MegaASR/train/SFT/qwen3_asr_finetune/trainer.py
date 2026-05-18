# coding=utf-8
import os

import torch
from safetensors.torch import load_file as safe_load_file
from transformers import Trainer

from .sampler import FixedRatioTrainLoaderMixin


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


class AdapterOnlyTrainer(CastFloatInputsTrainer):
    def __init__(
        self,
        *args,
        processor=None,
        base_model_path: str = "",
        merged_from_lora_path: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._processor = processor
        self._base_model_path = base_model_path
        self._merged_from_lora_path = merged_from_lora_path

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1) 保存 adapter（关键）
        self.model.thinker.save_pretrained(output_dir, safe_serialization=True)

        # 2) 也把 processor/tokenizer 存一份（方便部署/复现实验）
        if self._processor is not None:
            self._processor.save_pretrained(output_dir)

        # 3) 记录 base 模型（部署时 base+adapter 组合需要）
        if self._base_model_path:
            with open(os.path.join(output_dir, "base_model.txt"), "w", encoding="utf-8") as f:
                f.write(self._base_model_path + "\n")

        # 4) 防御：如果 Trainer 其他逻辑写出了大权重文件，删掉
        for fn in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"]:
            fp = os.path.join(output_dir, fn)
            if os.path.exists(fp):
                os.remove(fp)

        if self._merged_from_lora_path:
            with open(os.path.join(output_dir, "merged_from_lora.txt"), "w", encoding="utf-8") as f:
                f.write(self._merged_from_lora_path + "\n")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        支持从 adapter-only checkpoint 恢复：把 adapter 权重加载回 model.thinker。
        """
        model = model or self.model
        adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
        if os.path.isfile(adapter_path):
            sd = safe_load_file(adapter_path)
            model.thinker.load_state_dict(sd, strict=False)
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)


class MultiLRAdapterTrainer(AdapterOnlyTrainer):
    def __init__(
        self,
        *args,
        processor=None,
        base_model_path: str = "",
        merged_from_lora_path: str = "",
        lr_tower: float = 1e-4,
        lr_proj: float = 3e-4,
        lr_llm: float = 5e-5,
        **kwargs,
    ):
        super().__init__(
            *args,
            processor=processor,
            base_model_path=base_model_path,
            merged_from_lora_path=merged_from_lora_path,
            **kwargs,
        )
        self.lr_tower = lr_tower
        self.lr_proj = lr_proj
        self.lr_llm = lr_llm

    @staticmethod
    def _is_lora_param(name: str) -> bool:
        return "lora_" in name

    @staticmethod
    def _is_proj_param(name: str) -> bool:
        return (
            "audio_tower.conv_out" in name
            or "audio_tower.proj1" in name
            or "audio_tower.proj2" in name
        )

    @staticmethod
    def _is_tower_param(name: str) -> bool:
        return "audio_tower.layers." in name

    @staticmethod
    def _is_llm_param(name: str) -> bool:
        # 这里不要匹配 audio_tower.layers
        return ("model.layers." in name) and ("audio_tower.layers." not in name)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        tower_params = []
        proj_params = []
        llm_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # 只对 LoRA 参数做分组；其他可训练参数走 other
            if self._is_lora_param(name):
                if self._is_proj_param(name):
                    proj_params.append(param)
                elif self._is_tower_param(name):
                    tower_params.append(param)
                elif self._is_llm_param(name):
                    llm_params.append(param)
                else:
                    other_params.append(param)
            else:
                other_params.append(param)

        optim_groups = []
        if tower_params:
            optim_groups.append({
                "params": tower_params,
                "lr": self.lr_tower,
                "weight_decay": self.args.weight_decay,
            })
        if proj_params:
            optim_groups.append({
                "params": proj_params,
                "lr": self.lr_proj,
                "weight_decay": self.args.weight_decay,
            })
        if llm_params:
            optim_groups.append({
                "params": llm_params,
                "lr": self.lr_llm,
                "weight_decay": self.args.weight_decay,
            })
        if other_params:
            optim_groups.append({
                "params": other_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            })

        # 打印一下，方便确认是否真的分组成功
        if self.args.process_index == 0:
            print(f"[optimizer] tower params: {sum(p.numel() for p in tower_params)}")
            print(f"[optimizer] proj  params: {sum(p.numel() for p in proj_params)}")
            print(f"[optimizer] llm   params: {sum(p.numel() for p in llm_params)}")
            print(f"[optimizer] other params: {sum(p.numel() for p in other_params)}")
            print(f"[optimizer] lr_tower={self.lr_tower}, lr_proj={self.lr_proj}, lr_llm={self.lr_llm}, lr_other={self.args.learning_rate}")

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer


class FixedRatioMultiLRAdapterTrainer(FixedRatioTrainLoaderMixin, MultiLRAdapterTrainer):
    pass


class FixedRatioCastTrainer(FixedRatioTrainLoaderMixin, CastFloatInputsTrainer):
    pass
