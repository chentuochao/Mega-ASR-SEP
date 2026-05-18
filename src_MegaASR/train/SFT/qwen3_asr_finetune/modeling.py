# coding=utf-8
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import GenerationConfig

from qwen_asr import Qwen3ASRModel


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def load_qwen3_asr_model_and_processor(model_path: str):
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor
    print("padding_side =", processor.tokenizer.padding_side)

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    return model, processor, use_bf16


def get_lora_target_regex(lora_scope: str) -> str:
    if lora_scope == "tower":
        target_regex = (
            r"^audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$"
        )

    elif lora_scope == "proj":
        target_regex = r"^audio_tower\.(conv_out|proj1|proj2)$"

    elif lora_scope == "tower_proj":
        target_regex = (
            r"^(audio_tower\.(conv_out|proj1|proj2)$"
            r"|audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$)"
        )

    elif lora_scope == "towerb4_proj":
        target_regex = (
            r"^(audio_tower\.(conv_out|proj1|proj2)$"
            r"|audio_tower\.layers\.(20|21|22|23)\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$)"
        )

    elif lora_scope == "llm":
        target_regex = (
            r"^model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        )

    else:  # both = tower + proj + llm
        target_regex = (
            r"^(audio_tower\.(conv_out|proj1|proj2)$"
            r"|audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$"
            r"|model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$)"
        )
    return target_regex


def dump_trainable_names(model, max_lines=300):
    cnt = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print("[trainable]", n, tuple(p.shape))
            cnt += 1
            if cnt >= max_lines:
                print(f"... truncated at {max_lines} lines")
                break


def maybe_apply_lora(model, args_cli):
    merge_lora_into_base_from = (args_cli.merge_lora_into_base_from or "").strip()

    if args_cli.use_lora == 1:
        if merge_lora_into_base_from:
            if args_cli.resume == 1 or (args_cli.resume_from or "").strip():
                raise ValueError(
                    "merge_lora_into_base_from 和 resume/resume_from 不能同时用。"
                    "如果你要恢复当前这一轮训练，请用当前轮次自己的 checkpoint 做 resume；"
                    "如果你要以旧 LoRA 为底座开启新一轮训练，就只传 merge_lora_into_base_from。"
                )

            print(f"[merge_lora] load old adapter from: {merge_lora_into_base_from}")

            old_peft_model = PeftModel.from_pretrained(
                model.thinker,
                merge_lora_into_base_from,
                is_trainable=False,
            )

            model.thinker = old_peft_model.merge_and_unload()
            print("[merge_lora] old adapter merged into base thinker")

        # 冻结全部参数（新一轮只训练新 LoRA）
        for p in model.parameters():
            p.requires_grad = False

        target_regex = get_lora_target_regex(args_cli.lora_scope)

        lora_cfg = LoraConfig(
            r=args_cli.lora_r,
            lora_alpha=args_cli.lora_alpha,
            lora_dropout=args_cli.lora_dropout,
            bias=args_cli.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_regex,
        )

        model.thinker = get_peft_model(model.thinker, lora_cfg)
        model.thinker.print_trainable_parameters()
        dump_trainable_names(model)

    return model
