# coding=utf-8
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import GenerationConfig
from qwen_asr import Qwen3ASRModel


LORA_TARGETS = {
    "encoder": r"^audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$",
    "aligner": r"^audio_tower\.(conv_out|proj1|proj2)$",
    "encoder_aligner": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$)"
    ),
    "encoder_b4_aligner": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.(20|21|22|23)\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$)"
    ),
    "llm": r"^model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    "all": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$"
        r"|model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$)"
    ),
}


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker"):
        raise RuntimeError("Qwen3-ASR wrapper has no `thinker` module.")

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None,
                mix_input_features=None, mix_feature_attention_mask=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            mix_input_features=mix_input_features,
            mix_feature_attention_mask=mix_feature_attention_mask,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def load_qwen3_asr(model_path: str, use_fusion: bool = False, fusion_type: str = "late_gate"):
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if use_fusion:
        # Mixture-conditioned fusion path: load the vendored fusion-capable model
        # directly. The new fusion params (fusion_gate or conv2d1_mix, depending
        # on fusion_type) are missing from the pretrained checkpoint and are
        # initialised to a (near-)no-op, so the model starts numerically close to
        # the single-stream baseline.
        from transformers import AutoProcessor

        from models.configuration_qwen3_asr import Qwen3ASRConfig
        from models.modeling_qwen3_asr_sep import Qwen3ASRForConditionalGenerationSep

        config = Qwen3ASRConfig.from_pretrained(model_path)
        config.thinker_config.audio_config.use_fusion = True
        config.thinker_config.audio_config.fusion_type = fusion_type
        model = Qwen3ASRForConditionalGenerationSep.from_pretrained(
            model_path, config=config, dtype=dtype, device_map=None,
        )
        # from_pretrained re-inits the (missing) fusion params via HF's default
        # _init_weights (e.g. fusion_gate bias -> 0 => gate = 0.5). Restore the
        # (near-)no-op init so training starts close to the single-stream baseline.
        model.thinker.audio_tower.reset_fusion_params()
        processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    else:
        wrapper = Qwen3ASRModel.from_pretrained(
            model_path, dtype=dtype, device_map=None,
        )
        model, processor = wrapper.model, wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    return model, processor, use_bf16


def apply_lora(model, args):
    if not args.use_lora:
        if getattr(args, "freeze_llm", False):
            llm_prefixes = ("model.", "lm_head.")
            for name, param in model.thinker.named_parameters():
                param.requires_grad = not name.startswith(llm_prefixes)

            trainable = sum(p.numel() for p in model.thinker.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.thinker.parameters())
            print(f"[freeze_llm] trainable params: {trainable:,} / {total:,} "
                  f"({100 * trainable / total:.2f}%) -- LLM frozen, "
                  f"audio_tower full fine-tune")
        return model

    old_lora = args.merge_lora_into_base_from.strip()
    if old_lora:
        if args.resume or args.resume_from.strip():
            raise ValueError("Do not use --merge_lora_into_base_from with --resume.")
        print(f"[merge_lora] {old_lora}")
        model.thinker = PeftModel.from_pretrained(
            model.thinker, old_lora, is_trainable=False
        ).merge_and_unload()

    for param in model.parameters():
        param.requires_grad = False

    # The fusion params are small full-rank heads, not LoRA targets. Register
    # them via `modules_to_save` so PEFT keeps them trainable AND serialises
    # them in the adapter checkpoint (a plain trainable module would otherwise
    # be silently dropped by the adapter-only save path).
    if getattr(args, "use_fusion", False):
        fusion_type = getattr(args, "fusion_type", "late_gate")
        modules_to_save = ["fusion_gate"] if fusion_type == "late_gate" else ["conv2d1_mix"]
    else:
        modules_to_save = None

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGETS[args.lora_scope],
        modules_to_save=modules_to_save,
    )
    model.thinker = get_peft_model(model.thinker, lora_config)
    model.thinker.print_trainable_parameters()
    return model
