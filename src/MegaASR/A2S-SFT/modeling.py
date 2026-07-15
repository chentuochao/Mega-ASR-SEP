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


_FUSION_PARAM_NAME = {"late_gate": "fusion_gate", "early_conv": "conv2d1_mix"}


def _check_fusion_type_matches_checkpoint(fusion_type, unexpected_keys):
    """from_pretrained silently drops unexpected/mismatched keys (a warning at
    most, never an error) -- so requesting the WRONG fusion_type for a
    checkpoint actually trained with the other one "loads successfully" while
    discarding that checkpoint's real trained fusion weights and substituting
    a fresh, untrained (no-op) parameter in their place. The generic "newly
    initialized" warning HF prints for the missing param looks identical to
    what a brand-new training run prints, so this failure mode is otherwise
    silent. Catch it here instead of silently evaluating the wrong thing."""
    other_type = "early_conv" if fusion_type == "late_gate" else "late_gate"
    other_param = _FUSION_PARAM_NAME[other_type]
    hit = [k for k in unexpected_keys if other_param in k]
    if hit:
        this_param = _FUSION_PARAM_NAME[fusion_type]
        raise ValueError(
            f"fusion_type={fusion_type!r} was requested, but this checkpoint "
            f"contains trained {other_param!r} weights ({hit}) and no "
            f"{this_param!r} -- it was almost certainly trained with "
            f"fusion_type={other_type!r} instead. Loading it as {fusion_type!r} "
            f"would silently discard those trained weights and evaluate a "
            f"fresh, untrained {this_param!r} (a true no-op) instead of this "
            f"checkpoint's actual fusion behavior."
        )


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
        model, loading_info = Qwen3ASRForConditionalGenerationSep.from_pretrained(
            model_path, config=config, dtype=dtype, device_map=None,
            output_loading_info=True,
        )
        _check_fusion_type_matches_checkpoint(fusion_type, loading_info["unexpected_keys"])
        # Only reset on a FIRST-TIME init from a pristine checkpoint that never
        # had this fusion param (missing_keys) -- from_pretrained re-inits a
        # missing param via HF's default _init_weights (e.g. fusion_gate bias
        # -> 0 => gate = 0.5), so override that with the deliberate near-no-op
        # init a fresh training run should start from. An ALREADY-TRAINED
        # checkpoint loads its real fusion_gate/conv2d1_mix values here
        # instead (not in missing_keys) -- resetting unconditionally would
        # silently wipe those trained values back to init on every eval/reload.
        fusion_param = _FUSION_PARAM_NAME[fusion_type]
        if any(fusion_param in k for k in loading_info["missing_keys"]):
            model.thinker.audio_tower.reset_fusion_params()
        processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    else:
        wrapper = Qwen3ASRModel.from_pretrained(
            model_path, dtype=dtype, device_map=None,
        )
        model, processor = wrapper.model, wrapper.processor

    patch_outer_forward(model)
    # HF's Trainer decides whether a model normalizes loss across gradient-
    # accumulation micro-batches (via num_items_in_batch) by checking for a
    # bare **kwargs in forward() -- not whether num_items_in_batch is actually
    # used. patch_outer_forward's wrapper (and the thinker's forward) both end
    # in **kwargs, so Trainer wrongly concludes this model self-normalizes and
    # skips its own `loss / gradient_accumulation_steps` division, while the
    # model silently drops num_items_in_batch without ever using it. Net
    # effect: the loss actually backpropagated (and logged) is
    # gradient_accumulation_steps times too large. Per Trainer.compute_loss's
    # own docstring, force the safe fallback explicitly rather than relying on
    # the **kwargs heuristic.
    model.accepts_loss_kwargs = False
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
