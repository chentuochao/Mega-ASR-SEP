# coding=utf-8
import json

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


# Top-level attribute name(s) each fusion type adds to the audio tower (see
# models/modeling_qwen3_asr_sep.py's FUSION_ENCODER_CLASSES) -- used both to
# detect a fusion_type/checkpoint mismatch and to tell PEFT which modules to
# keep trainable (`modules_to_save`). Keep in sync when adding a new fusion
# type there.
_FUSION_MODULE_NAMES = {
    "late_gate": ["fusion_gate"],
    "early_conv": ["conv2d1_mix"],
    "fddt": ["cond_proj", "film_proj", "layer_gates"],
    "cross_attn": ["cross_attn"],
}


def _check_fusion_type_matches_checkpoint(fusion_type, unexpected_keys):
    """from_pretrained silently drops unexpected/mismatched keys (a warning at
    most, never an error) -- so requesting the WRONG fusion_type for a
    checkpoint actually trained with a different one "loads successfully"
    while discarding that checkpoint's real trained fusion weights and
    substituting a fresh, untrained (no-op) parameter in their place. The
    generic "newly initialized" warning HF prints for the missing param looks
    identical to what a brand-new training run prints, so this failure mode
    is otherwise silent. Catch it here instead of silently evaluating the
    wrong thing."""
    for other_type, other_names in _FUSION_MODULE_NAMES.items():
        if other_type == fusion_type:
            continue
        hit = [k for k in unexpected_keys if any(name in k for name in other_names)]
        if hit:
            this_names = _FUSION_MODULE_NAMES[fusion_type]
            raise ValueError(
                f"fusion_type={fusion_type!r} was requested, but this checkpoint "
                f"contains trained {other_type!r} weights ({hit}) and none of "
                f"{this_names} -- it was almost certainly trained with "
                f"fusion_type={other_type!r} instead. Loading it as {fusion_type!r} "
                f"would silently discard those trained weights and evaluate a "
                f"fresh, untrained set of {fusion_type!r} params (a true no-op) "
                f"instead of this checkpoint's actual fusion behavior."
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
        fusion_names = _FUSION_MODULE_NAMES[fusion_type]
        if any(name in k for k in loading_info["missing_keys"] for name in fusion_names):
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
        modules_to_save = list(_FUSION_MODULE_NAMES[fusion_type])
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


# --- Per-region training mode (encoder / fusion / aligner / llm), each ------
# independently freeze / lora / full -- see arguments.py's --ft_params and
# scripts/configs/example_ft_params.yaml. This supersedes apply_lora() above
# (kept as-is for any recipe that still passes the older --use_lora/
# --freeze_llm/--lora_scope flags instead of --ft_params) with a strictly
# more expressive mechanism: apply_lora can only LoRA-adapt-or-freeze "the
# rest of the model" as one unit plus always-full-FT the fusion params,
# whereas apply_train_mode lets every region pick its own mode independently
# (e.g. encoder=lora, fusion=full, aligner=full, llm=freeze in one run).
REGIONS = ("encoder", "fusion", "aligner", "llm")
_VALID_FT_MODES = ("freeze", "lora", "full")

# LoRA target_modules regex per region -- identical to LORA_TARGETS above,
# just split so apply_train_mode can union an arbitrary subset of them.
_REGION_LORA_TARGET_PATTERNS = {
    "encoder": LORA_TARGETS["encoder"],
    "aligner": LORA_TARGETS["aligner"],
    "llm": LORA_TARGETS["llm"],
}


def classify_param_region(name: str, fusion_module_names=()) -> str:
    """Classify a thinker-level parameter name into one of the four
    trainable regions (encoder/fusion/aligner/llm), or "other" for anything
    unrecognized. fusion and aligner are checked before the generic
    "audio_tower." catch-all so their params (which also live under
    audio_tower) aren't misclassified as encoder. Every check is a plain
    substring test (never startswith/anchored), so this works unchanged on
    a bare thinker's param names, on the top-level wrapper's "thinker."-
    prefixed names, and on PEFT-wrapped names (".../lora_A.default...",
    ".../modules_to_save.default...", "base_model.model...." prefix) alike --
    whatever wrapping is added, the substrings this checks for are still in
    there somewhere. Shared by apply_train_mode (this module, for freeze/
    lora/full decisions) and MegaASRTrainer's optimizer param-group builder
    (trainer.py) for per-region learning rates, so the two can never
    classify the same parameter into different regions."""
    if any(sig in name for sig in fusion_module_names):
        return "fusion"
    if any(f"audio_tower.{sig}" in name for sig in ("conv_out", "proj1", "proj2")):
        return "aligner"
    if "audio_tower." in name:
        return "encoder"
    if "model." in name or "lm_head." in name:
        return "llm"
    return "other"


def parse_ft_params(raw: str) -> dict:
    """Parse --ft_params's JSON string into {region: {"ft_mode": ..., "lr":
    ..., ...}} and validate it. Raises ValueError with a specific message on
    anything wrong rather than letting a malformed config silently train (or
    silently NOT train) the wrong thing."""
    try:
        ft_params = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--ft_params is not valid JSON: {e}\nGot: {raw!r}") from e

    missing = [r for r in REGIONS if r not in ft_params]
    if missing:
        raise ValueError(f"--ft_params is missing region(s) {missing}; all of {REGIONS} must be specified")

    for region in REGIONS:
        region_cfg = ft_params[region]
        mode = region_cfg.get("ft_mode")
        if mode not in _VALID_FT_MODES:
            raise ValueError(
                f"--ft_params[{region!r}].ft_mode must be one of {_VALID_FT_MODES}, got {mode!r}"
            )
        if "lr" not in region_cfg:
            raise ValueError(f"--ft_params[{region!r}] is missing 'lr'")
        # Coerce numeric fields explicitly rather than trusting the JSON's
        # own type: YAML's bare-exponent floats (e.g. "lr: 1e-5", no decimal
        # point) don't match PyYAML's float regex and come through as the
        # STRING "1e-5", not a float -- load_config.py JSON-serializes this
        # nested block value-for-value (unlike the flat scalar keys, which
        # get str()'d and re-typed by argparse), so that string would
        # otherwise reach torch.optim.AdamW's param group as-is and break.
        for key, caster in (("lr", float), ("lora_r", int), ("lora_alpha", int)):
            if key in region_cfg:
                try:
                    region_cfg[key] = caster(region_cfg[key])
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"--ft_params[{region!r}][{key!r}] = {region_cfg[key]!r} is not a valid "
                        f"{caster.__name__} (tip: YAML needs a decimal point for exponential "
                        f"notation to parse as a number, e.g. 1.0e-5 not 1e-5)"
                    ) from e

    if ft_params["fusion"]["ft_mode"] == "lora":
        raise ValueError(
            "--ft_params['fusion'].ft_mode='lora' doesn't make sense: fusion params "
            "(fusion_gate/conv2d1_mix/cond_proj+film_proj+layer_gates/cross_attn, "
            "depending on fusion_type) have no pretrained base weight to low-rank-"
            "adapt -- they're created fresh, untrained, at init. Use 'freeze' or 'full'."
        )
    return ft_params


def _region_full_ft_module_names(region: str, audio_tower) -> list:
    """Concrete nn.Module names for full-FT'ing `region` via PEFT's
    modules_to_save -- only used when some OTHER region is set to 'lora' (so
    the whole thinker gets PEFT-wrapped and plain requires_grad toggling
    isn't available). Every entry here is a real nn.Module, never a bare
    ModuleList/ModuleDict (which PEFT's modules_to_save explicitly refuses to
    wrap -- see check_module in peft/utils/other.py, and _FDDTLayerGates in
    modeling_qwen3_asr_sep.py for the same constraint hit and fixed there):
    `audio_tower.layers` itself is a ModuleList, so encoder is targeted one
    layer-module at a time rather than as a single entry; `thinker.model` is
    a proper composite Module (its own internal `.layers` ModuleList is
    irrelevant to PEFT, which only inspects the top-level wrapped attribute),
    so llm can be targeted as one entry."""
    if region == "encoder":
        return [f"audio_tower.layers.{i}" for i in range(len(audio_tower.layers))] + [
            "audio_tower.conv2d1", "audio_tower.conv2d2", "audio_tower.conv2d3", "audio_tower.ln_post",
        ]
    if region == "aligner":
        return ["audio_tower.conv_out", "audio_tower.proj1", "audio_tower.proj2"]
    if region == "llm":
        return ["model", "lm_head"]
    raise AssertionError(f"fusion (and any other region) must be handled by the caller, got {region!r}")


def _print_train_mode_summary(model, ft_params: dict, fusion_module_names) -> None:
    counts = {r: 0 for r in (*REGIONS, "other")}
    for name, param in model.thinker.named_parameters():
        if param.requires_grad:
            counts[classify_param_region(name, fusion_module_names)] += 1
    per_region = ", ".join(f"{r}={counts[r]} ({ft_params[r]['ft_mode']})" for r in REGIONS)
    print(f"[apply_train_mode] trainable tensor counts: {per_region}"
          + (f", other={counts['other']}" if counts["other"] else ""))


def apply_train_mode(model, ft_params: dict, fusion_type: str, use_fusion: bool,
                      lora_dropout: float = 0.05, lora_bias: str = "none"):
    """Independently freeze / full-FT / LoRA-adapt each of the four regions
    (encoder/fusion/aligner/llm). `ft_params` is parse_ft_params()'s output:
    {region: {"ft_mode": "freeze"|"lora"|"full", "lr": float, "lora_r"?: int,
    "lora_alpha"?: int}} for every region in REGIONS. Per-region lora_r/
    lora_alpha map to PEFT's rank_pattern/alpha_pattern (per-module-pattern
    overrides on top of one base LoraConfig) -- lora_dropout/lora_bias are
    necessarily global across all LoRA regions (PEFT has no per-pattern
    dropout/bias override), so those two stay plain function args."""
    fusion_module_names = list(_FUSION_MODULE_NAMES[fusion_type]) if use_fusion else []
    if ft_params["fusion"]["ft_mode"] != "freeze" and not use_fusion:
        print(f"[apply_train_mode] WARNING: fusion.ft_mode={ft_params['fusion']['ft_mode']!r} "
              f"but use_fusion=False -- there are no fusion params to train, this setting has no effect")

    lora_regions = [r for r in REGIONS if ft_params[r]["ft_mode"] == "lora"]
    full_regions = [r for r in REGIONS if ft_params[r]["ft_mode"] == "full"]

    if not lora_regions:
        # No region wants LoRA -> plain requires_grad toggling, no PEFT
        # wrapper at all: numerically and structurally identical to a
        # from-scratch full fine-tune (same checkpoint format verified by
        # tests/test_fusion_save_load.py's full-FT round trip).
        for name, param in model.thinker.named_parameters():
            param.requires_grad = classify_param_region(name, fusion_module_names) in full_regions
        _print_train_mode_summary(model, ft_params, fusion_module_names)
        return model

    for param in model.parameters():
        param.requires_grad = False

    target_modules = "|".join(f"(?:{_REGION_LORA_TARGET_PATTERNS[r]})" for r in lora_regions)

    modules_to_save = []
    for region in full_regions:
        if region == "fusion":
            modules_to_save.extend(fusion_module_names)
        else:
            modules_to_save.extend(_region_full_ft_module_names(region, model.thinker.audio_tower))

    base_region = lora_regions[0]
    base_r = ft_params[base_region].get("lora_r", 16)
    base_alpha = ft_params[base_region].get("lora_alpha", 32)
    rank_pattern, alpha_pattern = {}, {}
    for region in lora_regions[1:]:
        r = ft_params[region].get("lora_r", base_r)
        alpha = ft_params[region].get("lora_alpha", base_alpha)
        if r != base_r:
            rank_pattern[_REGION_LORA_TARGET_PATTERNS[region]] = r
        if alpha != base_alpha:
            alpha_pattern[_REGION_LORA_TARGET_PATTERNS[region]] = alpha

    lora_config = LoraConfig(
        r=base_r,
        lora_alpha=base_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        modules_to_save=modules_to_save or None,
    )
    model.thinker = get_peft_model(model.thinker, lora_config)
    model.thinker.print_trainable_parameters()
    _print_train_mode_summary(model, ft_params, fusion_module_names)
    return model
