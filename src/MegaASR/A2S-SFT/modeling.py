# coding=utf-8
import json
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers import GenerationConfig
from qwen_asr import Qwen3ASRModel

import action_format


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
                mix_input_features=None, mix_feature_attention_mask=None,
                action_input_ids=None, action_attention_mask=None, judge_only=False, **kwargs):
        if judge_only:
            audio = compute_audio_features(self, input_features, feature_attention_mask,
                                           mix_input_features, mix_feature_attention_mask)
            logits = action_pass_logits(self, input_ids, attention_mask, audio)
            return {"logits": logits.unsqueeze(1)}
        if action_input_ids is not None:
            # Keep both action_special_v1 decoder passes within one DDP forward.
            return two_pass_forward(
                self, input_ids, attention_mask, input_features, feature_attention_mask,
                mix_input_features, mix_feature_attention_mask,
                action_input_ids, action_attention_mask, **kwargs,
            )
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


def merge_lora_checkpoint(model, adapter_path: str):
    """Merge trained PEFT parameters into the model for adapter initialization.

    Load the production adapter, including saved audio and fusion modules,
    and merge its weights before apply_train_mode installs the action adapter.
    Optimizer state is not restored by this operation.
    """
    if not adapter_path:
        return model
    print(f"[merge_lora] initializing from production adapter {adapter_path}")
    model.thinker = PeftModel.from_pretrained(
        model.thinker, adapter_path, is_trainable=False
    ).merge_and_unload()
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
    # Checked first: action rows live on the thinker, and under PEFT their
    # names gain a "base_model.model." prefix that would otherwise match the
    # "model." llm test below and put them in the (frozen) llm group.
    if ACTION_ROWS_MODULE in name:
        return ACTION_ROWS_REGION
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
    counts = {r: 0 for r in (*REGIONS, ACTION_ROWS_REGION, "other")}
    for name, param in model.thinker.named_parameters():
        if param.requires_grad:
            counts[classify_param_region(name, fusion_module_names)] += 1
    per_region = ", ".join(f"{r}={counts[r]} ({ft_params[r]['ft_mode']})" for r in REGIONS)
    print(f"[apply_train_mode] trainable tensor counts: {per_region}"
          + (f", {ACTION_ROWS_REGION}={counts[ACTION_ROWS_REGION]}"
             if counts[ACTION_ROWS_REGION] else "")
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


# --- Protocol-token row overrides -----------------------------------------
# Train separate embedding and output rows for RESPOND, REJECT, and QUERY.
# The base embedding and LM-head matrices remain outside the optimizer.
#
# Layout after install_*:
#   thinker.action_token_rows          ActionTokenRows (owns the trainable rows)
#   thinker.model.embed_tokens         ActionEmbedding(base=<frozen nn.Embedding>)
#   thinker.lm_head                    ActionLMHead(base=<frozen nn.Linear>)
# The wrappers hold an unregistered reference to the holder, so each row
# parameter has exactly one name (".action_token_rows.") for classification,
# DDP, and the optimizer.
ACTION_ROWS_MODULE = "action_token_rows"
ACTION_ROWS_REGION = "action_rows"


class ActionTokenRows(nn.Module):
    """fp32 master rows for the protocol tokens (like PEFT's fp32 adapter
    weights -- bf16 cannot represent 1e-5-scale updates to ~1e-2 values).
    ``output_rows`` is None when the embedding and LM head are tied; the
    single ``input_rows`` then serves both, preserving the tie."""

    def __init__(self, token_ids, hidden_size, tied, info=None):
        super().__init__()
        self.register_buffer(
            "token_ids", torch.tensor(list(token_ids), dtype=torch.long), persistent=False
        )
        # Python copy for forward(): avoids a device->host sync per lookup.
        self.token_id_list = [int(x) for x in token_ids]
        n = len(token_ids)
        self.input_rows = nn.Parameter(torch.zeros(n, hidden_size, dtype=torch.float32))
        self.output_rows = (
            None if tied else nn.Parameter(torch.zeros(n, hidden_size, dtype=torch.float32))
        )
        self.tied = bool(tied)
        self.info = dict(info or {})

    def head_rows(self):
        return self.input_rows if self.tied else self.output_rows


class ActionEmbedding(nn.Module):
    def __init__(self, base: nn.Embedding, rows: ActionTokenRows):
        super().__init__()
        self.base = base
        object.__setattr__(self, "_rows", rows)

    @property
    def weight(self):
        return self.base.weight

    @weight.setter
    def weight(self, value):
        self.base.weight = value

    @property
    def num_embeddings(self):
        return self.base.num_embeddings

    @property
    def embedding_dim(self):
        return self.base.embedding_dim

    @property
    def padding_idx(self):
        return self.base.padding_idx

    def forward(self, input_ids):
        out = self.base(input_ids)
        if not getattr(self._rows, "enabled", True):
            return out
        rows = self._rows.input_rows.to(out.dtype)
        for k, token_id in enumerate(self._rows.token_id_list):
            out = torch.where(input_ids.eq(token_id).unsqueeze(-1), rows[k], out)
        return out


class ActionLMHead(nn.Module):
    def __init__(self, base: nn.Linear, rows: ActionTokenRows):
        super().__init__()
        self.base = base
        object.__setattr__(self, "_rows", rows)

    @property
    def weight(self):
        return self.base.weight

    @weight.setter
    def weight(self, value):
        self.base.weight = value

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, hidden_states):
        logits = self.base(hidden_states)
        if not getattr(self._rows, "enabled", True):
            return logits[..., :self._rows.info["lm_head_rows_original"]]
        action_logits = F.linear(hidden_states, self._rows.head_rows().to(hidden_states.dtype))
        ids = self._rows.token_ids.to(logits.device)
        return logits.index_copy(-1, ids, action_logits.to(logits.dtype))


def unwrap_thinker(model):
    thinker = model.thinker
    return thinker.get_base_model() if hasattr(thinker, "get_base_model") else thinker


def get_action_rows(model):
    return getattr(unwrap_thinker(model), ACTION_ROWS_MODULE, None)


@contextmanager
def production_transcription_mode(model):
    """Disable the judge LoRA AND row overrides, restoring production logits.

    Production adapters have already been merged into the frozen base. Do not
    reuse decoder caches across this boundary. Not safe for concurrent calls
    sharing the same model instance.
    """
    rows = get_action_rows(model)
    if rows is None or not isinstance(model.thinker, PeftModel):
        raise ValueError("Expected production-merged model with a judge adapter and rows")
    enabled = getattr(rows, "enabled", True)
    try:
        rows.enabled = False
        with model.thinker.disable_adapter():
            yield
    finally:
        rows.enabled = enabled


def _is_tied(thinker):
    return thinker.lm_head.weight is thinker.model.embed_tokens.weight


def _vocab_configs(model):
    """Every config object that records the text vocabulary size. _from_config
    may copy configs, so the wrapper's and the submodules' are checked
    independently rather than assumed shared."""
    thinker = unwrap_thinker(model)
    candidates = [
        getattr(getattr(model.config, "thinker_config", None), "text_config", None),
        getattr(thinker.config, "text_config", None),
        getattr(thinker.model, "config", None),
    ]
    seen, unique = set(), []
    for config in candidates:
        if config is not None and id(config) not in seen:
            seen.add(id(config))
            unique.append(config)
    return unique


def _set_vocab_size(model, new_rows):
    thinker = unwrap_thinker(model)
    for config in _vocab_configs(model):
        config.vocab_size = new_rows
    thinker.vocab_size = new_rows
    thinker.model.vocab_size = new_rows


def _expand_vocab_rows(model, new_rows):
    """Grow (never shrink or renumber) the thinker's embedding and LM head to
    ``new_rows`` rows. Done directly on the owning modules -- the top-level
    wrapper has no resize path that reaches the nested thinker config."""
    thinker = unwrap_thinker(model)
    emb, head = thinker.model.embed_tokens, thinker.lm_head
    old_rows = emb.weight.shape[0]
    if head.weight.shape[0] != old_rows:
        raise ValueError(f"Embedding rows ({old_rows}) and LM-head rows "
                         f"({head.weight.shape[0]}) differ; refusing to resize")
    if new_rows <= old_rows:
        return
    tied = _is_tied(thinker)
    hidden = emb.weight.shape[1]
    kwargs = {"dtype": emb.weight.dtype, "device": emb.weight.device}
    new_emb = nn.Embedding(new_rows, hidden, padding_idx=emb.padding_idx, **kwargs)
    new_head = nn.Linear(hidden, new_rows, bias=False, **kwargs)
    with torch.no_grad():
        new_emb.weight.zero_()
        new_emb.weight[:old_rows] = emb.weight
        if tied:
            new_head.weight = new_emb.weight
        else:
            new_head.weight.zero_()
            new_head.weight[:old_rows] = head.weight
    new_emb.weight.requires_grad = emb.weight.requires_grad
    new_head.weight.requires_grad = head.weight.requires_grad
    thinker.model.embed_tokens = new_emb
    thinker.lm_head = new_head
    _set_vocab_size(model, new_rows)


def _install_rows(model, rows: ActionTokenRows):
    thinker = unwrap_thinker(model)
    if hasattr(thinker, ACTION_ROWS_MODULE):
        raise ValueError("Action rows are already installed on this model")
    device = thinker.model.embed_tokens.weight.device
    rows.to(device)
    setattr(thinker, ACTION_ROWS_MODULE, rows)
    thinker.model.embed_tokens = ActionEmbedding(thinker.model.embed_tokens, rows)
    thinker.lm_head = ActionLMHead(thinker.lm_head, rows)
    return rows


def reserved_model_token_ids(model, tokenizer):
    """IDs an action token may never take: audio placeholders and the
    generation EOS tokens (offline evaluation must decode past both actions)."""
    thinker_config = getattr(model.config, "thinker_config", model.config)
    ids = []
    for attr in ("audio_token_id", "audio_start_token_id", "audio_end_token_id"):
        value = getattr(thinker_config, attr, None)
        if value is not None:
            ids.append(int(value))
    ids.extend(generation_eos_ids(tokenizer))
    return ids


def generation_eos_ids(tokenizer):
    """EOS set used for generation: tokenizer EOS plus the chat/document end
    tokens the Qwen3-ASR wrapper's generate() treats as EOS by default."""
    ids = []
    for token_id in (
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    ):
        if token_id is not None and token_id != tokenizer.unk_token_id and token_id not in ids:
            ids.append(int(token_id))
    return ids


def _vocab_dims(model):
    thinker = unwrap_thinker(model)
    return thinker.model.embed_tokens.weight.shape[0], thinker.lm_head.weight.shape[0]


def setup_action_vocabulary(model, processor):
    """Register protocol tokens and install initialized embedding/head rows.

    Expand vocabulary matrices when token IDs exceed their existing dimensions.
    Initialize protocol rows from ordinary label-token rows. Call after merging
    the production adapter and before apply_train_mode configures action LoRA.
    """
    tokenizer = processor.tokenizer
    thinker = unwrap_thinker(model)
    tokenizer_len_original = len(tokenizer)
    emb_rows_original, head_rows_original = _vocab_dims(model)
    tied = _is_tied(thinker)

    # Source rows come from the ORIGINAL tokenizer before registration.
    source_ids = {
        role: list(tokenizer(action_format.INIT_SOURCE_TEXT[role],
                             add_special_tokens=False)["input_ids"])
        for role in action_format.PROTOCOL_ROLES
    }
    for action, ids in source_ids.items():
        if not ids or max(ids) >= min(emb_rows_original, head_rows_original):
            raise ValueError(f"Invalid initialization source IDs for {action}: {ids}")

    token_ids = action_format.register_protocol_tokens(tokenizer)
    action_format.resolve_protocol_token_ids(
        tokenizer, forbidden_ids=reserved_model_token_ids(model, tokenizer)
    )
    required_rows = max(len(tokenizer), max(token_ids.values()) + 1)
    _expand_vocab_rows(model, max(required_rows, emb_rows_original))
    thinker = unwrap_thinker(model)
    emb_rows, head_rows = _vocab_dims(model)

    ordered_ids = [token_ids[role] for role in action_format.PROTOCOL_ROLES]
    info = {
        "tokenizer_len_original": tokenizer_len_original,
        "tokenizer_len_expanded": len(tokenizer),
        "embedding_rows_original": emb_rows_original,
        "embedding_rows_expanded": emb_rows,
        "lm_head_rows_original": head_rows_original,
        "lm_head_rows_expanded": head_rows,
        "hidden_size": thinker.model.embed_tokens.weight.shape[1],
        "tied_word_embeddings": tied,
        "ids_within_original_rows": all(i < emb_rows_original for i in ordered_ids),
        "initialization": {
            "policy": action_format.INIT_POLICY,
            "source_text": dict(action_format.INIT_SOURCE_TEXT),
            "source_token_ids": source_ids,
        },
    }
    rows = ActionTokenRows(ordered_ids, info["hidden_size"], tied, info)
    with torch.no_grad():
        emb_w = thinker.model.embed_tokens.weight
        head_w = thinker.lm_head.weight
        for k, action in enumerate(action_format.PROTOCOL_ROLES):
            src = torch.tensor(source_ids[action], device=emb_w.device)
            rows.input_rows[k] = emb_w.index_select(0, src).float().mean(0).cpu()
            if not tied:
                rows.output_rows[k] = head_w.index_select(0, src).float().mean(0).cpu()
    _install_rows(model, rows)
    print(f"[action_vocab] tokens={token_ids} tokenizer {tokenizer_len_original}->{len(tokenizer)} "
          f"embedding rows {emb_rows_original}->{emb_rows} lm_head rows "
          f"{head_rows_original}->{head_rows} tied={tied} init={action_format.INIT_POLICY}")
    return token_ids, rows


def install_action_vocabulary_for_reload(model, processor, metadata):
    """Reload path: rebuild row modules at the SAVED dimensions. Never
    registers tokens -- the processor must be the checkpoint's own, and its
    mapping must match the metadata exactly."""
    token_ids = action_format.verify_token_mapping(
        processor.tokenizer, metadata["protocol_tokens"],
        forbidden_ids=reserved_model_token_ids(model, processor.tokenizer),
    )
    vocab = metadata["vocabulary"]
    emb_rows, head_rows = _vocab_dims(model)
    if (emb_rows, head_rows) != (vocab["embedding_rows_original"], vocab["lm_head_rows_original"]):
        raise ValueError(
            f"Base model has embedding/head rows {(emb_rows, head_rows)}, but the checkpoint "
            f"was trained from {(vocab['embedding_rows_original'], vocab['lm_head_rows_original'])}; "
            "wrong base model?"
        )
    thinker = unwrap_thinker(model)
    if _is_tied(thinker) != bool(vocab["tied_word_embeddings"]):
        raise ValueError(f"Tying mismatch: model tied={_is_tied(thinker)}, "
                         f"checkpoint tied={vocab['tied_word_embeddings']}")
    if len(processor.tokenizer) != vocab["tokenizer_len_expanded"]:
        raise ValueError(f"Tokenizer length {len(processor.tokenizer)} != saved "
                         f"{vocab['tokenizer_len_expanded']}")
    _expand_vocab_rows(model, vocab["embedding_rows_expanded"])
    if _vocab_dims(model) != (vocab["embedding_rows_expanded"], vocab["lm_head_rows_expanded"]):
        raise ValueError(f"Could not rebuild saved vocabulary dimensions: {_vocab_dims(model)}")
    if max(token_ids.values()) >= vocab["embedding_rows_expanded"]:
        raise ValueError("Saved vocabulary dimensions do not cover the action token IDs")
    ordered_ids = [token_ids[role] for role in action_format.PROTOCOL_ROLES]
    rows = ActionTokenRows(ordered_ids, vocab["hidden_size"], vocab["tied_word_embeddings"], vocab)
    _install_rows(model, rows)
    return token_ids, rows


def enable_action_row_training(model):
    """Enable protocol-row gradients after PEFT configures adapter parameters."""
    rows = get_action_rows(model)
    if rows is None:
        raise ValueError("Action rows are not installed")
    params = [p for p in rows.parameters()]
    for param in params:
        param.requires_grad = True
    lora = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and ("lora_A" in n or "lora_B" in n))
    action = sum(p.numel() for p in params)
    other = sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and ACTION_ROWS_MODULE not in n
                and "lora_A" not in n and "lora_B" not in n)
    print(f"[action_vocab] trainable params: decoder/region LoRA={lora:,} "
          f"action rows={action:,} ({len(params)} tensors) other={other:,}")
    return {"lora": lora, "action_rows": action, "other": other}


def save_action_rows(model, output_dir):
    rows = get_action_rows(model)
    if rows is None:
        raise ValueError("Cannot save action rows: none are installed")
    tensors = {
        "token_ids": rows.token_ids.detach().cpu().contiguous(),
        "input_rows": rows.input_rows.detach().float().cpu().contiguous(),
    }
    if not rows.tied:
        tensors["output_rows"] = rows.output_rows.detach().float().cpu().contiguous()
    metadata = {
        "action_format": action_format.ACTION_SPECIAL_V1,
        "metadata_version": str(action_format.METADATA_VERSION),
        "tied_word_embeddings": str(rows.tied).lower(),
        "hidden_size": str(rows.input_rows.shape[1]),
        "embedding_rows_original": str(rows.info["embedding_rows_original"]),
        "embedding_rows_expanded": str(rows.info["embedding_rows_expanded"]),
        "lm_head_rows_original": str(rows.info["lm_head_rows_original"]),
        "lm_head_rows_expanded": str(rows.info["lm_head_rows_expanded"]),
        "roles": ",".join(action_format.PROTOCOL_ROLES),
    }
    path = os.path.join(output_dir, action_format.ROWS_FILENAME)
    safe_save_file(tensors, path, metadata=metadata)
    return path


def load_action_rows(model, checkpoint_dir):
    """Load saved row values into the installed holder; every check failure
    raises rather than leaving freshly-initialized rows in place."""
    rows = get_action_rows(model)
    if rows is None:
        raise ValueError("Install action rows before loading them")
    path = os.path.join(checkpoint_dir, action_format.ROWS_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing action-row artifact: {path}")
    with safe_open(path, framework="pt") as handle:
        file_meta = handle.metadata() or {}
    expected_meta = {
        "action_format": action_format.ACTION_SPECIAL_V1,
        "tied_word_embeddings": str(rows.tied).lower(),
        "embedding_rows_expanded": str(rows.info["embedding_rows_expanded"]),
        "lm_head_rows_expanded": str(rows.info["lm_head_rows_expanded"]),
        "roles": ",".join(action_format.PROTOCOL_ROLES),
    }
    for key, value in expected_meta.items():
        if file_meta.get(key) != value:
            raise ValueError(f"{path}: metadata {key}={file_meta.get(key)!r}, expected {value!r}")
    tensors = safe_load_file(path)
    expected_keys = {"token_ids", "input_rows"} | (set() if rows.tied else {"output_rows"})
    if set(tensors) != expected_keys:
        raise ValueError(f"{path}: tensors {sorted(tensors)}, expected {sorted(expected_keys)}")
    if not torch.equal(tensors["token_ids"].cpu(), rows.token_ids.cpu()):
        raise ValueError(f"{path}: token IDs {tensors['token_ids'].tolist()} do not match "
                         f"the tokenizer's {rows.token_ids.tolist()}")
    with torch.no_grad():
        for name in expected_keys - {"token_ids"}:
            param = getattr(rows, name)
            if tuple(tensors[name].shape) != tuple(param.shape):
                raise ValueError(f"{path}: {name} shape {tuple(tensors[name].shape)} != "
                                 f"{tuple(param.shape)}")
            param.copy_(tensors[name].to(param.device, param.dtype))
            if not torch.equal(param.detach().cpu(), tensors[name].to(param.dtype)):
                raise AssertionError(f"{name} did not load exactly from {path}")
    return path


def load_adapter_weights_strict(peft_thinker, checkpoint_dir):
    """Load PEFT adapter weights and verify every saved tensor after key mapping.

    Saved keys omit the adapter name present in live parameter names, such as
    '.lora_A.weight' versus '.lora_A.default.weight'.
    """
    path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    saved = safe_load_file(path)
    result = set_peft_model_state_dict(peft_thinker, saved)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise ValueError(f"Adapter checkpoint {path} has keys the model lacks: {unexpected[:10]}")
    verify_adapter_matches(peft_thinker, saved, path)
    return path


def verify_adapter_matches(peft_thinker, saved, source="adapter"):
    live = get_peft_model_state_dict(peft_thinker)
    if set(live) != set(saved):
        missing = sorted(set(live) - set(saved))[:10]
        extra = sorted(set(saved) - set(live))[:10]
        raise ValueError(f"{source}: adapter key mismatch; not in file={missing}, not in model={extra}")
    for key, value in saved.items():
        if not torch.equal(live[key].detach().cpu(), value.to(live[key].dtype)):
            raise AssertionError(f"{source}: adapter tensor {key} did not load exactly")


def load_processor(path):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(path, fix_mistral_regex=True)


def load_action_special_adapter(checkpoint_dir, metadata, use_fusion, fusion_type,
                                base_model_path, merged_from_lora_path):
    """Reconstruct a special-token action model or transcript-conditioned judge.

    Merge production weights at base vocabulary dimensions, load the checkpoint
    processor, and restore protocol rows and decoder LoRA at saved dimensions.
    Validate the token mapping and loaded weights before returning the model.
    """
    model, _base_processor, use_bf16 = load_qwen3_asr(
        base_model_path, use_fusion=use_fusion, fusion_type=fusion_type
    )
    if merged_from_lora_path:
        print(f"[load] merging production adapter {merged_from_lora_path} at original dims")
        model = merge_lora_checkpoint(model, merged_from_lora_path)
    processor = load_processor(checkpoint_dir)
    install_action_vocabulary_for_reload(model, processor, metadata)
    load_action_rows(model, checkpoint_dir)
    model.thinker = PeftModel.from_pretrained(model.thinker, checkpoint_dir, is_trainable=False)
    verify_adapter_matches(
        model.thinker,
        safe_load_file(os.path.join(checkpoint_dir, "adapter_model.safetensors")),
        checkpoint_dir,
    )
    rows = get_action_rows(model)
    thinker = unwrap_thinker(model)
    if not isinstance(thinker.model.embed_tokens, ActionEmbedding) or not isinstance(
        thinker.lm_head, ActionLMHead
    ):
        raise AssertionError("Action row wrappers were not installed on the reloaded model")
    print(f"[load] action_special_v1 checkpoint: tokens="
          f"{dict(zip(action_format.PROTOCOL_ROLES, rows.token_ids.tolist()))} "
          f"rows={_vocab_dims(model)} tied={rows.tied}")
    return model, processor, use_bf16


def materialize_action_rows(model):
    """Full-model export: write the trained rows into ordinary expanded
    embedding/head matrices and remove the wrappers. Call after LoRA has
    been merged (merge_and_unload)."""
    thinker = unwrap_thinker(model)
    rows = getattr(thinker, ACTION_ROWS_MODULE, None)
    if rows is None:
        raise ValueError("No action rows to materialize")
    emb = thinker.model.embed_tokens.base
    head = thinker.lm_head.base
    ids = rows.token_ids.to(emb.weight.device)
    with torch.no_grad():
        emb.weight[ids] = rows.input_rows.to(emb.weight.device, emb.weight.dtype)
        if not rows.tied:
            head.weight[ids] = rows.output_rows.to(head.weight.device, head.weight.dtype)
    thinker.model.embed_tokens = emb
    thinker.lm_head = head
    delattr(thinker, ACTION_ROWS_MODULE)
    if rows.tied and head.weight is not emb.weight:
        raise AssertionError("Tied embedding/head lost their tie during materialization")
    return model


# --- action_special_v1 decoder passes -----------------------------------------
# Transcript pass:  [prompt] -> transcript<EOS>           (plain ASR)
# Action pass:      [prompt] <|action_query|> -> action    (logits at the query)
# The audio tower runs once; its features are scattered into both passes'
# placeholder positions. Positions come from the attention mask's cumulative
# sum, so a left-padded action batch gives the prompt the same positions as
# in the transcript pass.

def compute_audio_features(model, input_features, feature_attention_mask,
                           mix_input_features=None, mix_feature_attention_mask=None):
    base = unwrap_thinker(model)
    kwargs = {"feature_attention_mask": feature_attention_mask}
    if mix_input_features is not None:
        kwargs["mix_input_features"] = mix_input_features
        kwargs["mix_feature_attention_mask"] = mix_feature_attention_mask
    return base.get_audio_features(input_features, **kwargs)


def _embed_with_audio(base, input_ids, audio_features):
    embeds = base.get_input_embeddings()(input_ids)
    audio = audio_features.to(embeds.device, embeds.dtype)
    mask = base.get_placeholder_mask(input_ids, inputs_embeds=embeds)
    if int(mask[..., 0].sum()) != audio.shape[0]:
        raise ValueError(f"{int(mask[..., 0].sum())} audio placeholders but "
                         f"{audio.shape[0]} audio feature rows")
    # MPS masked_scatter can flatten its source tensor's shape in place.
    # Pass a separate, already-flat view so the shared [frames, hidden]
    # features retain their shape for the second decoder pass (and gradients
    # still flow through this view when the acoustic tower is trainable).
    return embeds.masked_scatter(mask, audio.reshape(-1))


def action_pass_logits(model, action_input_ids, action_attention_mask, audio_features):
    """Full-vocabulary logits at the LAST position of a left-padded
    [prompt] <|action_query|> batch. Only the decoder runs; LoRA layers are
    injected in place, so they apply here too."""
    base = unwrap_thinker(model)
    if not bool(action_attention_mask[:, -1].all()):
        raise ValueError("Action pass must be left-padded so the query is the last column")
    rows = get_action_rows(model)
    if rows is not None:
        query_id = rows.token_id_list[action_format.PROTOCOL_ROLES.index(action_format.QUERY)]
        if not bool(action_input_ids[:, -1].eq(query_id).all()):
            raise ValueError("Every action-pass row must end with <|action_query|>")
    embeds = _embed_with_audio(base, action_input_ids, audio_features)
    position_ids, _ = base.get_rope_index(action_attention_mask)
    hidden = base.model(
        attention_mask=action_attention_mask, position_ids=position_ids,
        inputs_embeds=embeds, use_cache=False,
    )[0]
    return base.lm_head(hidden[:, -1, :])


def two_pass_forward(model, input_ids, attention_mask, input_features, feature_attention_mask,
                     mix_input_features, mix_feature_attention_mask,
                     action_input_ids, action_attention_mask, **kwargs):
    base = unwrap_thinker(model)
    audio = compute_audio_features(model, input_features, feature_attention_mask,
                                   mix_input_features, mix_feature_attention_mask)
    audio_id = base.config.audio_token_id
    if not torch.equal(input_ids.eq(audio_id).sum(1), action_input_ids.eq(audio_id).sum(1)):
        raise ValueError("Transcript and action passes have different audio placeholder counts")
    outputs = model.thinker(
        inputs_embeds=_embed_with_audio(base, input_ids, audio),
        attention_mask=attention_mask, use_cache=False, **kwargs,
    )
    outputs["action_logits"] = action_pass_logits(
        model, action_input_ids, action_attention_mask, audio
    )
    return outputs


@torch.no_grad()
def action_logits_fresh(model, inputs, query_id, mix_inputs=None):
    """Inference: one action pass on [prompt] <|action_query|> from scratch
    (used for early REJECT, before any transcript exists)."""
    audio = compute_audio_features(
        model, inputs["input_features"], inputs["feature_attention_mask"],
        None if mix_inputs is None else mix_inputs["input_features"],
        None if mix_inputs is None else mix_inputs["feature_attention_mask"],
    )
    ids = inputs["input_ids"]
    query = torch.full((ids.shape[0], 1), query_id, dtype=ids.dtype, device=ids.device)
    action_ids = torch.cat([ids, query], dim=1)
    action_mask = torch.cat([inputs["attention_mask"], torch.ones_like(query)], dim=1)
    return action_pass_logits(model, action_ids, action_mask, audio).float()


@torch.no_grad()
def action_logits_from_cache(model, past_key_values, prompt_length, query_id):
    """Inference after transcript generation (batch of one, no padding):
    crop the generation cache back to the prompt, so the generated transcript
    is invisible, and run the single query token on top of it."""
    if past_key_values is None:
        raise ValueError("generate() returned no cache; cannot reuse the prompt computation")
    past_key_values.crop(prompt_length)
    if past_key_values.get_seq_length() != prompt_length:
        raise ValueError("Generation cache is shorter than the prompt")
    device = unwrap_thinker(model).lm_head.weight.device
    query = torch.tensor([[query_id]], device=device)
    out = model.thinker(
        input_ids=query,
        attention_mask=torch.ones(1, prompt_length + 1, dtype=torch.long, device=device),
        past_key_values=past_key_values,
        cache_position=torch.tensor([prompt_length], device=device),
        use_cache=True,
    )
    return out.logits[0, -1].float()
