# coding=utf-8
"""Evaluate a trained Mega-ASR-SEP checkpoint's WER/CER on a validation JSONL.

Action checkpoints select their evaluator from metadata. Judge checkpoints
default to pipeline mode with unconstrained action decoding. Use --mode cached
for prepared saved candidates, or --task action for metadata-free action_first.
See docs/action-model.md for the shared CLI and output measures.

Loads the checkpoint via the SAME `load_qwen3_asr` used during training --
--use_fusion/--fusion_type/--collator/--wn_seed/--language must match how the
checkpoint was actually trained (e.g. a late-fusion checkpoint needs
--use_fusion 1 --fusion_type late_gate, or generation will silently run the
single-stream fallback path instead; a checkpoint trained with the default
--language English forces a "language English<asr_text>" prefix, and
evaluating it with language-ID unforced measures a prompt distribution it
never saw). These are auto-filled from <output_dir>/run_config.json if
present (finetune.py writes one there at the start of every training run --
checkpoints are always direct children of output_dir) and left unset on the
command line; pass any of them explicitly to override the auto-detected
value.

Audio is preprocessed via the SAME `dataloader.read_audio` the training
collator used, not `qwen_asr`'s own audio loader (which additionally
peak-normalizes) -- using different preprocessing at eval time than at
training time would measure a slightly different input distribution than
what the model actually learned on.

Usage (run from the A2S-SFT directory):

    cd src/MegaASR/A2S-SFT
    CUDA_VISIBLE_DEVICES=0 python tests/eval_checkpoint.py \
        --checkpoint_dir /mmfs1/gscratch/intelligentsystems/daniel/results/debug_none/checkpoint-8 \
        --val_file /scr/gigaspeech_qwen_asr_dataset/val.jsonl

For a late/early fusion checkpoint:
    ... --use_fusion 1 --fusion_type late_gate

For a checkpoint trained with the white_noise_test diagnostic collator (see
dataloader.py's Qwen3ASRCollator_WhiteNoise_Test): pass --collator
white_noise_test so eval audio is built the SAME way training built it --
BOTH streams synthesized from row["libritts_path"] via the identical
disjoint-noise-masking code, not read from audio_sep/audio_mix. Requires
--use_fusion 1 (the checkpoint must actually consume mix_input_features):
    ... --use_fusion 1 --fusion_type late_gate --collator white_noise_test --wn_seed 0

For a checkpoint where any modeling.py's apply_train_mode region (encoder/
fusion/aligner/llm) was ft_mode: lora (see arguments.py's --ft_params) --
equally, any checkpoint from the older --use_lora 1 path (finetune_fusion.sh)
-- the checkpoint directory is a PEFT ADAPTER (adapter_model.safetensors +
adapter_config.json), not a full model: trainer.py's save_model deletes any
full-model weight file from an adapter checkpoint's directory. Loading it
needs the ORIGINAL base checkpoint the adapter was trained from, applied via
PeftModel.from_pretrained -- see _load_model_for_eval below.
--base_model_path is auto-filled from run_config.json's "model_path" (which
finetune.py writes for every run, adapter or not) if left unset; pass it
explicitly only for a checkpoint saved before that field existed there (it's
also in that checkpoint's base_model.txt, written by trainer.py, as a
plain-text fallback for exactly this case). A plain full-FT checkpoint
(nothing was ft_mode: lora) needs none of this -- it's a full model, loaded
exactly as before.
"""
import argparse
import json
import os
import re
import statistics
import string
import subprocess
import sys
from datetime import datetime, timezone

import jiwer
import torch
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import action_format  # noqa: E402
import modeling  # noqa: E402
from dataloader import Qwen3ASRCollator_WhiteNoise_Test, read_audio  # noqa: E402
from modeling import load_qwen3_asr  # noqa: E402

# PEFT's two possible adapter-weights filenames (safetensors preferred, .bin
# fallback for an older/non-safetensors save) -- their presence in
# --checkpoint_dir is how we tell an adapter checkpoint apart from a full
# model, since trainer.py's save_model deletes any full-model weight file
# from an adapter checkpoint's directory (and vice versa).
_ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")

try:
    from num2words import num2words as _num2words
    _NUM2WORDS = True
except ImportError:
    _NUM2WORDS = False


def normalize_string(text: str) -> str:
    """Mirror hearvana-dataset-pipeline/evaluators/eval_model_performance.py's
    normalize_string, so WER/CER here are comparable to those reported there."""
    text = text.lower()
    text = text.replace('%', ' percent').replace('&', ' and').replace('@', ' at')
    text = text.replace('-', ' ')
    text = text.replace('per cent', 'percent')
    if _NUM2WORDS:
        text = re.sub(
            r'\b\d+\b',
            lambda m: _num2words(int(m.group())).replace('-', ' ').replace(',', ''),
            text,
        )
    for ch in string.punctuation:
        text = text.replace(ch, '')
    return ' '.join(text.split())


def _code_revision():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_prompt(processor, context="", force_language=None):
    """Mirror qwen_asr.Qwen3ASRModel._build_text_prompt: chat-template prefix
    with an empty audio placeholder, generation prompt appended. If
    force_language is given, append "language X<asr_text>" so the model skips
    language-ID and emits plain text directly (matches qwen_asr's transcribe(
    ..., language=...) behavior)."""
    messages = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if force_language:
        base = base + f"language {force_language}<asr_text>"
    return base


@torch.no_grad()
def transcribe(model, processor, audio, device, mix_audio=None, max_new_tokens=256, language=None):
    """Mirror qwen_asr.Qwen3ASRModel._infer_asr_transformers for one utterance."""
    from qwen_asr.inference.utils import parse_asr_output

    text = build_prompt(processor, force_language=language)
    inputs = processor(text=[text], audio=[audio], return_tensors="pt", padding=True)
    inputs = inputs.to(device).to(model.dtype)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    if mix_audio is not None:
        mix_inputs = processor(text=[text], audio=[mix_audio], return_tensors="pt", padding=True)
        mix_inputs = mix_inputs.to(device).to(model.dtype)
        gen_kwargs["mix_input_features"] = mix_inputs["input_features"]
        gen_kwargs["mix_feature_attention_mask"] = mix_inputs["feature_attention_mask"]

    output = model.generate(**inputs, **gen_kwargs)
    decoded = processor.batch_decode(
        output.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    _, parsed_text = parse_asr_output(decoded[0], user_language=language)
    return parsed_text


def _load_run_config(checkpoint_dir):
    """finetune.py writes run_config.json into output_dir at the start of
    every training run; checkpoints are always direct children of output_dir
    (HF Trainer's own convention), so it should sit one directory up from
    checkpoint_dir. Returns None if not found (e.g. an older checkpoint saved
    before this existed) -- callers fall back to hardcoded defaults."""
    candidate = os.path.join(os.path.dirname(os.path.normpath(checkpoint_dir)), "run_config.json")
    if os.path.exists(candidate):
        with open(candidate) as f:
            return json.load(f), candidate
    # Action checkpoint metadata provides a checkpoint-local configuration copy.
    metadata = action_format.read_checkpoint_metadata(checkpoint_dir)
    if metadata and metadata.get("run_config"):
        return metadata["run_config"], os.path.join(checkpoint_dir, action_format.METADATA_FILENAME)
    return None, candidate


def _is_adapter_checkpoint(checkpoint_dir):
    return any(os.path.exists(os.path.join(checkpoint_dir, name)) for name in _ADAPTER_WEIGHT_FILES)


def _read_base_model_txt(checkpoint_dir):
    """Plain-text fallback for a checkpoint saved before run_config.json
    carried "model_path" -- trainer.py's save_model writes this file for
    EVERY checkpoint (full-FT or adapter alike), unconditionally."""
    candidate = os.path.join(checkpoint_dir, "base_model.txt")
    if os.path.exists(candidate):
        with open(candidate) as f:
            return f.read().strip()
    return None


def _read_merged_from_lora_txt(checkpoint_dir):
    candidate = os.path.join(checkpoint_dir, "merged_from_lora.txt")
    if os.path.exists(candidate):
        with open(candidate) as f:
            return f.read().strip()
    return None


def _load_model_for_eval(checkpoint_dir, use_fusion, fusion_type, base_model_path,
                         merged_from_lora_path=None):
    """Load a checkpoint for eval, transparently handling both a full model
    (this codebase's original, and still the common, case) and a PEFT
    adapter checkpoint (any modeling.py apply_train_mode region was
    ft_mode: lora, or an older --use_lora 1 finetune_fusion.sh run).

    A full checkpoint IS the model -- load_qwen3_asr(checkpoint_dir, ...)
    works exactly as it always has. An adapter checkpoint is NOT a model on
    its own (trainer.py's save_model deletes any full-model weight file from
    an adapter checkpoint's directory): it has to be applied on top of the
    ORIGINAL base checkpoint it was trained from, via PeftModel.from_pretrained
    -- same mechanism tests/test_fusion_save_load.py's LoRA round-trip test
    already verified reloads modules_to_save/lora_A/lora_B values correctly.

    action_special_v1 checkpoints use their saved processor to preserve the
    protocol-token mapping. modeling.load_action_special_adapter reconstructs
    the base, production adapter, action rows, and decoder adapter.

    Returns (model, processor, use_bf16), same shape as load_qwen3_asr."""
    metadata = action_format.read_checkpoint_metadata(checkpoint_dir)
    if metadata is None and os.path.exists(os.path.join(checkpoint_dir, action_format.ROWS_FILENAME)):
        raise ValueError(
            f"{checkpoint_dir} has {action_format.ROWS_FILENAME} but no "
            f"{action_format.METADATA_FILENAME}; refusing a legacy fallback that would "
            "silently drop the action rows and tokenizer mapping"
        )
    if metadata and metadata["action_format"] == action_format.ACTION_JUDGE_V1:
        raise ValueError("Use eval_judge.py for judge checkpoints; ASR requires the judge adapter disabled")
    if metadata and action_format.is_special(metadata["action_format"]):
        if metadata.get("materialized"):
            if _is_adapter_checkpoint(checkpoint_dir):
                raise ValueError(f"{checkpoint_dir} claims to be a materialized export but has adapter files")
            model, processor, use_bf16 = load_qwen3_asr(
                checkpoint_dir, use_fusion=use_fusion, fusion_type=fusion_type
            )
            action_format.verify_token_mapping(
                processor.tokenizer, metadata["protocol_tokens"],
                forbidden_ids=modeling.reserved_model_token_ids(model, processor.tokenizer),
            )
            rows = metadata["vocabulary"]["embedding_rows_expanded"]
            if modeling._vocab_dims(model) != (rows, metadata["vocabulary"]["lm_head_rows_expanded"]):
                raise ValueError(f"Exported model dims {modeling._vocab_dims(model)} do not match "
                                 f"the recorded expanded vocabulary")
            return model, processor, use_bf16
        if not _is_adapter_checkpoint(checkpoint_dir):
            raise FileNotFoundError(f"{checkpoint_dir} is action_special_v1 but has no adapter weights")
        base_model_path = base_model_path or metadata.get("base_model_path")
        if merged_from_lora_path is None:
            merged_from_lora_path = metadata.get("production_adapter_path")
        if not base_model_path:
            raise ValueError(f"No base model path for action_special_v1 checkpoint {checkpoint_dir}")
        return modeling.load_action_special_adapter(
            checkpoint_dir, metadata, use_fusion, fusion_type, base_model_path,
            merged_from_lora_path,
        )

    if not _is_adapter_checkpoint(checkpoint_dir):
        return load_qwen3_asr(checkpoint_dir, use_fusion=use_fusion, fusion_type=fusion_type)

    if not base_model_path:
        raise ValueError(
            f"{checkpoint_dir} is a PEFT adapter checkpoint ({_ADAPTER_WEIGHT_FILES}), which has no "
            f"full model weights of its own -- couldn't find a base model path in run_config.json's "
            f"'model_path' or {os.path.join(checkpoint_dir, 'base_model.txt')} either. Pass "
            f"--base_model_path explicitly (the checkpoint this adapter was trained from)."
        )
    print(f"[load] {checkpoint_dir} is a PEFT adapter checkpoint -- loading base "
          f"model from {base_model_path} and applying the adapter on top")
    model, processor, use_bf16 = load_qwen3_asr(base_model_path, use_fusion=use_fusion, fusion_type=fusion_type)
    if merged_from_lora_path:
        print(f"[load] reconstructing adapter base by merging production adapter "
              f"{merged_from_lora_path}")
        model.thinker = PeftModel.from_pretrained(
            model.thinker, merged_from_lora_path, is_trainable=False
        ).merge_and_unload()
    model.thinker = PeftModel.from_pretrained(model.thinker, checkpoint_dir, is_trainable=False)
    return model, processor, use_bf16


def describe_model(model, use_fusion: bool, fusion_type: str, device: str) -> str:
    """Summarize exactly what architecture got loaded, so you can confirm at a
    glance that eval is exercising what you think it is -- which model class,
    which fusion parameter (if any) actually exists on the audio tower, and
    whether that parameter still looks like its untrained no-op init value
    (e.g. because of a --fusion_type mismatch that fell through some other
    way, or a checkpoint that barely trained) versus a value that's clearly
    moved from training."""
    lines = [f"[load] model={type(model).__name__} device={device} "
             f"use_fusion={use_fusion} fusion_type={fusion_type}"]

    tower = getattr(getattr(model, "thinker", None), "audio_tower", None)
    if tower is None:
        lines.append("  (could not find model.thinker.audio_tower)")
    elif not hasattr(tower, "fusion_type"):
        lines.append("  architecture: plain single-stream audio_tower (no fusion module at all)")
    else:
        lines.append(f"  audio_tower.fusion_type (as actually constructed) = {tower.fusion_type!r}")
        if hasattr(tower, "fusion_gate"):
            # weight AND bias are both zero-initialized (the near-zero-at-init
            # gate behavior comes from GATE_INIT_SHIFT, a fixed constant added
            # at forward time, not a stored/trained parameter -- see
            # Qwen3ASRAudioEncoderLateGate's docstring for why: a trainable
            # bias sitting at a large magnitude like -5.0 is invisible to bf16
            # optimizer updates). So "moved from init" here means either
            # tensor is no longer ~0, not a comparison to -5.0.
            w_absmax = tower.fusion_gate.weight.detach().float().abs().max().item()
            b_absmax = tower.fusion_gate.bias.detach().float().abs().max().item()
            near_init = w_absmax < 1e-4 and b_absmax < 1e-4
            lines.append(
                f"  fusion_gate present: weight.abs().max()={w_absmax:.5f} bias.abs().max()={b_absmax:.5f} "
                f"({'looks UNTRAINED / still ~no-op init (0.0)' if near_init else 'has moved from init -- trained'})"
            )
        if hasattr(tower, "conv2d1_mix"):
            w_absmax = tower.conv2d1_mix.weight.detach().float().abs().max().item()
            near_init = w_absmax < 1e-4
            lines.append(
                f"  conv2d1_mix present: weight.abs().max()={w_absmax:.5f} "
                f"({'looks UNTRAINED / still ~no-op init (0.0)' if near_init else 'has moved from init -- trained'})"
            )
        if hasattr(tower, "film_proj"):
            w_absmax = tower.film_proj.weight.detach().float().abs().max().item()
            near_init = w_absmax < 1e-4
            lines.append(
                f"  fddt film_proj present: weight.abs().max()={w_absmax:.5f} "
                f"({'looks UNTRAINED / still ~no-op init (0.0)' if near_init else 'has moved from init -- trained'})"
            )
        if hasattr(tower, "cross_attn"):
            w_absmax = tower.cross_attn.out_proj.weight.detach().float().abs().max().item()
            near_init = w_absmax < 1e-4
            lines.append(
                f"  cross_attn out_proj present: weight.abs().max()={w_absmax:.5f} "
                f"({'looks UNTRAINED / still ~no-op init (0.0)' if near_init else 'has moved from init -- trained'})"
            )

    if hasattr(model.thinker, "peft_config"):
        # PEFT-wrapped (any modeling.py apply_train_mode region was ft_mode:
        # lora) -- the per-attribute checks above assume a plain nn.Linear/
        # nn.Conv2d at those attributes, which no longer holds once PEFT
        # wraps a region's modules (lora_A/lora_B added alongside a frozen
        # base, or a modules_to_save wrapper substituted in). This summary
        # is authoritative regardless of wrapping: it classifies every
        # parameter the SAME way apply_train_mode/trainer.py's optimizer do
        # (modeling.classify_param_region).
        #
        # Deliberately NOT based on param.requires_grad here: _load_model_for_eval
        # loads with PeftModel.from_pretrained(..., is_trainable=False), which
        # sets requires_grad=False on EVERY parameter (correct for inference) --
        # checking requires_grad post-load would show all-zero regardless of
        # what was actually trained. lora_A/lora_B and modules_to_save are
        # structural (present or not, independent of requires_grad), so those
        # are what's counted instead.
        fusion_names = modeling._FUSION_MODULE_NAMES.get(fusion_type, []) if use_fusion else []
        lora_counts = {r: 0 for r in modeling.REGIONS}
        full_counts = {r: 0 for r in modeling.REGIONS}
        for name, _ in model.thinker.named_parameters():
            region = modeling.classify_param_region(name, fusion_names)
            if region not in modeling.REGIONS:
                continue
            if "lora_A" in name or "lora_B" in name:
                lora_counts[region] += 1
            if "modules_to_save" in name:
                full_counts[region] += 1
        lines.append("  PEFT adapter active (region activity shown structurally, not via "
                      "requires_grad -- eval loads with is_trainable=False):")
        lines.append(
            "    LoRA (lora_A/lora_B) tensor counts by region: "
            + ", ".join(f"{r}={lora_counts[r]}" for r in modeling.REGIONS)
        )
        lines.append(
            "    Full-FT-under-PEFT (modules_to_save) tensor counts by region: "
            + ", ".join(f"{r}={full_counts[r]}" for r in modeling.REGIONS)
        )

    total = sum(p.numel() for p in model.parameters())
    first_param = next(model.parameters())
    lines.append(f"  total params={total:,}  dtype={first_param.dtype}  param_device={first_param.device}")
    return "\n".join(lines)


def _eval_collator_name(train_collator_name):
    """Map a training-time collator name to this script's --collator choice.
    Only white_noise_test needs special eval-time audio construction; "none"
    and "mix" both mean "read audio_sep/audio_mix from the val jsonl as-is",
    which is this script's existing default ("none") path -- --use_fusion
    already controls whether audio_mix gets read in that path."""
    return "white_noise_test" if train_collator_name == "white_noise_test" else "none"


def _dispatch_action_eval(args):
    """Keep one CLI while retaining checkpoint-specific inference protocols."""
    metadata = action_format.read_checkpoint_metadata(args.checkpoint_dir)
    config, _ = _load_run_config(args.checkpoint_dir)
    fmt = metadata['action_format'] if metadata else (config or {}).get('action_format')
    enabled = (config or {}).get('action_training')
    is_action = bool(metadata or enabled or (enabled is None and fmt))
    task = ('action' if is_action else 'asr') if args.task == 'auto' else args.task
    if task == 'asr':
        if (args.mode != 'pipeline' or args.early_reject or args.decoding_mode != 'unconstrained'
                or args.wer_threshold is not None or args.quality_feature_mode is not None):
            raise ValueError('Action evaluation options require an action checkpoint or --task action')
        return False
    if not args.out_json or not args.summary_json:
        raise ValueError('Action evaluation requires --out_json and --summary_json')
    if args.num_shards != 1 or args.shard_index != 0 or args.skip_malformed:
        raise ValueError('Action evaluation does not support sharding or --skip_malformed')
    if args.collator not in (None, 'none') or args.audio_field != 'audio_sep':
        raise ValueError('Action evaluation requires the original audio_sep/audio_mix streams')
    if fmt == action_format.ACTION_JUDGE_V1:
        if args.early_reject or args.quality_feature_mode not in (None, 'none'):
            raise ValueError('Judge evaluation does not support early rejection or quality hints')
        import eval_judge
        eval_judge.evaluate(
            checkpoint=args.checkpoint_dir, manifest=args.val_file,
            out_json=args.out_json, summary_json=args.summary_json,
            mode=args.mode, decoding_mode=args.decoding_mode,
            max_new_tokens=args.max_new_tokens, device=args.device,
            limit=args.limit, base_model_path=args.base_model_path,
            production_adapter_path=args.merged_from_lora_path,
            use_fusion=args.use_fusion, fusion_type=args.fusion_type,
            language=args.language, sr=args.sr, wer_threshold=args.wer_threshold,
            quiet=args.quiet,
        )
    else:
        if args.mode != 'pipeline':
            raise ValueError('Cached mode requires an action_judge_v1 checkpoint')
        import eval_action_checkpoint
        forwarded = ['--checkpoint_dir', args.checkpoint_dir, '--val_file', args.val_file,
                     '--out_json', args.out_json, '--summary_json', args.summary_json,
                     '--max_new_tokens', str(args.max_new_tokens), '--sr', str(args.sr),
                     '--decoding_mode', args.decoding_mode]
        for name in ('base_model_path', 'merged_from_lora_path', 'use_fusion', 'fusion_type',
                     'language', 'quality_feature_mode', 'wer_threshold', 'device', 'limit'):
            value = getattr(args, name)
            if value is not None:
                forwarded.extend(['--' + name, str(value)])
        if args.quiet:
            forwarded.append('--quiet')
        if args.early_reject:
            forwarded.append('--early_reject')
        eval_action_checkpoint.main(forwarded)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument('--task', choices=['auto', 'asr', 'action'], default='auto',
                    help='Detect checkpoint metadata by default; use action for metadata-free action_first checkpoints.')
    ap.add_argument('--mode', '--action_mode', choices=['pipeline', 'cached'], default='pipeline',
                    help='Action judge: generate production transcripts or judge prepared saved candidates.')
    ap.add_argument('--decoding_mode', choices=action_format.DECODING_MODES, default='unconstrained')
    ap.add_argument('--wer_threshold', type=float, default=None)
    ap.add_argument('--quality_feature_mode', choices=['none', 'manifest'], default=None)
    ap.add_argument('--early_reject', action='store_true')
    ap.add_argument("--use_fusion", type=int, choices=[0, 1], default=None,
                    help="1 to enable fusion generation. Auto-filled from "
                         "run_config.json next to --checkpoint_dir if left unset.")
    ap.add_argument("--fusion_type", default=None,
                    choices=["late_gate", "early_conv", "fddt", "cross_attn"],
                    help="Auto-filled from run_config.json if left unset.")
    ap.add_argument("--base_model_path", default=None,
                    help="Only needed if --checkpoint_dir is a PEFT adapter "
                         "checkpoint (any ft_params region was ft_mode: lora, "
                         "or an older --use_lora 1 finetune_fusion.sh run) -- "
                         "the base checkpoint the adapter was trained from. "
                         "Auto-filled from run_config.json's 'model_path' "
                         "(or that checkpoint's base_model.txt, for one saved "
                         "before that field existed) if left unset. Ignored "
                         "for a plain full-model checkpoint.")
    ap.add_argument("--merged_from_lora_path", default=None,
                    help="Production adapter merged before this checkpoint's new "
                         "adapter was trained. Auto-filled from run_config.json or "
                         "merged_from_lora.txt; normally do not set manually.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--language", default=None,
                    help="force this language (e.g. English) so the model skips "
                         "language-ID and emits plain text directly, matching "
                         "qwen_asr's transcribe(..., language=...). Auto-filled "
                         "from run_config.json if left unset (falls back to "
                         "'English', finetune.py's own default, if no "
                         "run_config.json is found). Pass '' explicitly to "
                         "evaluate without any forced-language tag.")
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N rows")
    ap.add_argument("--skip_malformed", action="store_true",
                    help="Skip malformed JSONL lines while preserving their global line indices.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Split the (optionally limited) source rows into deterministic "
                         "global-index modulo shards for parallel outcome generation.")
    ap.add_argument("--shard_index", type=int, default=0,
                    help="Zero-based shard to evaluate; requires 0 <= index < num_shards.")
    ap.add_argument("--out_json", default=None,
                    help="if set, write per-sample results (index, audio_sep path, ref, hyp, wer) "
                         "as JSON lines to this path, for downstream analysis/plotting")
    ap.add_argument("--summary_json", default=None,
                    help="if set, write the CORPUS-level result (checkpoint_dir, "
                         "n_samples, corpus_wer, corpus_cer) as a single JSON object "
                         "to this path. Machine-readable counterpart of the summary "
                         "printed at the end -- scripts/sweep_eval.py reads this "
                         "instead of scraping stdout.")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the per-sample WER/ref/hyp lines. The corpus "
                         "summary is still printed. Use when evaluating many "
                         "checkpoints in a loop, where the per-sample dump is noise.")
    ap.add_argument("--audio_field", default="audio_sep",
                    help="which JSONL field to read as the input audio path (default: audio_sep). "
                         "E.g. pass 'libritts_path' to evaluate on the clean pre-mix source instead.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--collator", default=None, choices=["none", "white_noise_test"],
                    help="How to build eval audio per row. 'none' reads "
                         "audio_sep/audio_mix from the val jsonl as-is. "
                         "'white_noise_test' ignores audio_sep/audio_mix and "
                         "instead builds BOTH streams from row['libritts_path'] "
                         "using the exact same disjoint-noise-masking code as "
                         "training (dataloader.Qwen3ASRCollator_WhiteNoise_Test) "
                         "-- use this to evaluate a checkpoint trained with that "
                         "collator. Requires --use_fusion 1. Auto-filled from "
                         "run_config.json if left unset.")
    ap.add_argument("--wn_seed", type=int, default=None,
                    help="[--collator white_noise_test only] base seed, combined "
                         "with a stable per-path hash. Match the --wn_seed the "
                         "checkpoint was trained with to reproduce the identical "
                         "corruption pattern, or use a different seed to check "
                         "generalization to a fresh corruption draw. Auto-filled "
                         "from run_config.json if left unset.")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if args.max_new_tokens < 1:
        raise ValueError('--max_new_tokens must be positive')
    destinations = [os.path.realpath(p) for p in (args.out_json, args.summary_json) if p]
    if len(set(destinations)) != len(destinations) or os.path.realpath(args.val_file) in destinations:
        raise ValueError('Prediction, summary, and input manifest paths must be different')
    if _dispatch_action_eval(args):
        return
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"Invalid shard selection: shard_index={args.shard_index}, "
            f"num_shards={args.num_shards}"
        )

    run_config, run_config_path = _load_run_config(args.checkpoint_dir)
    if run_config is not None:
        print(f"[run_config] found {run_config_path}: {run_config}")
    else:
        print(f"[run_config] none found at {run_config_path} -- "
              f"falling back to hardcoded defaults for any unset flag")

    if args.use_fusion is None:
        args.use_fusion = int(run_config["use_fusion"]) if run_config else 0
    if args.fusion_type is None:
        args.fusion_type = run_config["fusion_type"] if run_config else "late_gate"
    if args.collator is None:
        args.collator = _eval_collator_name(run_config["collator"]) if run_config else "none"
    if args.wn_seed is None:
        args.wn_seed = run_config.get("wn_seed", 0) if run_config else 0
    if args.language is None:
        args.language = run_config.get("language", "English") if run_config else "English"
    if args.base_model_path is None:
        args.base_model_path = (
            (run_config.get("model_path") if run_config else None)
            or _read_base_model_txt(args.checkpoint_dir)
        )
    if args.merged_from_lora_path is None:
        args.merged_from_lora_path = (
            (run_config.get("merged_from_lora_path") if run_config else None)
            or _read_merged_from_lora_txt(args.checkpoint_dir)
        )

    if args.collator == "white_noise_test" and not args.use_fusion:
        raise ValueError(
            "--collator white_noise_test requires --use_fusion 1 -- the "
            "checkpoint must actually consume mix_input_features for this "
            "diagnostic to test anything."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, _ = _load_model_for_eval(
        args.checkpoint_dir, bool(args.use_fusion), args.fusion_type,
        args.base_model_path, args.merged_from_lora_path,
    )
    model.to(device).eval()
    if device == 'cpu':
        model.float()
    print(describe_model(model, bool(args.use_fusion), args.fusion_type, device))

    rows = []
    malformed_indices = []
    with open(args.val_file) as source_handle:
        for global_index, line in enumerate(source_handle):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if not args.skip_malformed:
                    raise ValueError(
                        f"Malformed JSON at {args.val_file}:{global_index + 1}: {exc}"
                    ) from exc
                malformed_indices.append(global_index)
                continue
            rows.append((global_index, row))
    if args.limit:
        rows = rows[: args.limit]
    indexed_rows = [
        (global_index, row)
        for global_index, row in rows
        if global_index % args.num_shards == args.shard_index
    ]
    if not indexed_rows:
        raise ValueError('No samples selected for evaluation')
    for output in (args.out_json, args.summary_json):
        if output:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    # Reuse the EXACT training-time masking code (not a reimplementation) so
    # eval audio is constructed identically to what the checkpoint was
    # trained on -- only the constructor's audio-loading/masking step is
    # used here, no processor/tokenization from this collator.
    wn_collator = None
    if args.collator == "white_noise_test":
        wn_collator = Qwen3ASRCollator_WhiteNoise_Test(
            processor=processor, sampling_rate=args.sr, seed=args.wn_seed
        )

    out_f = open(args.out_json, "w") if args.out_json else None

    refs_n, hyps_n, utterance_wers = [], [], []
    for i, row in indexed_rows:
        if wn_collator is not None:
            audios, audios_mix, _ = wn_collator._load_masked_channels([row])
            sep_audio, mix_audio = audios[0], audios_mix[0]
        else:
            audio_path = row.get(args.audio_field, row.get("audio_sep", row.get("audio")))
            sep_audio = read_audio(audio_path, args.sr)
            mix_audio = None
            if args.use_fusion:
                mix_path = row.get("audio_mix")
                if mix_path:
                    mix_audio = read_audio(mix_path, args.sr)

        hyp = transcribe(model, processor, sep_audio, device, mix_audio=mix_audio,
                          max_new_tokens=args.max_new_tokens, language=args.language)
        ref = row["text"]
        ref_n, hyp_n = normalize_string(ref), normalize_string(hyp)
        sample_wer = jiwer.wer(ref_n, hyp_n)
        utterance_wers.append(sample_wer)

        refs_n.append(ref_n)
        hyps_n.append(hyp_n)

        if not args.quiet:
            print(f"[{i:04d}] WER={sample_wer:.3f}")
            print(f"  ref: {ref_n}")
            print(f"  hyp: {hyp_n}")

        if out_f is not None:
            out_f.write(json.dumps({
                "index": i,
                "audio_path": audio_path if wn_collator is None else row.get("libritts_path"),
                "ref": ref_n,
                "hyp": hyp_n,
                "wer": sample_wer,
            }) + "\n")
            out_f.flush()

    if out_f is not None:
        out_f.close()

    corpus_wer = jiwer.wer(refs_n, hyps_n)
    corpus_cer = jiwer.cer(refs_n, hyps_n)
    print(f"\n=== {len(indexed_rows)} samples (shard {args.shard_index}/{args.num_shards}) ===")
    print(f"corpus WER (normalized): {corpus_wer:.4f}")
    print(f"corpus CER (normalized): {corpus_cer:.4f}")
    print(f"mean utterance WER: {statistics.mean(utterance_wers):.4f}")
    print(f"median utterance WER: {statistics.median(utterance_wers):.4f}")

    if args.summary_json:
        # Write aggregate metrics after every selected sample has been evaluated.
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump({
                "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
                "val_file": os.path.abspath(args.val_file),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "code_revision": _code_revision(),
                "use_fusion": bool(args.use_fusion),
                "fusion_type": args.fusion_type,
                "collator": args.collator,
                "language": args.language,
                "sampling_rate": args.sr,
                "normalization": "normalize_string_v1",
                "n_samples": len(indexed_rows),
                "source_rows_before_sharding": len(rows),
                "malformed_source_indices_skipped": malformed_indices,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "corpus_wer": corpus_wer,
                "corpus_cer": corpus_cer,
                "mean_utterance_wer": statistics.mean(utterance_wers),
                "median_utterance_wer": statistics.median(utterance_wers),
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
            }, f, indent=2)
        print(f"[summary_json] wrote {args.summary_json}")


if __name__ == "__main__":
    main()
