#!/usr/bin/env python3
"""Free-generation evaluation for an action Mega-ASR checkpoint.

The output format is dispatched from the checkpoint's recorded action_format
(checkpoint-local action_format.json, else run_config.json, else legacy
action_first) -- never guessed from generated text.

action_special_v1 runs two passes per utterance:
  1. transcript pass -- ordinary greedy ASR generation from the prompt (no
     action token is ever in this sequence);
  2. action pass -- the generation cache is cropped back to the prompt and
     <|action_query|> is run on top of it, so the action sees only the prompt
     (template + audio), never the transcript; the greedy token at the query
     position is the action.
Both are parsed from token IDs, never from text decoded with
skip_special_tokens=True. Invalid outputs are kept in every metric: an
invalid action's effective policy action is REJECT, reported separately from
successfully generated actions, and WER uses whatever transcript text
preceded the problem (empty if none).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter

import jiwer
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import action_format  # noqa: E402
import modeling  # noqa: E402
from dataloader import build_quality_prefix, read_audio  # noqa: E402
from eval_checkpoint import (  # noqa: E402
    _load_model_for_eval,
    _load_run_config,
    _read_base_model_txt,
    _read_merged_from_lora_txt,
    build_prompt,
    describe_model,
    normalize_string,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--base_model_path", default=None)
    parser.add_argument("--merged_from_lora_path", default=None)
    parser.add_argument("--use_fusion", type=int, default=None)
    parser.add_argument(
        "--fusion_type", default=None,
        choices=["late_gate", "early_conv", "fddt", "cross_attn"],
    )
    parser.add_argument("--language", default=None)
    parser.add_argument("--quality_feature_mode", choices=["none", "manifest"], default=None)
    parser.add_argument("--wer_threshold", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--decoding_mode", choices=action_format.DECODING_MODES, default="unconstrained",
        help="[action_special_v1] 'unconstrained' greedy (full-vocabulary argmax at the "
             "query) is the baseline. 'constrained' picks the larger of the two action "
             "logits and suppresses protocol tokens during transcript generation; it "
             "guarantees syntax, so always report an unconstrained run too.",
    )
    parser.add_argument(
        "--early_reject", action="store_true",
        help="[action_special_v1] serving simulation: run the action pass FIRST and "
             "skip transcript generation on REJECT (transcript=null, format validity "
             "not applicable). Offline WER is then not comparable to a full-output run.",
    )
    parser.add_argument("--audio_field", default="audio_sep")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


@torch.no_grad()
def generate_action_text(
    model,
    processor,
    row,
    sep_audio,
    device,
    language,
    quality_feature_mode,
    mix_audio=None,
    max_new_tokens=256,
):
    prompt = build_prompt(processor, context=row.get("prompt", ""), force_language=language)
    prompt += build_quality_prefix(row.get("quality_context", ""), quality_feature_mode)
    inputs = processor(text=[prompt], audio=[sep_audio], return_tensors="pt", padding=True)
    inputs = inputs.to(device).to(model.dtype)
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if mix_audio is not None:
        mix_inputs = processor(text=[prompt], audio=[mix_audio], return_tensors="pt", padding=True)
        mix_inputs = mix_inputs.to(device).to(model.dtype)
        generate_kwargs["mix_input_features"] = mix_inputs["input_features"]
        generate_kwargs["mix_feature_attention_mask"] = mix_inputs["feature_attention_mask"]
    generated = model.generate(**inputs, **generate_kwargs)
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    return processor.batch_decode(
        sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


@torch.no_grad()
def generate_special(
    model,
    processor,
    row,
    sep_audio,
    device,
    language,
    quality_feature_mode,
    protocol_ids,
    eos_ids,
    mix_audio=None,
    max_new_tokens=256,
    decoding_mode="unconstrained",
    early_reject=False,
):
    """Same audio + prompt as training. Returns (transcript continuation IDs
    or None if skipped, chosen action token ID, logged action logits)."""
    prompt = build_prompt(processor, context=row.get("prompt", ""), force_language=language)
    prompt += build_quality_prefix(row.get("quality_context", ""), quality_feature_mode)
    inputs = processor(text=[prompt], audio=[sep_audio], return_tensors="pt", padding=True)
    inputs = inputs.to(device).to(model.dtype)
    prompt_length = inputs["input_ids"].shape[1]
    query_id = protocol_ids[action_format.QUERY]
    mix_inputs = None
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "eos_token_id": list(eos_ids),
        "use_cache": True,
    }
    if decoding_mode == "constrained":
        generate_kwargs["suppress_tokens"] = list(protocol_ids.values())
    if mix_audio is not None:
        mix_inputs = processor(text=[prompt], audio=[mix_audio], return_tensors="pt", padding=True)
        mix_inputs = mix_inputs.to(device).to(model.dtype)
        generate_kwargs["mix_input_features"] = mix_inputs["input_features"]
        generate_kwargs["mix_feature_attention_mask"] = mix_inputs["feature_attention_mask"]

    action_logits = None
    if early_reject:
        action_logits = modeling.action_logits_fresh(model, inputs, query_id, mix_inputs)[0]
        token = action_format.choose_action_token(action_logits, protocol_ids, decoding_mode)
        if token == protocol_ids["REJECT"]:
            return None, token, _logged_logits(action_logits, protocol_ids, token)

    generated = model.generate(**inputs, **generate_kwargs)
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    continuation = sequences[0, prompt_length:].tolist()
    if action_logits is None:
        action_logits = modeling.action_logits_from_cache(
            model, generated.past_key_values, prompt_length, query_id
        )
    token = action_format.choose_action_token(action_logits, protocol_ids, decoding_mode)
    return continuation, token, _logged_logits(action_logits, protocol_ids, token)


def _logged_logits(action_logits, protocol_ids, token):
    """Raw query-position logits of the two actions plus the overall argmax.
    A two-action softmax of these is not a calibrated correctness probability."""
    logged = {action: float(action_logits[protocol_ids[action]])
              for action in action_format.ACTIONS}
    logged["argmax_id"] = int(action_logits.argmax().item())
    logged["chosen_id"] = int(token)
    return logged


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def action_metrics(truth, predictions):
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Action metric inputs must be non-empty and have equal lengths")
    tp = sum(t == "REJECT" and p == "REJECT" for t, p in zip(truth, predictions))
    tn = sum(t == "RESPOND" and p == "RESPOND" for t, p in zip(truth, predictions))
    fp = sum(t == "RESPOND" and p == "REJECT" for t, p in zip(truth, predictions))
    fn = sum(t == "REJECT" and p == "RESPOND" for t, p in zip(truth, predictions))
    reject_precision = _safe_div(tp, tp + fp)
    reject_recall = _safe_div(tp, tp + fn)
    respond_recall = _safe_div(tn, tn + fp)
    return {
        "accuracy": (tp + tn) / len(truth),
        "balanced_accuracy": (reject_recall + respond_recall) / 2,
        "reject_precision": reject_precision,
        "reject_recall": reject_recall,
        "reject_f1": _safe_div(2 * reject_precision * reject_recall, reject_precision + reject_recall),
        "false_accept_rate": _safe_div(fn, tp + fn),
        "false_reject_rate": _safe_div(fp, tn + fp),
        "confusion_matrix": {
            "actual_REJECT_pred_REJECT": tp,
            "actual_REJECT_pred_RESPOND": fn,
            "actual_RESPOND_pred_REJECT": fp,
            "actual_RESPOND_pred_RESPOND": tn,
        },
    }


def main(argv=None):
    args = parse_args(argv)
    checkpoint_metadata = action_format.read_checkpoint_metadata(args.checkpoint_dir)
    run_config, run_config_path = _load_run_config(args.checkpoint_dir)
    if checkpoint_metadata and checkpoint_metadata.get("run_config"):
    # Checkpoint-local configuration takes precedence over the parent run config.
        run_config = checkpoint_metadata["run_config"]
        run_config_path = os.path.join(args.checkpoint_dir, action_format.METADATA_FILENAME)
    if run_config:
        print(f"[run_config] found {run_config_path}: {run_config}")
    if checkpoint_metadata:
        fmt = checkpoint_metadata["action_format"]
    else:
        fmt = (run_config or {}).get("action_format") or action_format.ACTION_FIRST
    fmt = action_format.validate_format(fmt)
    if fmt == action_format.ACTION_JUDGE_V1:
        raise ValueError("Transcript-conditioned judge requires eval_judge.py; refusing audio-only evaluation")
    special = action_format.is_special(fmt)
    if special and checkpoint_metadata is None:
        raise ValueError(
            f"run_config says action_special_v1 but {args.checkpoint_dir} has no "
            f"{action_format.METADATA_FILENAME}; refusing to guess the token mapping"
        )
    if not special and (args.decoding_mode != "unconstrained" or args.early_reject):
        raise ValueError("--decoding_mode/--early_reject only apply to action_special_v1")
    print(f"[action_format] {fmt}")
    if args.use_fusion is None:
        args.use_fusion = int(run_config.get("use_fusion", False)) if run_config else 0
    if args.fusion_type is None:
        args.fusion_type = run_config.get("fusion_type", "late_gate") if run_config else "late_gate"
    if args.language is None:
        args.language = run_config.get("language", "English") if run_config else "English"
    if args.quality_feature_mode is None:
        args.quality_feature_mode = run_config.get("quality_feature_mode", "none") if run_config else "none"
    if args.wer_threshold is None:
        args.wer_threshold = float(run_config.get("action_wer_threshold", 0.75)) if run_config else 0.75
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
    if not math.isfinite(args.wer_threshold) or args.wer_threshold < 0:
        raise ValueError("WER threshold must be finite and non-negative")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, _ = _load_model_for_eval(
        args.checkpoint_dir,
        bool(args.use_fusion),
        args.fusion_type,
        args.base_model_path,
        args.merged_from_lora_path,
    )
    model.to(device).eval()
    print(describe_model(model, bool(args.use_fusion), args.fusion_type, device))
    tokenizer = processor.tokenizer
    protocol_ids = eos_ids = None
    if special:
        protocol_ids = action_format.verify_token_mapping(
            tokenizer, checkpoint_metadata["protocol_tokens"],
            forbidden_ids=modeling.reserved_model_token_ids(model, tokenizer),
        )
        eos_ids = modeling.generation_eos_ids(tokenizer)
        print(f"[protocol_tokens] {protocol_ids} eos_ids={eos_ids} decoding_mode={args.decoding_mode} "
              f"early_reject={args.early_reject}")

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    with open(args.val_file, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Validation manifest is empty")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    static_truth = []
    realized_truth = []
    predictions = []
    references = []
    hypotheses = []
    baseline_hypotheses = []
    respond_references = []
    respond_hypotheses = []
    respond_utterance_wers = []
    labeled_respond_references = []
    labeled_respond_hypotheses = []
    labeled_respond_utterance_wers = []
    utterance_wers = []
    parses = []
    generated_lengths = []
    latencies = []

    with open(args.out_json, "w", encoding="utf-8") as output:
        for index, row in enumerate(rows):
            action = str(row.get("action", "")).upper()
            if action not in {"RESPOND", "REJECT"}:
                raise ValueError(f"Missing/invalid static action at validation row {index}")
            sep_path = row.get(args.audio_field) or row.get("audio_sep") or row.get("audio")
            sep_audio = read_audio(sep_path, args.sr)
            mix_audio = None
            if args.use_fusion:
                mix_path = row.get("audio_mix")
                if not mix_path:
                    raise ValueError(f"Fusion evaluation row {index} lacks audio_mix")
                mix_audio = read_audio(mix_path, args.sr)

            started = time.perf_counter()
            action_logits = None
            if special:
                continuation, action_token, action_logits = generate_special(
                    model, processor, row, sep_audio, device, args.language,
                    args.quality_feature_mode, protocol_ids, eos_ids,
                    mix_audio=mix_audio, max_new_tokens=args.max_new_tokens,
                    decoding_mode=args.decoding_mode, early_reject=args.early_reject,
                )
                parsed = action_format.combine_parse(
                    continuation, action_token, protocol_ids, eos_ids, decode,
                    pad_id=tokenizer.pad_token_id, max_new_tokens=args.max_new_tokens,
                )
                generated_lengths.append(len(continuation or []))
            else:
                raw_output = generate_action_text(
                    model,
                    processor,
                    row,
                    sep_audio,
                    device,
                    args.language,
                    args.quality_feature_mode,
                    mix_audio=mix_audio,
                    max_new_tokens=args.max_new_tokens,
                )
                parsed = action_format.parse_legacy_text(raw_output)
                generated_lengths.append(len(tokenizer(raw_output, add_special_tokens=False)["input_ids"]))
            latencies.append(time.perf_counter() - started)
            parses.append(parsed)
            predicted_action = parsed.effective_action
            valid_format = parsed.format_valid
            reference_n = normalize_string(row["text"])
            hypothesis_n = normalize_string(parsed.transcript or "") if parsed.transcript_generated else None
            sample_wer = jiwer.wer(reference_n, hypothesis_n) if parsed.transcript_generated else None
            realized_action = (
                ("RESPOND" if sample_wer <= args.wer_threshold else "REJECT")
                if sample_wer is not None else None
            )

            static_truth.append(action)
            realized_truth.append(realized_action)
            predictions.append(predicted_action)
            references.append(reference_n)
            hypotheses.append(hypothesis_n)
            utterance_wers.append(sample_wer)
            baseline_hypotheses.append(normalize_string(row.get("label_hypothesis", "")))
            if predicted_action == "RESPOND":
                respond_references.append(reference_n)
                respond_hypotheses.append(hypothesis_n)
                respond_utterance_wers.append(sample_wer)
            if action == "RESPOND":
                labeled_respond_references.append(reference_n)
                labeled_respond_hypotheses.append(hypothesis_n)
                labeled_respond_utterance_wers.append(sample_wer)

            result = {
                "index": index,
                "source_index": row.get("source_index", index),
                "audio_path": sep_path,
                "static_action": action,
                "realized_action": realized_action,
                "predicted_action": predicted_action,
                "parsed_action": parsed.parsed_action,
                "action_valid": parsed.action_valid,
                "valid_format": valid_format,
                "failure_reason": parsed.failure_reason,
                "action_failure_reason": parsed.action_failure_reason,
                "action_token_id": parsed.action_token_id,
                "transcript_generated": parsed.transcript_generated,
                "raw_output": parsed.raw_text,
                "raw_generated_ids": parsed.raw_ids if special else None,
                "action_logits": action_logits,
                "reference": reference_n,
                "hypothesis": hypothesis_n,
                "wer": sample_wer,
            }
            output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            if not args.quiet:
                print(
                    f"[{index:04d}] static={action} predicted={predicted_action} "
                    f"realized={realized_action} WER={sample_wer if sample_wer is not None else 'not generated'} "
                    f"action_valid={parsed.action_valid} valid={valid_format}"
                    + (f" reason={parsed.failure_reason}" if parsed.failure_reason else "")
                )

    static = action_metrics(static_truth, predictions)
    realized = action_metrics(realized_truth, predictions) if not args.early_reject else None
    counts = Counter(static_truth)
    n = len(rows)
    corpus_wer = jiwer.wer(references, hypotheses) if not args.early_reject else None
    action_valid_count = sum(p.action_valid for p in parses)
    format_applicable = [p for p in parses if p.format_valid is not None]
    format_valid_count = sum(bool(p.format_valid) for p in format_applicable)
    summary = {
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "val_file": os.path.abspath(args.val_file),
        "n_samples": n,
        "action_format": fmt,
        "protocol_token_ids": protocol_ids,
        "layout": (
            "transcript pass ([prompt] -> transcript<EOS>) then action pass "
            "([prompt] <|action_query|> -> action, generation cache cropped to the prompt)"
            if special else "legacy single sequence ACTION: X\\nTRANSCRIPT: ..."
        ),
        "decoding_mode": args.decoding_mode if special else "legacy_text_greedy",
        "early_reject": bool(args.early_reject),
        "transcript_metrics_available": not args.early_reject,
        "transcript_metrics_unavailable_reason": (
            "early_reject is a serving simulation; run full generation for transcript-quality metrics"
            if args.early_reject else None
        ),
        "max_new_tokens": args.max_new_tokens,
        "invalid_output_policy": (
            "effective action is RESPOND only for a valid RESPOND with a valid transcript; "
            "otherwise REJECT (an invalid action is never counted as generated, and "
            "parsed_action keeps the model's choice); invalid transcript -> WER uses the "
            "text before the first EOS/protocol token, else the empty string"
            if special else
            "legacy: effective action REJECT with an empty transcript for WER"
        ),
        "wer_threshold": args.wer_threshold,
        "static_label_counts": dict(counts),
        "majority_class": counts.most_common(1)[0][0],
        "majority_accuracy_baseline": counts.most_common(1)[0][1] / n,
        "action_valid_rate": action_valid_count / n,
        "format_valid_rate": (
            format_valid_count / len(format_applicable) if format_applicable else None
        ),
        "format_validity_applicable_count": len(format_applicable),
        "invalid_format_rate": (
            1 - format_valid_count / len(format_applicable) if format_applicable else None
        ),
        "failure_reason_counts": dict(Counter(p.failure_reason for p in parses if p.failure_reason)),
        "action_failure_reason_counts": dict(
            Counter(p.action_failure_reason for p in parses if p.action_failure_reason)
        ),
        "generated_action_counts": dict(Counter(p.parsed_action for p in parses if p.action_valid)),
        "transcript_generated_count": sum(p.transcript_generated for p in parses),
        "mean_generated_tokens": sum(generated_lengths) / n,
        "mean_latency_sec": sum(latencies) / n,
        "static_label_metrics": static,
        "realized_policy_metrics": realized,
        "corpus_wer": corpus_wer,
        "mean_utterance_wer": sum(utterance_wers) / len(utterance_wers) if not args.early_reject else None,
        "median_utterance_wer": statistics.median(utterance_wers) if utterance_wers and not args.early_reject else None,
        "baseline_corpus_wer": jiwer.wer(references, baseline_hypotheses),
        "coverage": predictions.count("RESPOND") / n,
        "unsafe_coverage": sum(
            predicted == "RESPOND" and realized_action == "REJECT"
            for predicted, realized_action in zip(predictions, realized_truth)
        ) / n if not args.early_reject else None,
        "respond_corpus_wer": (
            jiwer.wer(respond_references, respond_hypotheses)
            if respond_references and not args.early_reject else None
        ),
        "respond_mean_utterance_wer": (
            sum(respond_utterance_wers) / len(respond_utterance_wers)
            if respond_utterance_wers and not args.early_reject else None
        ),
        "respond_median_utterance_wer": (
            statistics.median(respond_utterance_wers)
            if respond_utterance_wers and not args.early_reject else None
        ),
        "labeled_respond_corpus_wer": (
            jiwer.wer(labeled_respond_references, labeled_respond_hypotheses)
            if labeled_respond_references and not args.early_reject else None
        ),
        "labeled_respond_mean_utterance_wer": (
            sum(labeled_respond_utterance_wers) / len(labeled_respond_utterance_wers)
            if labeled_respond_utterance_wers and not args.early_reject else None
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"[results] wrote {args.out_json}")
    print(f"[summary] wrote {args.summary_json}")


if __name__ == "__main__":
    main()
