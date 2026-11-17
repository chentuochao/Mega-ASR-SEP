# coding=utf-8
import json
import math
import os
from collections import Counter

from transformers import TrainingArguments

import action_format
from arguments import parse_args
from checkpointing import MakeCheckpointInferableCallback, find_latest_checkpoint
from dataloader import build_collator, build_datasets
from modeling import (
    apply_lora,
    apply_train_mode,
    enable_action_row_training,
    load_qwen3_asr,
    merge_lora_checkpoint,
    parse_ft_params,
    setup_action_vocabulary,
)
from trainer import MegaASRTrainer

# Written once to output_dir/RUN_CONFIG_FILENAME at the start of training --
# use_fusion/fusion_type/collator/wn_seed aren't recoverable from a saved
# checkpoint's config.json (collator especially: it's a data-loading choice,
# not a model-architecture field, so there's nowhere else for it to live).
# tests/eval_checkpoint.py looks for this file next to a --checkpoint_dir
# (checkpoints are always direct children of output_dir) to auto-fill those
# flags instead of requiring them to be re-typed and kept in sync by hand.
RUN_CONFIG_FILENAME = "run_config.json"


def _manifest_paths(spec):
    paths = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        path, separator, possible_weight = item.rpartition(":")
        if separator:
            try:
                float(possible_weight)
                item = path
            except ValueError:
                pass
        paths.append(item)
    return paths


def validate_action_manifest_metadata(args):
    if not args.action_training:
        return
    label_checkpoint = args.action_label_checkpoint.strip()
    initialization_checkpoint = args.merge_lora_into_base_from.strip()
    if not label_checkpoint:
        raise ValueError("--action_label_checkpoint is required for action training")
    if os.path.normpath(label_checkpoint) != os.path.normpath(initialization_checkpoint):
        raise ValueError(
            "Action labels and model initialization must use the same production "
            f"checkpoint, got labels={label_checkpoint!r}, "
            f"initialization={initialization_checkpoint!r}"
        )
    for manifest in _manifest_paths(args.train_file) + _manifest_paths(args.eval_file):
        metadata_path = manifest + ".meta.json"
        if not os.path.isfile(metadata_path):
            raise ValueError(f"Missing action-manifest metadata sidecar: {metadata_path}")
        with open(metadata_path) as handle:
            metadata = json.load(handle)
        if getattr(args, "action_format", None) == action_format.ACTION_JUDGE_V1:
            from judge_data import validate_judge_manifest
            validate_judge_manifest(manifest, label_checkpoint, args.action_wer_threshold)
        if not math.isclose(
            float(metadata.get("wer_threshold", float("nan"))),
            args.action_wer_threshold,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Threshold mismatch in {metadata_path}: "
                f"manifest={metadata.get('wer_threshold')}, "
                f"requested={args.action_wer_threshold}"
            )
        if os.path.normpath(str(metadata.get("checkpoint_id", ""))) != os.path.normpath(label_checkpoint):
            raise ValueError(
                f"Label-checkpoint mismatch in {metadata_path}: "
                f"manifest={metadata.get('checkpoint_id')!r}, "
                f"requested={label_checkpoint!r}"
            )


def validate_action_format_args(args):
    if not action_format.is_special(args.action_format):
        return
    if not args.action_training:
        raise ValueError("--action_format action_special_v1 requires --action_training 1")
    if not args.ft_params.strip():
        raise ValueError("--action_format action_special_v1 requires --ft_params "
                         "(the legacy --use_lora path is not supported)")
    llm_mode = json.loads(args.ft_params).get("llm", {}).get("ft_mode")
    if llm_mode != "lora":
        raise ValueError(
            "--action_format action_special_v1 requires ft_params llm.ft_mode='lora': the "
            "action rows are trained over frozen original vocabulary matrices, and "
            f"checkpoints are PEFT adapters (got llm.ft_mode={llm_mode!r})"
        )
    if args.action_token_lr <= 0:
        raise ValueError("--action_token_lr must be positive")
    if args.action_format == action_format.ACTION_JUDGE_V1:
        recipe = json.loads(args.ft_params)
        if any(recipe.get(r, {}).get("ft_mode") != "freeze" for r in ("encoder", "fusion", "aligner")):
            raise ValueError("Judge training requires frozen encoder, fusion and aligner")
        if not args.merge_lora_into_base_from.strip():
            raise ValueError("Judge training requires a production checkpoint")
        if args.quality_feature_mode != "none" or not args.use_fusion or args.collator != "action_mix":
            raise ValueError("Judge v1 requires action_mix fusion inputs and no quality hints")


def build_training_args(args, use_bf16: bool):
    report_to = [] if args.report_to.lower() in ["", "none"] else [args.report_to]

    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=bool(args.pin_memory),
        dataloader_persistent_workers=bool(args.persistent_workers),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        do_eval=bool(args.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=report_to,
        run_name="Mega-ASR-A2S-SFT",
    )


def main():
    args = parse_args()
    validate_action_manifest_metadata(args)
    validate_action_format_args(args)
    special_tokens = action_format.is_special(args.action_format)

    model, processor, use_bf16 = load_qwen3_asr(
        args.model_path, use_fusion=bool(args.use_fusion), fusion_type=args.fusion_type
    )

    if args.padding_side != "auto":
        processor.tokenizer.padding_side = args.padding_side
    if args.action_format == action_format.ACTION_JUDGE_V1:
        processor.tokenizer.padding_side = "left"
    print("padding_side =", processor.tokenizer.padding_side)

    if args.ft_params.strip():
        production_adapter = args.merge_lora_into_base_from.strip()
        if production_adapter:
            model = merge_lora_checkpoint(model, production_adapter)
        ft_params = parse_ft_params(args.ft_params)
        if special_tokens:
            # Register protocol tokens after the production merge and before LoRA.
            # Checkpoint resume verifies the saved token mapping and dimensions.
            setup_action_vocabulary(model, processor)
        model = apply_train_mode(
            model, ft_params, fusion_type=args.fusion_type, use_fusion=bool(args.use_fusion),
            lora_dropout=args.lora_dropout, lora_bias=args.lora_bias,
        )
        if special_tokens:
            enable_action_row_training(model)
        lr_encoder = ft_params["encoder"]["lr"]
        lr_aligner = ft_params["aligner"]["lr"]
        lr_llm = ft_params["llm"]["lr"]
        lr_fusion = ft_params["fusion"]["lr"]
    else:
        model = apply_lora(model, args)
        lr_encoder, lr_aligner, lr_llm = args.lr_encoder, args.lr_aligner, args.lr_llm
        # Preserves this codebase's original behavior (fusion params ride on
        # the encoder LR) unless explicitly overridden.
        lr_fusion = args.lr_fusion if args.lr_fusion is not None else args.lr_encoder

    dataset = build_datasets(args.train_file, args.eval_file)
    action_reject_weight = args.action_reject_weight
    if args.action_training and action_reject_weight == 0:
        action_counts = Counter(str(value).upper() for value in dataset["train"]["action"])
        if not action_counts["RESPOND"] or not action_counts["REJECT"]:
            raise ValueError(
                "Automatic action class weighting requires both RESPOND and REJECT "
                f"examples, got {dict(action_counts)}"
            )
        action_reject_weight = action_counts["RESPOND"] / action_counts["REJECT"]
        print(
            f"[action] class counts={dict(action_counts)}; automatic "
            f"REJECT weight={action_reject_weight:.6f}"
        )
    if action_reject_weight <= 0:
        raise ValueError("--action_reject_weight must be positive, or exactly 0 for auto")
    if args.collator == "auto":
        prefix = "action_" if args.action_training else ""
        collator_name = prefix + ("mix" if args.use_fusion else "none")
    else:
        collator_name = args.collator
    if bool(args.action_training) != collator_name.startswith("action_"):
        raise ValueError(
            f"--action_training={args.action_training} is incompatible with "
            f"--collator={collator_name!r}; use an action_* collator exactly when "
            "action training is enabled."
        )
    collator = build_collator(
        collator_name,
        use_fusion=bool(args.use_fusion),
        processor=processor,
        sampling_rate=args.sr,
        language=args.language,
        seed=args.wn_seed,
        quality_feature_mode=args.quality_feature_mode,
        reject_weight=action_reject_weight,
        action_format=args.action_format,
    )
    training_args = build_training_args(args, use_bf16)

    os.makedirs(args.output_dir, exist_ok=True)
    run_config = {
        "use_fusion": bool(args.use_fusion),
        "fusion_type": args.fusion_type,
        "collator": collator_name,
        "wn_seed": args.wn_seed,
        "language": args.language,
        "sr": args.sr,
        "action_training": bool(args.action_training),
        "action_loss_weight": args.action_loss_weight,
        "action_reject_weight": action_reject_weight,
        "action_reject_weight_requested": args.action_reject_weight,
        "action_wer_threshold": args.action_wer_threshold,
        "action_label_checkpoint": args.action_label_checkpoint.strip() or None,
        "action_format": args.action_format,
        "action_token_lr": args.action_token_lr if special_tokens else None,
        "quality_feature_mode": args.quality_feature_mode,
        # eval_checkpoint.py's _load_model_for_eval needs this for any
        # checkpoint that's a PEFT adapter (i.e. any ft_params region was
        # "lora") -- an adapter-only checkpoint has no base weights of its
        # own, so it has to know which base to load and apply itself on top
        # of. Also written by trainer.py's save_model as base_model.txt
        # (a plain-text fallback for old checkpoints saved before this field
        # existed here); this JSON copy is what eval_checkpoint.py actually
        # reads.
        "model_path": args.model_path,
        "merged_from_lora_path": args.merge_lora_into_base_from.strip() or None,
        # Not recoverable from a saved checkpoint's config.json (it's a
        # training-recipe choice, not a model-architecture field) --
        # keeping it here means a future "resume with the same per-region
        # setup" or "audit what a checkpoint was actually trained with"
        # doesn't have to reconstruct it from the run's shell history.
        "ft_params": ft_params if args.ft_params.strip() else None,
    }
    run_config_path = os.path.join(args.output_dir, RUN_CONFIG_FILENAME)
    if training_args.should_save:
        with open(run_config_path, "w") as f:
            json.dump(run_config, f, indent=2)
    print(f"[run_config] wrote {run_config_path}: {run_config}")

    action_metadata = None
    if special_tokens:
        action_metadata = {
            "run_config": run_config,
            "loss": {
                "objective": "L_transcript + action_loss_weight * L_action",
                "transcript_loss": "mean CE over transcript+EOS in the transcript pass "
                                   "([prompt] -> transcript<EOS>, no action token)",
                "action_loss": "class-weighted CE at the <|action_query|> position of a "
                               "separate action pass ([prompt] <|action_query|>), over the "
                               "full expanded vocabulary",
                "action_loss_weight": args.action_loss_weight,
                "action_reject_weight": action_reject_weight,
            },
            "warm_start": "production checkpoint (fresh decoder LoRA and action rows)",
        }

    if args.action_format == action_format.ACTION_JUDGE_V1:
        from judge_data import validate_judge_manifest
        action_metadata["loss"] = {"objective": "class-weighted action CE over the full vocabulary",
                                   "transcription_loss": False, "action_reject_weight": action_reject_weight}
        action_metadata["candidate_manifests"] = {
            p: validate_judge_manifest(p, args.action_label_checkpoint, args.action_wer_threshold)
            for p in _manifest_paths(args.train_file) + _manifest_paths(args.eval_file)
        }

    trainer = MegaASRTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        data_collator=collator,
        processing_class=processor,
        callbacks=[MakeCheckpointInferableCallback(args.model_path, preserve_processor=special_tokens)],
        processor=processor,
        base_model_path=args.model_path,
        merged_from_lora_path=args.merge_lora_into_base_from.strip(),
        lr_encoder=lr_encoder,
        lr_aligner=lr_aligner,
        lr_llm=lr_llm,
        lr_fusion=lr_fusion,
        fusion_type=args.fusion_type,
        use_fusion=bool(args.use_fusion),
        action_loss_weight=args.action_loss_weight,
        action_format_name=args.action_format,
        action_token_lr=args.action_token_lr,
        action_metadata=action_metadata,
    )

    resume_from = args.resume_from.strip()
    if not resume_from and args.resume:
        resume_from = find_latest_checkpoint(args.output_dir) or ""

    if resume_from:
        print(f"[resume] {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    if args.action_training:
        final_dir = os.path.join(args.output_dir, "final")
        trainer.save_model(final_dir)
        print(f"[action] saved final adapter to {final_dir}")


if __name__ == "__main__":
    main()
