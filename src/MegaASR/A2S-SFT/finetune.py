# coding=utf-8
import json
import os

from transformers import TrainingArguments

from arguments import parse_args
from checkpointing import MakeCheckpointInferableCallback, find_latest_checkpoint
from dataloader import build_collator, build_datasets
from modeling import apply_lora, apply_train_mode, load_qwen3_asr, parse_ft_params
from trainer import MegaASRTrainer

# Written once to output_dir/RUN_CONFIG_FILENAME at the start of training --
# use_fusion/fusion_type/collator/wn_seed aren't recoverable from a saved
# checkpoint's config.json (collator especially: it's a data-loading choice,
# not a model-architecture field, so there's nowhere else for it to live).
# tests/eval_checkpoint.py looks for this file next to a --checkpoint_dir
# (checkpoints are always direct children of output_dir) to auto-fill those
# flags instead of requiring them to be re-typed and kept in sync by hand.
RUN_CONFIG_FILENAME = "run_config.json"


def build_training_args(args, use_bf16: bool):
    report_to = [] if args.report_to.lower() in ["", "none"] else [args.report_to]

    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
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

    model, processor, use_bf16 = load_qwen3_asr(
        args.model_path, use_fusion=bool(args.use_fusion), fusion_type=args.fusion_type
    )

    if args.padding_side != "auto":
        processor.tokenizer.padding_side = args.padding_side
    print("padding_side =", processor.tokenizer.padding_side)

    if args.ft_params.strip():
        ft_params = parse_ft_params(args.ft_params)
        model = apply_train_mode(
            model, ft_params, fusion_type=args.fusion_type, use_fusion=bool(args.use_fusion),
            lora_dropout=args.lora_dropout, lora_bias=args.lora_bias,
        )
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
    collator_name = args.collator if args.collator != "auto" else ("mix" if args.use_fusion else "none")
    collator = build_collator(
        collator_name,
        use_fusion=bool(args.use_fusion),
        processor=processor,
        sampling_rate=args.sr,
        language=args.language,
        seed=args.wn_seed,
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
        # eval_checkpoint.py's _load_model_for_eval needs this for any
        # checkpoint that's a PEFT adapter (i.e. any ft_params region was
        # "lora") -- an adapter-only checkpoint has no base weights of its
        # own, so it has to know which base to load and apply itself on top
        # of. Also written by trainer.py's save_model as base_model.txt
        # (a plain-text fallback for old checkpoints saved before this field
        # existed here); this JSON copy is what eval_checkpoint.py actually
        # reads.
        "model_path": args.model_path,
        # Not recoverable from a saved checkpoint's config.json (it's a
        # training-recipe choice, not a model-architecture field) --
        # keeping it here means a future "resume with the same per-region
        # setup" or "audit what a checkpoint was actually trained with"
        # doesn't have to reconstruct it from the run's shell history.
        "ft_params": ft_params if args.ft_params.strip() else None,
    }
    run_config_path = os.path.join(args.output_dir, RUN_CONFIG_FILENAME)
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"[run_config] wrote {run_config_path}: {run_config}")

    trainer = MegaASRTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        data_collator=collator,
        processing_class=processor,
        callbacks=[MakeCheckpointInferableCallback(args.model_path)],
        processor=processor,
        base_model_path=args.model_path,
        merged_from_lora_path=args.merge_lora_into_base_from.strip(),
        lr_encoder=lr_encoder,
        lr_aligner=lr_aligner,
        lr_llm=lr_llm,
        lr_fusion=lr_fusion,
        fusion_type=args.fusion_type,
        use_fusion=bool(args.use_fusion),
    )

    resume_from = args.resume_from.strip()
    if not resume_from and args.resume:
        resume_from = find_latest_checkpoint(args.output_dir) or ""

    if resume_from:
        print(f"[resume] {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
