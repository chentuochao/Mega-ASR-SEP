# coding=utf-8
from transformers import TrainingArguments

from .arguments import parse_args
from .checkpointing import MakeEveryCheckpointInferableCallback, find_latest_checkpoint
from .data import DataCollatorForQwen3ASRFinetuning, build_datasets
from .modeling import load_qwen3_asr_model_and_processor, maybe_apply_lora
from .sampler import SamplerEpochCallback
from .trainer import (
    CastFloatInputsTrainer,
    FixedRatioCastTrainer,
    FixedRatioMultiLRAdapterTrainer,
    MultiLRAdapterTrainer,
)


def build_training_args(args_cli, use_bf16: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="wandb",
        weight_decay=args_cli.weight_decay,
        run_name=args_cli.run_name,
        max_grad_norm=args_cli.max_grad_norm,
    )


def build_trainer(args_cli, model, processor, ds, collator, training_args):
    common_trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        processing_class=processor,
        callbacks=[
            MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path),
            SamplerEpochCallback(),
        ],
    )

    if args_cli.use_lora == 1 and args_cli.save_adapter_only == 1:
        if args_cli.use_fixed_ratio_sampler == 1:
            trainer = FixedRatioMultiLRAdapterTrainer(
                **common_trainer_kwargs,
                processor=processor,
                base_model_path=args_cli.model_path,
                merged_from_lora_path=(args_cli.merge_lora_into_base_from or "").strip(),
                lr_tower=args_cli.lr_tower,
                lr_proj=args_cli.lr_proj,
                lr_llm=args_cli.lr_llm,
                mix_target_ratio=args_cli.mix_target_ratio,
                mix_domain_field=args_cli.mix_domain_field,
                mix_target_value=args_cli.mix_target_value,
            )
        else:
            trainer = MultiLRAdapterTrainer(
                **common_trainer_kwargs,
                processor=processor,
                base_model_path=args_cli.model_path,
                merged_from_lora_path=(args_cli.merge_lora_into_base_from or "").strip(),
                lr_tower=args_cli.lr_tower,
                lr_proj=args_cli.lr_proj,
                lr_llm=args_cli.lr_llm,
            )
    else:
        if args_cli.use_fixed_ratio_sampler == 1:
            trainer = FixedRatioCastTrainer(
                **common_trainer_kwargs,
                mix_target_ratio=args_cli.mix_target_ratio,
                mix_domain_field=args_cli.mix_domain_field,
                mix_target_value=args_cli.mix_target_value,
            )
        else:
            trainer = CastFloatInputsTrainer(
                **common_trainer_kwargs,
            )

    return trainer


def main():
    args_cli = parse_args()

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    model, processor, use_bf16 = load_qwen3_asr_model_and_processor(args_cli.model_path)
    model = maybe_apply_lora(model, args_cli)

    ds = build_datasets(args_cli.train_file, args_cli.eval_file, processor)
    collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=args_cli.sr,
    )

    training_args = build_training_args(args_cli, use_bf16)
    trainer = build_trainer(args_cli, model, processor, ds, collator, training_args)

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
