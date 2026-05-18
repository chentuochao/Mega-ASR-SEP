# coding=utf-8
import argparse


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    # 不同模块不同学习率
    p.add_argument("--lr_tower", type=float, default=1e-4)
    p.add_argument("--lr_proj", type=float, default=3e-4)
    p.add_argument("--lr_llm", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    # LoRA
    p.add_argument("--use_lora", type=int, default=1)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--run_name", type=str, default="qwen3-asr-lora")

    # Fixed-ratio sampler
    p.add_argument("--use_fixed_ratio_sampler", type=int, default=1)
    p.add_argument("--mix_target_ratio", type=float, default=0.3)
    p.add_argument("--mix_domain_field", type=str, default="domain")
    p.add_argument("--mix_target_value", type=str, default="targeted")

    # 选择 LoRA 范围：tower / proj / tower_proj / towerb4_proj / llm / both
    p.add_argument(
        "--lora_scope",
        type=str,
        default="both",
        choices=["tower", "proj", "tower_proj", "towerb4_proj", "llm", "both"],
    )

    # 每个 checkpoint 只保存 adapter
    p.add_argument("--save_adapter_only", type=int, default=1)
    # 重新加载 checkpoint，归零训
    p.add_argument("--merge_lora_into_base_from", type=str, default="")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    return p.parse_args()
