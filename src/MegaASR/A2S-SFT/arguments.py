# coding=utf-8
import argparse


def parse_args():
    p = argparse.ArgumentParser("Mega-ASR A2S-SFT")

    # paths
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="outputs/a2s_sft")

    # data
    p.add_argument("--sr", type=int, default=16000)
    # "right" (the tokenizer's own saved default, and what "auto" resolves to
    # for this checkpoint) silently corrupts the forward pass for any batch
    # with variable-length sequences: solo vs batched logits diverge
    # completely (0% argmax match, loss ~13 vs ~1.4), almost certainly because
    # the model's 3D-RoPE position-id computation assumes left-padded (or
    # unpadded) input. Default to "left", which was verified to fix it.
    p.add_argument("--padding_side", type=str, default="left",
                   choices=["auto", "left", "right"])

    # training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--lr_aligner", type=float, default=1e-5)
    p.add_argument("--lr_llm", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--report_to", type=str, default="none")

    # dataloader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # save / resume
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--resume_from", type=str, default="")

    # fusion (mixture-conditioned) — see PLAN.md
    p.add_argument("--use_fusion", type=int, default=0,
                   help="Fuse the pre-separation mixture stream into the audio "
                        "tower. Requires 'audio_mix' in the dataset. "
                        "0 => single-stream baseline (upstream model).")
    p.add_argument("--fusion_type", type=str, default="late_gate",
                   choices=["late_gate", "early_conv"],
                   help="late_gate: gated residual on the tower OUTPUT (2 tower "
                        "passes). early_conv: parallel conv on the mixture mel "
                        "INPUT (1 tower pass). Ignored if --use_fusion 0.")

    # full-parameter fine-tuning (only used when --use_lora 0)
    p.add_argument("--freeze_llm", type=int, default=0,
                   help="When --use_lora 0: freeze the LLM decoder + lm_head "
                        "('model.*'/'lm_head.*') and fully fine-tune everything "
                        "else (audio_tower -- encoder, aligner, and fusion params "
                        "if --use_fusion 1). Ignored when --use_lora 1; use "
                        "--lora_scope llm/encoder/etc. instead.")

    # lora
    p.add_argument("--use_lora", type=int, default=1)
    p.add_argument("--lora_scope", type=str, default="encoder_aligner",
                   choices=["encoder", "aligner", "encoder_aligner",
                            "encoder_b4_aligner", "llm", "all"])
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--merge_lora_into_base_from", type=str, default="")

    return p.parse_args()
