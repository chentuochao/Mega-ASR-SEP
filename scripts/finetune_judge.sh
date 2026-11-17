#!/bin/bash
# Train decoder LoRA and protocol-token rows from prepared judge manifests.
set -euo pipefail
: "${BASE_MODEL_PATH:?Set BASE_MODEL_PATH}"
: "${PRODUCTION_CHECKPOINT:?Set PRODUCTION_CHECKPOINT}"
: "${TRAIN_JSONL:?Set TRAIN_JSONL to a prepared judge manifest}"
: "${VAL_JSONL:?Set VAL_JSONL to a prepared judge manifest}"
: "${OUT_DIR:?Set a new judge output directory}"
cd "$(dirname "${BASH_SOURCE[0]}")/../src/MegaASR"
mkdir -p "$OUT_DIR"
args=()
if [[ -n "${RESUME_FROM:-}" ]]; then args+=(--resume_from "$RESUME_FROM"); fi
torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" --master_port="${MASTER_PORT:-29521}" A2S-SFT/finetune.py \
  --model_path "$BASE_MODEL_PATH" --merge_lora_into_base_from "$PRODUCTION_CHECKPOINT" \
  --train_file "$TRAIN_JSONL" --eval_file "$VAL_JSONL" --output_dir "$OUT_DIR" \
  --use_fusion 1 --fusion_type early_conv --collator action_mix \
  --action_training 1 --action_format action_judge_v1 \
  --action_label_checkpoint "$PRODUCTION_CHECKPOINT" --action_wer_threshold "${WER_THRESHOLD:-0.75}" \
  --action_token_lr "${ACTION_TOKEN_LR:-1e-5}" --action_reject_weight "${ACTION_REJECT_WEIGHT:-0}" \
  --quality_feature_mode none --padding_side left --language English \
  --ft_params '{"encoder":{"ft_mode":"freeze","lr":0},"fusion":{"ft_mode":"freeze","lr":0},"aligner":{"ft_mode":"freeze","lr":0},"llm":{"ft_mode":"lora","lr":1e-5,"lora_r":8,"lora_alpha":16}}' \
  --batch_size "${BATCH_SIZE:-2}" --grad_acc "${GRAD_ACC:-1}" --epochs "${EPOCHS:-20}" \
  --save_steps "${SAVE_STEPS:-40}" --save_total_limit 5 --warmup_ratio 0.03 --weight_decay 0.01 \
  --report_to none "${args[@]}" 2>&1 | tee -a "$OUT_DIR/train.log"
