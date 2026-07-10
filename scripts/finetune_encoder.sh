#!/bin/bash
set -euo pipefail

# torchrun below launches "A2S-SFT/finetune.py" as a relative path, so this
# script must run with CWD = src/MegaASR regardless of where/how it's invoked
# (e.g. `bash ./scripts/finetune_encoder.sh` from the repo root).
cd "$(dirname "${BASH_SOURCE[0]}")/../src/MegaASR"

# Mega-ASR-SEP fine-tuning: FULL-PARAMETER fine-tune of the audio encoder
# (audio_tower -- transformer layers, aligner, stem convs, and fusion params
# if FUSION_MODE != none), with the LLM decoder frozen (--use_lora 0
# --freeze_llm 1). No LoRA involved. Select fusion mechanism with FUSION_MODE,
# same three options as finetune_fusion.sh:
#   FUSION_MODE=none  ... bash scripts/finetune_encoder.sh
#   FUSION_MODE=late  ... bash scripts/finetune_encoder.sh
#   FUSION_MODE=early ... bash scripts/finetune_encoder.sh
#
#   none  -> original model, separated audio only. Dataset rows need
#            {"audio": "sep.wav", "text": "..."} (or "audio_sep").
#   late  -> gated residual on the audio-tower OUTPUT (2 tower passes).
#   early -> parallel conv on the mixture mel INPUT (1 tower pass).
#   late/early both require EVERY dataset row to carry BOTH:
#            {"audio_sep": "sep.wav", "audio_mix": "mix.wav", "text": "..."}
#            (a batch with any row missing "audio_mix" silently falls back to
#            single-stream for that batch — see dataloader.py's Qwen3ASRCollator)
#
# Frozen: model.* (LLM decoder + embed_tokens) and lm_head.*
# Trained (full parameter, not LoRA): audio_tower.* -- encoder transformer
# layers, aligner (conv_out/proj1/proj2), stem convs, ln_post, and
# fusion_gate/conv2d1_mix if fusion is enabled. See modeling.py's apply_lora
# and trainer.py's _group_name for exactly how this is wired.

# # wandb
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_API_KEY="" # your wandb key
# export WANDB_PROJECT=""
# export WANDB_ENTITY=""
# export WANDB_MODE=online

FUSION_MODE="${FUSION_MODE:-none}"        # none | late | early
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"      # GPUs to use; set 1 for a quick sanity run
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-8}"
EPOCHS="${EPOCHS:-2}"
SAVE_STEPS="${SAVE_STEPS:-200}"
REPORT_TO="${REPORT_TO:-none}"            # set to "none" to disable wandb
LR_ENCODER="${LR_ENCODER:-1e-6}"          # full-FT encoder transformer layers LR
LR_ALIGNER="${LR_ALIGNER:-1e-6}"          # full-FT aligner (conv_out/proj1/proj2) LR

# Data path will be replaced according to your actual requirements. All of
# these can still be overridden via env var (e.g. TRAIN_JSONL=... bash
# finetune_encoder.sh) -- RUN_NAME folds in FUSION_MODE by default so
# none/late/early runs never collide in the same OUT_DIR, and defaults to a
# different prefix than finetune_fusion.sh's LoRA runs so the two don't clash.
RUN_NAME="debug"
MODEL_PATH="/home/ubuntu/Hearvana/Scripts/Mega-ASR/ckpt/Mega-ASR/Qwen3-ASR-1.7B"
RUN_DIR="/home/ubuntu/Hearvana/datasets/Results_ASR"
DATA_BASE="/home/ubuntu/Hearvana/datasets/Mix_Qwen_ASR_dataset"

RUN_NAME="${RUN_NAME:-encoder_fullft}_${FUSION_MODE}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_BASE}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_BASE}/val.jsonl}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/train.log}"
mkdir -p "${OUT_DIR}"

case "$FUSION_MODE" in
  none)
    FUSION_ARGS=(--use_fusion 0)
    ;;
  late)
    FUSION_ARGS=(--use_fusion 1 --fusion_type late_gate)
    ;;
  early)
    FUSION_ARGS=(--use_fusion 1 --fusion_type early_conv)
    ;;
  *)
    echo "FUSION_MODE must be 'none', 'late', or 'early' (got '$FUSION_MODE')" >&2
    exit 1
    ;;
esac
echo "[finetune_encoder] FUSION_MODE=${FUSION_MODE} -> ${FUSION_ARGS[*]}  (full-FT encoder, LLM frozen)"

torchrun --nproc_per_node="${NPROC_PER_NODE}" A2S-SFT/finetune.py \
  --model_path "${MODEL_PATH}" \
  --train_file "${TRAIN_JSONL}" \
  --eval_file "${VAL_JSONL}" \
  --output_dir "${OUT_DIR}" \
  "${FUSION_ARGS[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_acc "${GRAD_ACC}" \
  --lr_encoder "${LR_ENCODER}" \
  --lr_aligner "${LR_ALIGNER}" \
  --epochs "${EPOCHS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 300 \
  --use_lora 0 \
  --freeze_llm 1 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --report_to "${REPORT_TO}" \
  2>&1 | tee -a "${LOG_FILE}"
