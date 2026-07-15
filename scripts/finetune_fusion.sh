#!/bin/bash
set -euo pipefail

# Mega-ASR-SEP fine-tuning: no-fusion baseline, late fusion, or early fusion.
# Select with FUSION_MODE, e.g.:
#   FUSION_MODE=none  TRAIN_JSONL=... VAL_JSONL=... OUT_DIR=... LOG_FILE=... bash scripts/finetune_fusion.sh
#   FUSION_MODE=late  TRAIN_JSONL=... VAL_JSONL=... OUT_DIR=... LOG_FILE=... bash scripts/finetune_fusion.sh
#   FUSION_MODE=early TRAIN_JSONL=... VAL_JSONL=... OUT_DIR=... LOG_FILE=... bash scripts/finetune_fusion.sh
#
#   none  -> original model, separated audio only. Dataset rows need
#            {"audio": "sep.wav", "text": "..."} (or "audio_sep").
#   late  -> gated residual on the audio-tower OUTPUT (2 tower passes).
#   early -> parallel conv on the mixture mel INPUT (1 tower pass).
#   late/early both require EVERY dataset row to carry BOTH:
#            {"audio_sep": "sep.wav", "audio_mix": "mix.wav", "text": "..."}
#            (dataloader.py's Qwen3ASRCollatorMix now hard-asserts on any row
#            missing "audio_mix" -- it fails the batch loudly instead of
#            silently falling back to single-stream)
#
# Data collator is picked automatically from FUSION_MODE via --collator auto
# (see dataloader.py's COLLATORS/build_collator): none -> "none", late/early
# -> "mix". Override with COLLATOR=<name> to use a different collator with a
# fusion-enabled run, e.g. the white_noise_test diagnostic probe (both audio
# streams synthesized from ONE clean "libritts_path" reference, corrupted
# with disjoint extreme-white-noise chunks per stream, to test whether
# fusion can combine two complementary streams in isolation from real
# separation-quality issues):
#   FUSION_MODE=late COLLATOR=white_noise_test ... bash scripts/finetune_fusion.sh
# build_collator() rejects any FUSION_MODE/COLLATOR combo that mismatches
# use_fusion (e.g. COLLATOR=white_noise_test with FUSION_MODE=none) at
# startup, so a bad combination fails fast instead of silently misconfiguring.

# # wandb
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_API_KEY="" # your wandb key
# export WANDB_PROJECT=""
# export WANDB_ENTITY=""
# export WANDB_MODE=online

FUSION_MODE="${FUSION_MODE:-none}"        # none | late | early
COLLATOR="${COLLATOR:-auto}"              # auto | none | mix | white_noise_test
                                           # (auto derives from FUSION_MODE; see header comment)
MODEL_PATH="${MODEL_PATH:-Qwen3-ASR-1.7B}" # absolute path recommended -- the bare
                                            # default is relative and only resolves
                                            # if CWD happens to contain it
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"      # GPUs to use; set 1 for a quick sanity run
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-8}"
EPOCHS="${EPOCHS:-2}"
SAVE_STEPS="${SAVE_STEPS:-200}"
REPORT_TO="${REPORT_TO:-none}"            # set to "none" to disable wandb
LANGUAGE="${LANGUAGE:-English}"           # forced language tag baked into the training
                                           # prefix (see dataloader.py's Qwen3ASRCollator);
                                           # set "" to disable forcing (auto language-ID)

# Data path will be replaced according to your actual requirements. All of
# these can still be overridden via env var (e.g. TRAIN_JSONL=... bash
# finetune_fusion.sh) -- RUN_NAME folds in FUSION_MODE by default so
# none/late/early runs never collide in the same OUT_DIR.
RUN_DIR="/home/ubuntu/Hearvana/datasets/Results_ASR"
DATA_BASE="/home/ubuntu/Hearvana/datasets/Mix_Qwen_ASR_dataset"

RUN_NAME="${RUN_NAME:-debug}_${FUSION_MODE}"
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
echo "[finetune_fusion] FUSION_MODE=${FUSION_MODE} -> ${FUSION_ARGS[*]}  COLLATOR=${COLLATOR}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" A2S-SFT/finetune.py \
  --model_path "${MODEL_PATH}" \
  --train_file "${TRAIN_JSONL}" \
  --eval_file "${VAL_JSONL}" \
  --output_dir "${OUT_DIR}" \
  "${FUSION_ARGS[@]}" \
  --collator "${COLLATOR}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_acc "${GRAD_ACC}" \
  --lr 1e-6 \
  --lr_encoder 1e-6 \
  --lr_aligner 1e-6 \
  --lr_llm 1e-6 \
  --language "${LANGUAGE}" \
  --epochs "${EPOCHS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 300 \
  --use_lora 1 \
  --lora_scope encoder_aligner \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --report_to "${REPORT_TO}" \
  2>&1 | tee -a "${LOG_FILE}"
