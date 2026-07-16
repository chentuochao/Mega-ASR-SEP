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
#   FUSION_MODE=late COLLATOR=white_noise_test ... bash scripts/finetune_encoder.sh
# build_collator() rejects any FUSION_MODE/COLLATOR combo that mismatches
# use_fusion (e.g. COLLATOR=white_noise_test with FUSION_MODE=none) at
# startup, so a bad combination fails fast instead of silently misconfiguring.
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
COLLATOR="${COLLATOR:-auto}"              # auto | none | mix | white_noise_test
                                           # (auto derives from FUSION_MODE; see header comment)
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"      # GPUs to use; set 1 for a quick sanity run
MASTER_PORT="${MASTER_PORT:-29500}"        # set a distinct port per job to run
                                            # multiple torchrun jobs on one machine
                                            # concurrently (e.g. different GPU sets)
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-8}"
EPOCHS="${EPOCHS:-4}"
SAVE_STEPS="${SAVE_STEPS:-200}"
REPORT_TO="${REPORT_TO:-none}"            # set to "none" to disable wandb
LR_ENCODER="${LR_ENCODER:-1e-5}"          # full-FT encoder transformer layers LR
LR_ALIGNER="${LR_ALIGNER:-1e-5}"          # full-FT aligner (conv_out/proj1/proj2) LR
LANGUAGE="${LANGUAGE:-English}"           # forced language tag baked into the training
                                           # prefix (see dataloader.py's Qwen3ASRCollator);
                                           # set "" to disable forcing (auto language-ID)

# Data path will be replaced according to your actual requirements. All of
# these can still be overridden via env var (e.g. TRAIN_JSONL=... bash
# finetune_encoder.sh) -- RUN_NAME folds in FUSION_MODE by default so
# none/late/early runs never collide in the same OUT_DIR, and defaults to a
# different prefix than finetune_fusion.sh's LoRA runs so the two don't clash.
RUN_NAME="cleanfix"
MODEL_PATH="/home/ubuntu/Hearvana/Scripts/Mega-ASR/ckpt/Mega-ASR/Qwen3-ASR-1.7B"
RUN_DIR="/home/ubuntu/Hearvana/datasets/Results_ASR"
DATA_BASE="/home/ubuntu/Hearvana/datasets/Mix_Qwen_ASR_dataset"

RUN_NAME="${RUN_NAME:-encoder_fullft}_${FUSION_MODE}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_BASE}/train_clean.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_BASE}/val_clean.jsonl}"
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
echo "[finetune_encoder] FUSION_MODE=${FUSION_MODE} -> ${FUSION_ARGS[*]}  COLLATOR=${COLLATOR}  (full-FT encoder, LLM frozen)"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" A2S-SFT/finetune.py \
  --model_path "${MODEL_PATH}" \
  --train_file "${TRAIN_JSONL}" \
  --eval_file "${VAL_JSONL}" \
  --output_dir "${OUT_DIR}" \
  "${FUSION_ARGS[@]}" \
  --collator "${COLLATOR}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_acc "${GRAD_ACC}" \
  --lr_encoder "${LR_ENCODER}" \
  --lr_aligner "${LR_ALIGNER}" \
  --language "${LANGUAGE}" \
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
