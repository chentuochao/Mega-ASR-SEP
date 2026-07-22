#!/bin/bash
set -euo pipefail

# Resolved once, before the `cd` below changes CWD -- both CONFIG resolution
# and locating load_config.py need this to still work after that `cd`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# bash scripts/finetune_new.sh path/to.yaml  (or CONFIG=path/to.yaml bash
# scripts/finetune_new.sh) loads per-run settings from a YAML file -- see
# scripts/configs/example_ft_params.yaml for a working example. Every
# top-level YAML key maps to the same-named env var below (key.upper()), and
# any env var already set when this script runs takes priority over the
# config file, which in turn takes priority over this script's own defaults
# (scripts/load_config.py). Resolved to an absolute path here, before the `cd`
# below, so a config path relative to the invocation directory still resolves
# correctly afterward.
CONFIG="${CONFIG:-${1:-}}"
if [[ -n "$CONFIG" ]]; then
  CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
fi

# torchrun below launches "A2S-SFT/finetune.py" as a relative path, so this
# script must run with CWD = src/MegaASR regardless of where/how it's invoked
# (e.g. `bash ./scripts/finetune_new.sh` from the repo root).
cd "${SCRIPT_DIR}/../src/MegaASR"

if [[ -n "$CONFIG" ]]; then
  CONFIG_EXPORTS="$(python3 "${SCRIPT_DIR}/load_config.py" "$CONFIG")"
  eval "$CONFIG_EXPORTS"
fi

# Mega-ASR-SEP fine-tuning: THE single script for every training strategy --
# supersedes finetune_encoder.sh (always full-FT encoder, LLM frozen) and
# finetune_fusion.sh (always LoRA) with one mechanism that expresses both of
# those AND anything in between: FT_PARAMS independently sets each of the
# four regions {encoder, fusion, aligner, llm} to freeze / lora / full, each
# with its own lr (and lora_r/lora_alpha when that region is lora) -- see
# arguments.py's --ft_params and modeling.py's parse_ft_params/
# apply_train_mode. FT_PARAMS is REQUIRED here (this script's whole point is
# driving training through it); set it via a config file's nested
# "ft_params:" block (see scripts/configs/example_ft_params.yaml) --
# load_config.py JSON-serializes that block into the FT_PARAMS env var since
# it's one coherent nested unit, not an independent scalar knob like the
# other config keys.
#
# Examples of what FT_PARAMS can express in one run (all via config, no code
# changes): today's finetune_encoder.sh recipe (encoder/fusion/aligner=full,
# llm=freeze); today's finetune_fusion.sh recipe (encoder/aligner=lora,
# fusion=full, llm=freeze); LoRA on the LLM while full-FT'ing the encoder;
# training ONLY the fusion gate with everything else frozen (a probe); etc.
#
# Select the fusion mechanism (orthogonal to FT_PARAMS -- which architecture
# vs. how to train it) with FUSION_MODE, same as the other two scripts:
#   FUSION_MODE=none       ... bash scripts/finetune_new.sh
#   FUSION_MODE=late       ... bash scripts/finetune_new.sh
#   FUSION_MODE=early      ... bash scripts/finetune_new.sh
#   FUSION_MODE=fddt       ... bash scripts/finetune_new.sh
#   FUSION_MODE=cross_attn ... bash scripts/finetune_new.sh
#
#   none       -> original model, separated audio only. Dataset rows need
#                 {"audio": "sep.wav", "text": "..."} (or "audio_sep").
#   late       -> gated residual on the audio-tower OUTPUT (2 tower passes).
#   early      -> parallel conv on the mixture mel INPUT (1 tower pass).
#   fddt       -> mix is primary (full tower pass), sep is a per-layer FiLM
#                 condition, DiCoW-style (1 tower pass + cheap conv-only
#                 side pass on sep).
#   cross_attn -> like late, but the gate is replaced by local windowed
#                 cross-attention between sep/mix tower outputs (2 tower
#                 passes).
#   late/early/fddt/cross_attn all require EVERY dataset row to carry BOTH:
#            {"audio_sep": "sep.wav", "audio_mix": "mix.wav", "text": "..."}
#            (dataloader.py's Qwen3ASRCollatorMix hard-asserts on any row
#            missing "audio_mix" -- it fails the batch loudly instead of
#            silently falling back to single-stream)
#
# Data collator is picked automatically from FUSION_MODE via --collator auto
# (see dataloader.py's COLLATORS/build_collator): none -> "none", anything
# else -> "mix". Override with COLLATOR=<name> to use a different collator
# with a fusion-enabled run, e.g. the white_noise_test diagnostic probe:
#   FUSION_MODE=late COLLATOR=white_noise_test ... bash scripts/finetune_new.sh
# build_collator() rejects any FUSION_MODE/COLLATOR combo that mismatches
# use_fusion at startup, so a bad combination fails fast instead of silently
# misconfiguring.

# # wandb
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_API_KEY="" # your wandb key
# export WANDB_PROJECT=""
# export WANDB_ENTITY=""
# export WANDB_MODE=online

FUSION_MODE="${FUSION_MODE:-none}"        # none | late | early | fddt | cross_attn
COLLATOR="${COLLATOR:-auto}"              # auto | none | mix | white_noise_test
                                           # (auto derives from FUSION_MODE; see header comment)
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"      # GPUs to use; set 1 for a quick sanity run
MASTER_PORT="${MASTER_PORT:-29500}"        # set a distinct port per job to run
                                            # multiple torchrun jobs on one machine
                                            # concurrently (e.g. different GPU sets)
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-8}"
EPOCHS="${EPOCHS:-3}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-300}"
REPORT_TO="${REPORT_TO:-none}"            # set to "none" to disable wandb
LANGUAGE="${LANGUAGE:-English}"           # forced language tag baked into the training
                                           # prefix (see dataloader.py's Qwen3ASRCollator);
                                           # set "" to disable forcing (auto language-ID)
LR="${LR:-1e-5}"                          # fallback LR for any param that doesn't fall into
                                           # encoder/fusion/aligner/llm (should be none, in
                                           # practice -- see trainer.py's optimizer grouping)
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"      # global across every region set to ft_mode: lora
                                           # (PEFT has no per-region override for this one --
                                           # only lora_r/lora_alpha are per-region, inside
                                           # FT_PARAMS itself)
LORA_BIAS="${LORA_BIAS:-none}"            # ditto, global across every lora region
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

# The one thing this script cannot default sensibly: which region does what.
# See scripts/configs/example_ft_params.yaml for the schema; every region in
# {encoder, fusion, aligner, llm} must be present.
FT_PARAMS="${FT_PARAMS:-}"
if [[ -z "$FT_PARAMS" ]]; then
  echo "FT_PARAMS is required (this script drives training entirely through" >&2
  echo "it -- see scripts/configs/example_ft_params.yaml). Set it via a" >&2
  echo "config file's 'ft_params:' block, or directly as an env var, e.g.:" >&2
  echo '  FT_PARAMS='"'"'{"encoder":{"ft_mode":"lora","lr":1e-4,"lora_r":16,"lora_alpha":32},"fusion":{"ft_mode":"full","lr":1e-5},"aligner":{"ft_mode":"full","lr":1e-5},"llm":{"ft_mode":"freeze","lr":1e-5}}'"'"' bash scripts/finetune_new.sh' >&2
  exit 1
fi
python3 -c "import json,sys; json.loads(sys.argv[1])" "$FT_PARAMS" 2>/dev/null || {
  echo "FT_PARAMS is not valid JSON: ${FT_PARAMS}" >&2
  exit 1
}

for var in RUN_NAME TRAIN_JSONL VAL_JSONL; do
  if [[ -z "${!var:-}" ]]; then
    echo "${var} is required -- set it via the config file (CONFIG=...) or as an env var." >&2
    exit 1
  fi
done

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
  fddt)
    FUSION_ARGS=(--use_fusion 1 --fusion_type fddt)
    ;;
  cross_attn)
    FUSION_ARGS=(--use_fusion 1 --fusion_type cross_attn)
    ;;
  *)
    echo "FUSION_MODE must be 'none', 'late', 'early', 'fddt', or 'cross_attn' (got '$FUSION_MODE')" >&2
    exit 1
    ;;
esac
echo "[finetune_new] FUSION_MODE=${FUSION_MODE} -> ${FUSION_ARGS[*]}  COLLATOR=${COLLATOR}"
echo "[finetune_new] FT_PARAMS=${FT_PARAMS}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" A2S-SFT/finetune.py \
  --model_path "${MODEL_PATH}" \
  --train_file "${TRAIN_JSONL}" \
  --eval_file "${VAL_JSONL}" \
  --output_dir "${OUT_DIR}" \
  "${FUSION_ARGS[@]}" \
  --collator "${COLLATOR}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_acc "${GRAD_ACC}" \
  --lr "${LR}" \
  --ft_params "${FT_PARAMS}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_bias "${LORA_BIAS}" \
  --language "${LANGUAGE}" \
  --epochs "${EPOCHS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --report_to "${REPORT_TO}" \
  2>&1 | tee -a "${LOG_FILE}"
