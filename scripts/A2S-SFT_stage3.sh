

#这个是从10w条中抽出3w条，先进行一个encoder+aligner的Lora
export CUDA_VISIBLE_DEVICES=6,7
#!/bin/bash  只训encoder+aligner的
set -euo pipefail

######################
# 0. 基础环境变量 (wandb)
######################
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_API_KEY=""
# export WANDB_PROJECT="qwen3-noise"    # 对应截图里的项目名
# export WANDB_ENTITY="pang_kaiyu-none"        # 对应截图里的 Entity

# 让 wandb 在多卡训练时只开一个进程写日志（可选）
export WANDB_MODE=online

# # 数据路径按你的实际替换
# TRAIN_JSONL=.jsonl
# VAL_JSONL=.jsonl
# OUT_DIR=
# LOG_FILE=.txt
# RUN_NAME=

torchrun --nproc_per_node=2 --master_port=29520 train.py \
  --model_path <The previous merge model> \
  --train_file ${TRAIN_JSONL} \
  --eval_file ${VAL_JSONL} \
  --output_dir ${OUT_DIR} \
  --batch_size <BATCH_SIZE> \
  --grad_acc <GRAD_ACC> \
  --lr <LR> \
  --lr_tower <LR_TOWER> \
  --lr_proj <LR_PROJ> \
  --lr_llm <LR_LLM> \
  --epochs <EPOCHS> \
  --save_steps <SAVE_STEPS> \
  --save_total_limit <SAVE_TOTAL_LIMIT> \
  --use_lora <USE_LORA> \
  --lora_scope all \
  --lora_r <LORA_R> \
  --lora_alpha <LORA_ALPHA> \
  --lora_dropout <LORA_DROPOUT> \
  --warmup_ratio <WARMUP_RATIO> \
  --max_grad_norm <MAX_GRAD_NORM> \
  --weight_decay <WEIGHT_DECAY> \
  --run_name ${RUN_NAME} \
  --use_fixed_ratio_sampler 0 \
  --merge_lora_into_base_from <The previous training checkpoint> \
  --save_adapter_only 1 2>&1 | tee -a ${LOG_FILE}

    # --mix_target_ratio 0.2 \
  # --mix_domain_field domain \
  # --mix_target_value targeted \