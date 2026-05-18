

#这个是从10w条中抽出3w条，先进行一个encoder+aligner的Lora
export CUDA_VISIBLE_DEVICES=2,3
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

# 数据路径按你的实际替换
# TRAIN_JSONL=/data/haobin/batch_process/lora_0311_10w+55w+noise_nost_error_train90_with_domain_wer0to3_train90.jsonl
# VAL_JSONL=/data/haobin/batch_process/lora_0311_10w+55w+noise_nost_error_train90_with_domain_wer0to3_val2.jsonl
# OUT_DIR=/data/haobin/pky_train/qwen3/out_qwen3-asr-lora-0317_550000_wer3_both_2gpu_bs128_2e-5_5e-5_5e-6
# LOG_FILE=/data/haobin/pky_train/qwen3/log_file/out_qwen3-asr-lora-0317_550000_wer3_both_2gpu_bs128_2e-5_5e-5_5e-6.txt
# RUN_NAME=qwen3-asr-lora-0317_550000_wer3_both_2gpu_bs128_2e-5_5e-5_5e-6

torchrun --nproc_per_node=2 --master_port=29505 train.py \
  --model_path /data/haobin/pky_train/qwen3/Qwen3-ASR-1.7B \
  --train_file ${TRAIN_JSONL} \
  --eval_file ${VAL_JSONL} \
  --output_dir ${OUT_DIR} \
  --batch_size 8 \
  --grad_acc 4 \
  --lr 1e-6 \
  --lr_tower 2e-5 \
  --lr_proj 5e-5 \
  --lr_llm 5e-6 \
  --epochs 2 \
  --save_steps 500 \
  --save_total_limit 300 \
  --use_lora 1 \
  --lora_scope llm \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --run_name ${RUN_NAME} \
  --use_fixed_ratio_sampler 1 \
  --save_adapter_only 1 2>&1 | tee -a ${LOG_FILE}