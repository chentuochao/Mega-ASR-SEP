

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
# TRAIN_JSONL=/data/haobin/batch_process/lora_0323_10w+55w+error+syn_with_domain_train90.jsonl
# VAL_JSONL=/data/haobin/batch_process/lora_0323_10w+55w+error+syn_with_domain_val5.jsonl
# OUT_DIR=/data/haobin/pky_train/qwen3/out_qwen3-asr-lora-0317_550000+our0320_both_2gpu_bs128_1e-5_2e-5_3e-6_resume_1
# LOG_FILE=/data/haobin/pky_train/qwen3/log_file/out_qwen3-asr-lora-0317_550000+our0320_both_2gpu_bs128_1e-5_2e-5_3e-6_resume.txt
# RUN_NAME=qwen3-asr-lora-0317_550000+our0320_both_2gpu_bs128_1e-5_2e-5_3e-6_resume

torchrun --nproc_per_node=2 --master_port=29520 train.py \
  --model_path /data/haobin/pky_train/qwen3/Qwen3-ASR-1.7B \
  --train_file ${TRAIN_JSONL} \
  --eval_file ${VAL_JSONL} \
  --output_dir ${OUT_DIR} \
  --batch_size 8 \
  --grad_acc 8 \
  --lr 1e-6 \
  --lr_tower 1e-5 \
  --lr_proj 2e-5 \
  --lr_llm 3e-6 \
  --epochs 2 \
  --save_steps 500 \
  --save_total_limit 300 \
  --use_lora 1 \
  --lora_scope both \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --run_name ${RUN_NAME} \
  --use_fixed_ratio_sampler 0 \
  --merge_lora_into_base_from /data/haobin/pky_train/qwen3/out_qwen3-asr-lora-0317_550000_wer3_towerb4+proj_2gpu_bs128/checkpoint-1000 \
  --save_adapter_only 1 2>&1 | tee -a ${LOG_FILE}

    # --mix_target_ratio 0.2 \
  # --mix_domain_field domain \
  # --mix_target_value targeted \