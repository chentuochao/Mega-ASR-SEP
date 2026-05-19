CUDA_VISIBLE_DEVICES=5 python inference_MegaASR_for_all.py \
  --audio example/distortion.wav \
  --model_path ckpt/Mega-ASR/Qwen3-ASR-1.7B \
  --lora_b_dir ckpt/Mega-ASR/ckpt/A2S-SFT-lora/lora1 \
  --lora_c_dir ckpt/Mega-ASR/ckpt/A2S-SFT-lora/lora2 \
  --lora_d_dir ckpt/Mega-ASR/ckpt/A2S-SFT-lora/lora3 \
  --quality_checkpoint ckpt/Mega-ASR/router/runs/exp_20260211_1layer/best_acc_model.pt \
  --quality_model_dir ckpt/Mega-ASR/router \
  --quality_threshold 0.5 
