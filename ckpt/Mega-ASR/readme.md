## Checkpoints

Mega-ASR requires four checkpoint/model folders. We provide all required files in the Hugging Face repository:

```text
pangkaiyu/Mega-ASR
```

After downloading, the expected local directory structure is:

```text
ckpt/
└── Mega-ASR/
    ├── Qwen3-ASR-1.7B/
    ├── Mega-ASR_for_dirty/
    ├── router/
    └── A2S-SFT-lora/
```

In our scripts and examples, these folders are referenced as:

```text
ckpt/Mega-ASR/Qwen3-ASR-1.7B
ckpt/Mega-ASR/Mega-ASR_for_dirty
ckpt/Mega-ASR/router
ckpt/Mega-ASR/A2S-SFT-lora
```

### Checkpoint Description

| Folder | Description |
|---|---|
| `ckpt/Mega-ASR/Qwen3-ASR-1.7B` | The base Qwen3-ASR model used as the backbone ASR model. |
| `ckpt/Mega-ASR/Mega-ASR_for_dirty` | The Mega-ASR model/checkpoint adapted for dirty or degraded speech recognition. |
| `ckpt/Mega-ASR/router` | The audio-quality router used to determine whether the input audio should be processed by the general ASR path or the dirty-speech-enhanced path. |
| `ckpt/Mega-ASR/A2S-SFT-lora` | The A2S-SFT LoRA adapter used for supervised fine-tuning on complex acoustic conditions. |

### Download Checkpoints

First, make sure you have logged in to Hugging Face if the repository is private:

```bash
hf auth login
```

Then download the full repository into `ckpt/Mega-ASR/`:

```bash
mkdir -p ckpt/Mega-ASR

huggingface-cli download pangkaiyu/Mega-ASR \
  --repo-type model \
  --local-dir ckpt/Mega-ASR
```

Alternatively, you can use the newer `hf` command:

```bash
mkdir -p ckpt/Mega-ASR

hf download pangkaiyu/Mega-ASR \
  --repo-type model \
  --local-dir ckpt/Mega-ASR
```

After downloading, check the folder structure:

```bash
ls ckpt/Mega-ASR
```

You should see:

```text
A2S-SFT-lora
Mega-ASR_for_dirty
Qwen3-ASR-1.7B
router
README.md
.gitattributes
```

The extra files such as `README.md` and `.gitattributes` are normal and can be kept.

### Download Only Required Folders

If you only want to download the four required folders, run:

```bash
mkdir -p ckpt/Mega-ASR

huggingface-cli download pangkaiyu/Mega-ASR \
  --repo-type model \
  --include "Qwen3-ASR-1.7B/*" \
  --include "Mega-ASR_for_dirty/*" \
  --include "router/*" \
  --include "A2S-SFT-lora/*" \
  --local-dir ckpt/Mega-ASR
```

After this step, the model paths used by the inference scripts should be:

```bash
--model_path ckpt/Mega-ASR/Qwen3-ASR-1.7B
--dirty_model_path ckpt/Mega-ASR/Mega-ASR_for_dirty
--quality_model_dir ckpt/Mega-ASR/router
--adapter_dir ckpt/Mega-ASR/A2S-SFT-lora
```

For example, the router-based inference script may use:

```bash
python inference_MegaASR_for_all.py \
  --model_path ckpt/Mega-ASR/Qwen3-ASR-1.7B \
  --lora_b_dir ckpt/Mega-ASR/A2S-SFT-lora \
  --quality_model_dir ckpt/Mega-ASR/router \
  --quality_checkpoint ckpt/Mega-ASR/router/runs/exp_20260211_1layer/best_acc_model.pt \
  --audio examples/noise.wav
```

Please adjust the exact checkpoint file name under `ckpt/Mega-ASR/router/` according to the downloaded router directory.