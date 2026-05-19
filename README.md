<p align="center">
  <img src="assets/figures/mega_asr_logo.png" alt="Mega-ASR Logo" width="220">
</p>

<h1 align="center">Mega-ASR: Towards In-the-Wild Speech Recognition</h1>

<p align="center">
  <b>Robust Automatic Speech Recognition for Complex Real-World Acoustic Scenarios</b>
</p>

<p align="center">
  <a href="https://xzf-thu.github.io/Mega-ASR/"><b>Homepage</b></a> |
  <a href="#model-download"><b>Model Download</b></a> |
  <a href="#installation"><b>Our Bench Download</b></a> |
  <a href="#paper"><b>Paper</b></a> |
</p>

<p align="center">
  <a href="https://xzf-thu.github.io/Mega-ASR/">
    <img src="https://img.shields.io/badge/Project-Homepage-purple">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange">
  <img src="https://img.shields.io/badge/ASR-Robust%20Speech%20Recognition-brightgreen">
  <img src="https://img.shields.io/badge/License-Apache--2.0-green">
</p>



<p>
We present <b>Mega-ASR</b>, an open-source speech recognition model designed for stable and robust ASR under complex dirty speech conditions, especially on medium- and high-error-rate audio.

<p>
This repository contains the official implementation, model weights, core training data, and evaluation toolkit for Mega-ASR.
</p>


<p align="center">
  <b>🚀 When conventional ASR systems fail under real-world acoustic interference, come to Mega-ASR!</b>
</p>




## 👀 What You Must See

The following examples compare Mega-ASR with several representative ASR systems under challenging dirty speech conditions. Click **Listen** to play each audio sample.

---

<details open>
<summary><b>▶️ Empty Output Recovery</b></summary>

<br>

🎧 [Listen to audio](assets/case_study/empty_output_recovery.wav)

| Model | WER | Transcript |
|---|---|---|
| Ground Truth | Reference | "...and said to him let us go and eat some honey. Whose honey? inquired Kobay cautiously. My father's, Soongoora replied. Oh, all right, I'm with you, said the tortoise eagerly, and away they went." |
| **Mega-ASR (Ours)** | ✅ <mark><b>47.1</b></mark> | <b>"He said to him let's go and eat some honey. It's honey? he inquired very cautiously. My father is Superabundant — oh, all right, I will, he said to her eagerly, and away they went."</b> |
| Qwen3-ASR | 🔴 **100.0** | <i>&lt;empty&gt;</i> |
| Gemini-3-Pro | 🔴 **86.1** | "But tell me, that's how she met my father's sister. Oh, alright. I wish... I really..." |
| Seed-ASR | 🔴 **85.3** | "My father is. Oh, all right, I wish you can." |
| Whisper | 🔴 **92.5** | "...to him... some honey... oh yeah..." |

</details>

---

<details>
<summary><b>▶️ Long-Utterance Semantic Recovery</b></summary>

<br>

🎧 [Listen to audio](assets/case_study/long_utterance_recovery.wav)

| Model | WER | Transcript |
|---|---|---|
| Ground Truth | Reference | "To waste, I skip forty years, said the baker in tears, and proceed without further remark to the day when you took me aboard your ship to help you in hunting the snark." |
| **Mega-ASR (Ours)** | ✅ <mark><b>5.9</b></mark> | <b>"To witness, I skip forty years, said the baker in tears, and proceed without further remark to the day when you took me aboard of your ship to help you in hunting the snark."</b> |
| Qwen3-ASR | 🟠 **64.7** | "I skipped 40 years. Second day in here. Ever since you left, I've been a monk..." |
| Gemini-3-Pro | 🟠 **64.7** | "I spent forty years at sea and never seen a rougher than the day that you took me aboard your ship..." |
| Seed-ASR | 🟡 **38.2** | "To wait. I skip forty years. Saturday and years. And proceed without further remark..." |
| Whisper | 🟠 **71.5** | "I skip forty years... to the day you took me on a ship... to hunt the shark." |

</details>

---

<details>
<summary><b>▶️ Babble Noise & Hallucination</b></summary>

<br>

🎧 [Listen to audio](assets/case_study/babble_noise_hallucination.wav)

| Model | WER | Transcript |
|---|---|---|
| Ground Truth | Reference | "The friendly gang left the drug store." |
| **Mega-ASR (Ours)** | ✅ <mark><b>8.0</b></mark> | <b>"The friendly gang left the drug store."</b> |
| Qwen3-ASR | 🟠 **57.1** | "It's a friendly gang. That's the drug gang." |
| Gemini-3-Pro | 🟡 **42.9** | "Friendly gang left the drugs." |
| Seed-ASR | 🟢 **28.6** | "The friendly gang left the drugstore." |
| Whisper | 🟠 **62.3** | "A friendly young man left the drug store." |

</details>

---

<details>
<summary><b>▶️ Restaurant Noise Recovery</b></summary>

<br>

🎧 [Listen to audio](assets/case_study/restaurant_noise_recovery.wav)

| Model | WER | Transcript |
|---|---|---|
| Ground Truth | Reference | "The set of china hit the floor with a crash." |
| **Mega-ASR (Ours)** | ✅ <mark><b>8.0</b></mark> | <b>"The set of china hit the floor with a crash."</b> |
| Qwen3-ASR | 🟡 **40.0** | "The bed is fine. It hit the floor with a crash." |
| Gemini-3-Pro | 🔴 **100.0** | "He said it's fine I hit the forward slash." |
| Seed-ASR | 🟢 **20.0** | "The sound of china hits the floor with a crash." |
| Whisper | 🟠 **55.0** | "The chef of China hit the floor with a clash." |

</details>

---

<details>
<summary><b>▶️ Financial Entity Recovery</b></summary>

<br>

🎧 [Listen to audio](assets/case_study/financial_entity_recovery.wav)

| Model | WER | Transcript |
|---|---|---|
| Ground Truth | Reference | "Among export-led electrical and computer makers, Japan Victor Company fell fifty to two thousand three hundred twenty." |
| **Mega-ASR (Ours)** | ✅ <mark><b>11.1</b></mark> | <b>"Among export-led computer makers, Japan Victor Company fell fifty to two thousand three hundred twenty."</b> |
| Qwen3-ASR | 🟡 **38.9** | "Among export-led computer makers, Japan VictorNet sold fifty-two thousand three hundred fifty." |
| Gemini-3-Pro | 🟡 **35.7** | "Among export-led computer makers, Japan Victor Co. fell 50 to 2,350 yen." |
| Seed-ASR | 🟠 **50.0** | "Among export-led in computer makers, Japan Victor Company sell 50 to 2300 unit." |
| Whisper | 🟠 **66.7** | "Among exporters, computer makers in Japan victor companies sold fifty..." |

</details>

---

<details>
<summary><b>▶️ Phrase Recovery</b></summary>

<br>

🎧 [Listen to audio](assets/case_study/phrase_recovery.wav)

| Model | WER | Transcript |
|---|---|---|
| Ground Truth | Reference | "Has exposure really been reduced?" |
| **Mega-ASR (Ours)** | ✅ <mark><b>8.0</b></mark> | <b>"Has exposure really been reduced."</b> |
| Qwen3-ASR | 🟡 **40.0** | "Has exposure really done you?" |
| Gemini-3-Pro | 🔴 **80.0** | "Has the closure really affected you?" |
| Seed-ASR | 🟠 **60.0** | "Has exposure to beauty products." |
| Whisper | 🔴 **78.5** | "Have those who really been refused?" |

</details>

## 🔥🔥🔥 News!!

- **May 20, 2025**: 🔥 We release **Mega-ASR**. Model weights on Hugging Face are coming soon.
- **May 20, 2025**: 🔥 We release **Voices-in-the-Wild-2M**, a benchmark for in-the-wild ASR robustness evaluation. [[Dataset]](https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-test-v2)
- **Coming soon**: 🔥 We will release the **DAPO-LoRA training code**.

## Contents

- [Introduction](#introduction)
- [Model Download](#model-download)
- [Main Results](#main-results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Inference](#inference)
- [Finetune](#finetune)
- [Evaluation](#evaluation)



## Introduction


Mega-ASR is designed for speech recognition in complex real-world acoustic environments, where speech signals are often affected by noise, reverberation, far-field recording, low volume, distortion, stuttering, echo, obstruction, and multiple overlapping interferences. Unlike general-purpose ASR systems that mainly perform well on clean or moderately noisy speech, Mega-ASR focuses on medium- and high-error-rate audio conditions, where recognition stability becomes more challenging.

To improve robustness, Mega-ASR is built with large-scale dirty speech data and a two-stage robustness training pipeline. The released resources include model weights, core training data, evaluation benchmarks, and WER/CER evaluation scripts, enabling reproducible research and further development of robust ASR systems for in-the-wild scenarios.

- **Robust dirty and general ASR**: supports stable recognition for both in-the-wild dirty speech and general audio.
- **2M-scale dirty speech corpus**: covers noise, far-field recording, distortion, stuttering, echo, obstruction, and mixed acoustic interference.
- **SFT + RL robustness training**: improves recognition stability under complex acoustic conditions through supervised fine-tuning and reinforcement learning.
- **Reproducible WER/CER evaluation**: provides standard scripts and benchmarks for ASR robustness evaluation.
- **DAPO-LoRA roadmap**: reinforcement learning training code will be released in a future update.


## Model Download

We provide two Mega-ASR model variants for different usage scenarios.

| Model | Description | Download |
|---|---|---|
| **Mega-ASR for Dirty** | Optimized for dirty speech scenarios, including noisy, far-field, low-volume, degraded, and hard-to-recognize audio. | Coming soon |
| **Mega-ASR for All** | Built upon Mega-ASR for Dirty with a lightweight routing module that automatically distinguishes clean speech from degraded speech and selects the appropriate recognition path. | Coming soon |

After downloading the model weights, please specify the model path in the corresponding inference script or pass it through command-line arguments.


## Project Structure


<p align="center">
  <img src="assets/figures/method_overview.png" alt="Mega-ASR Method Overview" width="95%">
</p>

<p align="center">
  <b>Figure 1.</b> Overview of the Mega-ASR training pipeline, including acoustic-to-speech supervised fine-tuning and reward-based optimization for robust speech recognition.
</p>

```text
Mega-ASR/
├─ assets/
│  └─ Figures, logos, and other README resources.
│
├─ configs/
│  └─ Configuration files for SFT-LoRA and DAPO-LoRA training.
│
├─ data/
│  └─ Local data directory. Large-scale audio data is not tracked by Git.
│
├─ eval/
│  └─ evaluate_wer.py
│     WER/CER evaluation utilities for ASR robustness testing.
│
└─ src_MegaASR/
   ├─ inference/
   │  ├─ inference_MegaASR_for_dirty.py
   │  │  Dirty-speech inference without routing, designed for degraded audio.
   │  │
   │  └─ inference_MegaASR_for_all.py
   │     General inference with routing, supporting both dirty and general audio.
   │
   └─ train/
      ├─ SFT_lora/
      │  └─ SFT_lora.py
      │     SFT-LoRA training pipeline for acoustic robustness adaptation.
      │
      └─ DAPO_lora/
         └─ DAPO-LoRA training module, to be released in a future update.
```

## Main Results

Mega-ASR is evaluated across three benchmark families, including noisy and robust ASR benchmarks, Voices-in-the-Wild-Bench, and standard ASR benchmarks. Lower WER/CER indicates better recognition performance.

<p align="center">
  <img src="assets/figures/radar_results.png" alt="Radar comparison of Mega-ASR" width="95%">
</p>

<p align="center">
  <b>Figure 2.</b> Radar comparison of Qwen3-ASR-1.7B and Mega-ASR across selected ASR evaluation subsets.
</p>

### Noisy and Robust ASR Benchmarks

<p align="center">
  <img src="assets/tables/noisy_robust_asr_benchmarks.png" alt="Performance comparison on noisy and robust ASR benchmarks" width="95%">
</p>

<p align="center">
  <b>Table 1.</b> Performance comparison on noisy and robust ASR benchmarks.
</p>

### Voices-in-the-Wild-Bench

<p align="center">
  <img src="assets/tables/voices_in_the_wild_breakdown.png" alt="Breakdown results on Voices-in-the-Wild-Bench" width="95%">
</p>

<p align="center">
  <b>Table 2.</b> Breakdown results on Voices-in-the-Wild-Bench by acoustic scenario.
</p>

### Standard ASR Benchmarks

<p align="center">
  <img src="assets/tables/standard_asr_benchmarks.png" alt="Performance comparison on standard ASR benchmarks" width="95%">
</p>

<p align="center">
  <b>Table 3.</b> Performance comparison on standard ASR benchmarks. For LibriSpeech, each entry is reported as clean/other.
</p>

## Quick Start

### 1. Create Environment

We recommend using Conda to create an isolated Python environment.

```bash
conda create -n mega-asr2 python=3.12 -y
conda activate mega-asr2
```

Upgrade basic Python build tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch

Install PyTorch with CUDA 12.8 support:

```bash
pip install  torch==2.10.0   torchaudio==2.10.0   torchvision==0.25.0
```

### 3. Install Mega-ASR Dependencies

```bash
pip install -r mega_asr_requirements.txt
```

### 4. Install Qwen3-ASR Dependency

Mega-ASR is built upon Qwen3-ASR. Please prepare the Qwen3-ASR source code locally and install it in editable mode:

```bash
pip install -e /path/to/Qwen3-ASR --no-deps
```

For example, replace `/path/to/Qwen3-ASR` with the actual local path of your Qwen3-ASR repository.

## Inference

Mega-ASR provides two inference modes for different usage scenarios.



### 1. Inference for Dirty Audio

This mode is designed for degraded or hard-to-recognize speech, such as noisy, far-field, distorted, or mixed-interference audio.



```bash
bash scripts/inference_MegaASR_for_dirty.sh
```

### 2. Inference for General Audio

This mode supports both dirty speech and general audio. It uses a routing mechanism to select the appropriate recognition path automatically.

### 3. Web-based Inference

We also provide Gradio-based web inference scripts for interactive testing.

For dirty-audio inference:

```bash
bash scripts/web_inference_MegaASR_for_dirty.sh
```

For general audio inference with automatic routing:

```bash
bash scripts/web_inference_MegaASR_for_all.sh
```



## Evaluation

We provide `evaluate_wer.py` to run Qwen3-ASR inference and compute WER/CER on JSONL-formatted evaluation data.

```bash
CUDA_VISIBLE_DEVICES=6,7 python evaluate_wer.py \
  --input_jsonl example/examples.jsonl \
  --output_jsonl output_with_wer.jsonl
```

The script loads Qwen3-ASR, transcribes each audio file, writes the generated transcription to the `prediction` field, and computes the error rate against the reference transcription.

- English samples are evaluated with **WER**.
- Chinese samples are evaluated with **CER**.
- The output JSONL keeps the original fields and adds prediction and error-rate information.
- The input JSONL does **not** need to contain a `prediction` field. The `prediction` field is generated by `evaluate_wer.py`.
- Please make sure that `evaluate_wer.py` and `cn_tn.py` are placed in the same directory. The `cn_tn.py` module is used for Chinese text normalization before CER computation.

### Input Format

Each line in the input file should be a JSON object. An example is shown below:

```json
{
  "index": 1755,
  "audio_path": "examples/noise.wav",
  "question": "Please transcribe the audio content into text.",
  "answer": "I usually take the quieter road home because the main street gets crowded after work.",
}
```

Required fields are:

| Field | Description |
|---|---|
| `audio_path` | Path to the input audio file. |
| `answer` | Ground-truth transcription. |

Optional fields such as `index`will be kept unchanged in the output JSONL.

If the audio path is relative, it should be relative to the current working directory or to the JSONL file location, depending on how the script is executed.

### Output Format

The output file is also a JSONL file. It preserves the original fields and adds Qwen3-ASR prediction and error-rate information.

Example output:

```json
{
  "index": 1755,
  "audio_path": "examples/noise.wav",
  "question": "Please transcribe the audio content into text.",
  "answer": "I usually take the quieter road home because the main street gets crowded after work.",
  "prediction": "I usually take the quieter road home because the main street gets crowded after work.",
  "pred_language": "English",
  "wer": 0.0,
  "metric": "wer",
  "num_edits": 0,
  "ref_len": 15
}
```

For Chinese samples, `metric` will be set to `"cer"`. For compatibility with the evaluation pipeline, the error-rate value is still stored in the `wer` field, but it represents CER when `metric` is `"cer"`.

### WER / CER Computation

For English samples, the script computes Word Error Rate (WER):

```text
WER = (S + D + I) / N
```

where:

- `S` is the number of substitutions.
- `D` is the number of deletions.
- `I` is the number of insertions.
- `N` is the number of words in the reference transcription.

For Chinese samples, the script computes Character Error Rate (CER):

```text
CER = (S + D + I) / N
```

where `N` is the number of characters in the reference transcription.

Before computing WER/CER, the script performs text normalization. English text is normalized before word-level comparison. Chinese text is normalized with `cn_tn.py` before character-level comparison.

### File Placement

Please place `evaluate_wer.py` and `cn_tn.py` in the same directory:

```text
eval/
├── evaluate_wer.py
├── cn_tn.py
└── example/
    └── examples.jsonl
```

Then run the evaluation command from the same directory:

```bash
cd eval

CUDA_VISIBLE_DEVICES=6,7 python evaluate_wer.py \
  --input_jsonl example/examples.jsonl \
  --output_jsonl output_with_wer.jsonl
```


## Dirty Speech Data Generation

The data generation code for our Voices-in-the-Wild training data is provided under:

```text
dataset/dataloader/
```

The core scheduling entry is:

```text
dataset/dataloader/scheduler.py
```

This module is responsible for organizing the data generation workflow and scheduling different data construction components.

Please note that the detailed implementations of individual perturbation functions, as well as the exact parameter settings used for single or mixed acoustic corruptions, are not included in this public release. These components are currently closed-source. The released dataloader code mainly provides the overall data scheduling interface and the public structure used in our training pipeline.




## Finetune

Mega-ASR supports acoustic robustness adaptation through both supervised fine-tuning and reinforcement learning based optimization.



### 1. A2S-SFT

SFT-LoRA is used to adapt Mega-ASR to complex dirty speech scenarios with supervised training data.

```bash
bash A2S-SFT_stage1.sh
bash A2S-SFT_stage2.sh
bash A2S-SFT_stage3.sh
```


#### Hyperparameter Configuration

The following training hyperparameters are exposed as placeholders in the public script. Users should set them according to their own dataset scale, GPU memory, LoRA configuration, and training objective.

```bash
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
--lora_scope <LORA_SCOPE> \
--lora_r <LORA_R> \
--lora_alpha <LORA_ALPHA> \
--lora_dropout <LORA_DROPOUT> \
--warmup_ratio <WARMUP_RATIO> \
--max_grad_norm <MAX_GRAD_NORM> \
--weight_decay <WEIGHT_DECAY>
```

Here, `<LR_TOWER>`, `<LR_PROJ>`, and `<LR_LLM>` correspond to the learning rates of different LoRA parameter groups. We do not provide fixed default values for these hyperparameters, since they are sensitive to the training data, corruption distribution, batch size, and target LoRA scope.

Please record the final hyperparameter values used in your own experiments for reproducibility.


#### Training Data Format

The SFT training data should be provided in JSONL format. Each line corresponds to one audio-text training sample.

The expected format is:

```json
{
  "audio": ".../wavs/test-clean/61/70968/61-70968-0000.wav",
  "text": "language English<asr_text>THE TRANSCRIPT TEXT",
  "prompt": ""
}
```

Field descriptions:

| Field | Description |
|---|---|
| `audio` | Path to the input audio file. |
| `text` | Target transcription. The transcription should follow the format `language <LANGUAGE><asr_text><TRANSCRIPTION>`. |
| `prompt` | Optional prompt field. It can be left empty for standard ASR training. |

For English ASR data, the `text` field should typically be written as:

```text
language English<asr_text>THE TRANSCRIPT TEXT
```

For Chinese ASR data, the language tag can be changed accordingly, for example:

```text
language Chinese<asr_text>转写文本
```

#### LibriSpeech Format Conversion

We provide a script to convert the default LibriSpeech JSONL metadata into the Mega-ASR SFT training format.

The conversion script is located at:

```text
dataset/data_format/convert_libri_to_sft_format.sh
```

Example usage:

```bash
bash dataset/data_format/convert_libri_to_sft_format.sh
```

The converted output JSONL follows the training format required by Mega-ASR:

```json
{
  "audio": ".../wavs/test-clean/61/70968/61-70968-0000.wav",
  "text": "language English<asr_text>THE TRANSCRIPT TEXT",
  "prompt": ""
}
```

This script is mainly intended for converting LibriSpeech-style metadata into the unified Mega-ASR SFT format. It can also be used as a reference for preparing custom ASR training data.


### 2. DG-WGPO

DG-WGPO is designed for reinforcement learning based robustness optimization after supervised fine-tuning.


The DG-WGPO training module is under active research and will be released in a future update.




## Citation

If you find this project useful, please consider citing our work. Citation information will be updated after the release of the paper.

## License

This project will be released under the Apache-2.0 License.

123