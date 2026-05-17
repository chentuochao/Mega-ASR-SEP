# Mega-ASR: Towards In-the-Wild Speech Recognition

<p align="center">
  <b>Robust Automatic Speech Recognition for Complex Real-World Acoustic Scenarios</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange">
  <img src="https://img.shields.io/badge/ASR-Robust%20Speech%20Recognition-brightgreen">
  <img src="https://img.shields.io/badge/License-Apache--2.0-green">
</p>


## Introduction

Mega-ASR is a speech recognition model designed for complex dirty speech scenarios, including noise, far-field speech, distortion, stuttering, echo, obstruction, and mixed acoustic interference. Compared with general-purpose ASR models, Mega-ASR focuses on stable recognition under medium- and high-error-rate audio conditions, achieving lower word error rates (WER) on challenging real-world speech.

This open-source release includes model weights, core training data, and newly constructed evaluation datasets, enabling researchers and developers to reproduce results, evaluate robustness, and advance ASR research in complex acoustic environments.

This repository is currently under active development.

## Highlights


- **Dirty and general ASR support**: robust recognition for both in-the-wild dirty speech and general audio.
- **550K-scale dirty speech corpus**: a large-scale degraded speech corpus covering noise, far-field speech, distortion, stuttering, echo, obstruction, and mixed acoustic interference.
- **SFT + RL robustness training**: a two-stage pipeline for improving recognition stability under complex acoustic conditions.
- **WER evaluation toolkit**: standard scripts for reproducible ASR robustness evaluation.
- **DAPO-LoRA roadmap**: reinforcement learning training code will be released in a future update.

## Project Structure

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

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Mega-ASR.git
cd Mega-ASR
pip install -r requirements.txt
```

## Inference

### Inference for Dirty Audio

```bash
python src_MegaASR/inference/inference_MegaASR_for_dirty.py \
  --audio path/to/audio.wav \
  --model_path path/to/model
```

### Inference for General Audio

```bash
python src_MegaASR/inference/inference_MegaASR_for_all.py \
  --audio path/to/audio.wav \
  --model_path path/to/model
```

## Evaluation

```bash
python eval/evaluate_wer.py \
  --pred predictions.jsonl \
  --ref references.jsonl
```

## Training

### SFT-LoRA Training

```bash
python src_MegaASR/train/SFT_lora/SFT_lora.py \
  --config configs/sft_lora.yaml
```

### DAPO-LoRA Training

The DAPO-LoRA training module is under active research and will be released in a future version.

## Roadmap

- [x] Repository structure
- [ ] Inference for dirty audio
- [ ] Inference for general audio
- [ ] WER evaluation
- [ ] SFT-LoRA training
- [ ] Model checkpoint release
- [ ] DAPO-LoRA release

## Citation

If you find this project useful, please consider citing our work. Citation information will be updated after the release of the paper.

## License

This project will be released under the Apache-2.0 License.