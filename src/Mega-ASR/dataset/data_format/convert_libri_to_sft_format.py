#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert LibriSpeech JSONL to Mega-ASR SFT JSONL.

Input JSONL format:
{
  "index": 0,
  "audio_path": ".../LibriSpeech/test-clean/61/70968/61-70968-0000.flac",
  "answer": "THE TRANSCRIPT TEXT",
  "subset": "test_clean",
  "task_type": "understanding"
}

Output JSONL format:
{
  "audio": ".../wavs/test-clean/61/70968/61-70968-0000.wav",
  "text": "language English<asr_text>THE TRANSCRIPT TEXT",
  "prompt": ""
}

This script can optionally convert FLAC audio to WAV.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import librosa
import soundfile as sf
from tqdm import tqdm


def normalize_text(text: str, text_case: str = "none") -> str:
    text = " ".join(str(text).strip().split())

    if text_case == "lower":
        return text.lower()
    if text_case == "upper":
        return text.upper()
    if text_case == "none":
        return text

    raise ValueError(f"Unknown text_case: {text_case}")


def get_relative_audio_path(audio_path: str, input_audio_root: Optional[str] = None) -> Path:
    """
    尽量保留 LibriSpeech 的目录结构，例如：
    /data/.../LibriSpeech/test-clean/61/70968/61-70968-0000.flac
    ->
    test-clean/61/70968/61-70968-0000.wav
    """
    src = Path(audio_path).resolve()

    if input_audio_root:
        root = Path(input_audio_root).resolve()
        try:
            return src.relative_to(root)
        except ValueError:
            return Path(src.name)

    parts = src.parts
    if "LibriSpeech" in parts:
        idx = parts.index("LibriSpeech")
        return Path(*parts[idx + 1:])

    return Path(src.name)


def convert_flac_to_wav(
    audio_path: str,
    wav_dir: str,
    input_audio_root: Optional[str] = None,
    sr: int = 16000,
    overwrite: bool = False,
) -> str:
    """
    Convert one FLAC/WAV/audio file to WAV.

    - 默认转成 16kHz mono wav
    - 使用 PCM_16 保存
    - 保留 LibriSpeech 相对目录结构
    """
    src = Path(audio_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")

    rel_path = get_relative_audio_path(str(src), input_audio_root=input_audio_root)
    wav_path = Path(wav_dir).resolve() / rel_path.with_suffix(".wav")
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if wav_path.exists() and not overwrite:
        return str(wav_path)

    wav, _ = librosa.load(str(src), sr=sr, mono=True)
    sf.write(str(wav_path), wav, sr, subtype="PCM_16")

    return str(wav_path)


def get_answer_text(item: dict) -> str:
    """
    兼容两类输入：
    1. {"answer": "..."}
    2. {"text": "..."}
    """
    if "answer" in item and item["answer"] is not None:
        return str(item["answer"])

    if "text" in item and item["text"] is not None:
        text = str(item["text"])
        if "<asr_text>" in text:
            return text.split("<asr_text>", 1)[1]
        return text

    raise ValueError(f"Missing answer/text field in item: {item}")


def get_audio_path(item: dict) -> str:
    """
    兼容两类输入：
    1. {"audio_path": "..."}
    2. {"audio": "..."}
    """
    audio = item.get("audio_path", None)
    if audio is None:
        audio = item.get("audio", None)

    if not audio:
        raise ValueError(f"Missing audio_path/audio field in item: {item}")

    return str(audio)


def convert_one_item(
    item: dict,
    language: str,
    prompt: str,
    text_case: str,
    convert_to_wav: bool,
    wav_dir: Optional[str],
    input_audio_root: Optional[str],
    sr: int,
    overwrite_wav: bool,
) -> dict:
    audio_path = get_audio_path(item)
    answer = get_answer_text(item)
    answer = normalize_text(answer, text_case=text_case)

    if convert_to_wav:
        if wav_dir is None:
            raise ValueError("wav_dir must be provided when convert_to_wav=True")

        audio_path = convert_flac_to_wav(
            audio_path=audio_path,
            wav_dir=wav_dir,
            input_audio_root=input_audio_root,
            sr=sr,
            overwrite=overwrite_wav,
        )

    return {
        "audio": audio_path,
        "text": f"language {language}<asr_text>{answer}",
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert LibriSpeech JSONL to Mega-ASR SFT JSONL."
    )

    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)

    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--text_case",
        type=str,
        default="none",
        choices=["none", "lower", "upper"],
        help="LibriSpeech transcripts are usually uppercase. Default keeps original text.",
    )

    parser.add_argument(
        "--convert_to_wav",
        action="store_true",
        help="Convert input FLAC files to WAV and write WAV paths to output JSONL.",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=None,
        help="Directory to save converted WAV files. Required when --convert_to_wav is set.",
    )
    parser.add_argument(
        "--input_audio_root",
        type=str,
        default=None,
        help=(
            "Root directory used to preserve relative paths. "
            "Example: datasets/LibriSpeech_test/LibriSpeech"
        ),
    )
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--overwrite_wav", action="store_true")

    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.convert_to_wav and args.wav_dir is None:
        # 默认放在输出 jsonl 同级目录下的 wavs/
        args.wav_dir = str(output_jsonl.parent / "wavs")

    count = 0

    with input_jsonl.open("r", encoding="utf-8") as fin, output_jsonl.open(
        "w", encoding="utf-8"
    ) as fout:
        lines = [line for line in fin if line.strip()]

        for line in tqdm(lines, desc="Converting LibriSpeech"):
            item = json.loads(line)

            out = convert_one_item(
                item=item,
                language=args.language,
                prompt=args.prompt,
                text_case=args.text_case,
                convert_to_wav=args.convert_to_wav,
                wav_dir=args.wav_dir,
                input_audio_root=args.input_audio_root,
                sr=args.sr,
                overwrite_wav=args.overwrite_wav,
            )

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    print(f"[done] converted {count} samples")
    print(f"[output_jsonl] {output_jsonl}")

    if args.convert_to_wav:
        print(f"[wav_dir] {Path(args.wav_dir).resolve()}")


if __name__ == "__main__":
    main()