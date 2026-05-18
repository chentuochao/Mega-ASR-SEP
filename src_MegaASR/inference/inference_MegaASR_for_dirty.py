#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean version: Qwen3-ASR merged model single-audio inference with language output.

加载方式对应你的 Qwen3-ASR-1_7B-0514-merged_v2 版本：
- 模型路径默认：
  /data/haobin/Qwen3-ASR/Qwen3-ASR-1.7B-lora-merged_v2

默认只输出必要内容：
1. 初始化完成提示
2. 最终识别结果，例如：
   language English<asr_text>xxxx

不会输出：
- TVM / torch / transformers warning
- Loading checkpoint shards 进度条
- generation flags warning

需要调试时加 --verbose。
"""

import argparse
import contextlib
import json
import os
import sys
import warnings
from typing import Any

# 必须尽量放在 import torch / transformers / qwen_asr 之前
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def maybe_suppress_stderr(verbose: bool = False):
    """
    默认压掉第三方库 warning / 进度条 / 加载日志。
    verbose=True 时保留完整日志。
    """
    if verbose:
        yield
    else:
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr


def vprint(msg: str, verbose: bool):
    if verbose:
        print(msg, file=sys.stderr)


with maybe_suppress_stderr(verbose=False):
    import torch

    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


class Qwen3ASRMerged:
    NAME = "Qwen3-ASR-1_7B-0514-merged"

    def __init__(
        self,
        model_path: str = "/data/haobin/Qwen3-ASR/Qwen3-ASR-1.7B-lora-merged_v2",
        device_map: str = "cuda:0",
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.verbose = verbose

        with maybe_suppress_stderr(verbose):
            from qwen_asr import Qwen3ASRModel

        with maybe_suppress_stderr(verbose):
            self.model = Qwen3ASRModel.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map=device_map,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
            )

        # 保留必要初始化提示
        print(f"[Qwen3ASR] Model loaded from {self.model_path}", file=sys.stderr)

    @staticmethod
    def _unwrap_audio(audio):
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            return audio[0]
        return audio

    @staticmethod
    def _result_to_dict(result: Any) -> dict[str, str]:
        """
        Qwen3-ASR 官方示例里结果包含 result.language 和 result.text。
        这里直接读取这两个字段，不做语言映射。
        """
        language = getattr(result, "language", "")
        text = getattr(result, "text", None)

        if text is None:
            text = str(result)

        return {
            "language": str(language),
            "text": str(text),
        }

    @torch.no_grad()
    def transcribe_with_language(self, audio: str) -> list[dict[str, str]]:
        audio = self._unwrap_audio(audio)

        with maybe_suppress_stderr(self.verbose):
            results = self.model.transcribe(
                audio=audio,
                language=None,  # automatic language detection
            )

        if not isinstance(results, list):
            results = [results]

        return [self._result_to_dict(r) for r in results]

    @torch.no_grad()
    def infer_one(self, audio: str) -> dict[str, Any]:
        items = self.transcribe_with_language(audio)
        return {
            "audio": audio,
            "items": items,
            "model_path": self.model_path,
        }


def format_output(result: dict[str, Any], output_format: str) -> str:
    items = result["items"]

    if output_format == "tagged":
        return "\n".join(
            f"language {item['language']}<asr_text>{item['text']}"
            for item in items
        )

    if output_format == "language_text":
        return "\n".join(
            f"language: {item['language']}\ntext: {item['text']}"
            for item in items
        )

    if output_format == "text":
        return "\n".join(item["text"] for item in items)

    if output_format == "json":
        return json.dumps(result, ensure_ascii=False, indent=2)

    raise ValueError(f"Unknown output_format: {output_format}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean Qwen3-ASR merged model single-audio inference."
    )

    parser.add_argument("--audio", type=str, required=True, help="输入音频路径。")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/haobin/Qwen3-ASR/Qwen3-ASR-1.7B-lora-merged_v2",
        help="merged 模型路径。",
    )
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--max_inference_batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument(
        "--output_format",
        type=str,
        default="tagged",
        choices=["tagged", "language_text", "text", "json"],
        help=(
            "tagged: language English<asr_text>xxx；"
            "language_text: 分两行输出 language 和 text；"
            "text: 只输出 text；"
            "json: 输出完整 JSON。"
        ),
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="显示模型加载和 warning 信息。")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    model = Qwen3ASRMerged(
        model_path=args.model_path,
        device_map=args.device_map,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )

    result = model.infer_one(args.audio)
    output = format_output(result, args.output_format)
    print(output)

    if args.save_path is not None:
        save_dir = os.path.dirname(os.path.abspath(args.save_path))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        with open(args.save_path, "w", encoding="utf-8") as f:
            f.write(output + "\n")


if __name__ == "__main__":
    main()
