#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean version: Qwen3-ASR router single-audio inference with language output.

默认只输出最终结果，例如：
language English<asr_text>xxxx

不会输出：
- TVM / torch / transformers warning
- Loading checkpoint shards 进度条
- LoRA loaded_delta 日志
- route 日志

需要调试时加 --verbose。
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time
import warnings
from typing import Any

# 必须放在 import torch / transformers / qwen_asr 之前
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def maybe_suppress_stderr(verbose: bool = False):
    """
    默认压掉第三方库 warning / 进度条 / 加载日志。
    出错时 Python traceback 仍会在 context 退出后正常显示。
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
    import soundfile as sf
    from scipy.signal import resample_poly
    from safetensors.torch import load_file as safe_load_file

    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


class AudioQualityPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        model_dir: str,
        device: str = "cuda",
        threshold: float = 0.5,
        verbose: bool = False,
    ):
        self.device = device
        self.threshold = threshold
        self.enabled = False
        self.verbose = verbose

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            vprint(f"[QualityPredictor] checkpoint 不存在: {checkpoint_path}", verbose)
            return

        if not os.path.exists(model_dir):
            vprint(f"[QualityPredictor] model_dir 不存在: {model_dir}", verbose)
            return

        try:
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)

            with maybe_suppress_stderr(verbose):
                from src.model import create_model
                from src.dataset import LogMelSpectrogram

                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                config = checkpoint.get("config", {}).get("model", {})

                self.model = create_model(config)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.to(device)
                self.model.eval()

                self.mel_extractor = LogMelSpectrogram(
                    sample_rate=16000,
                    n_mels=config.get("n_mels", 80),
                ).to(device)

            self.enabled = True
            vprint(f"[QualityPredictor] 加载成功, threshold={threshold}", verbose)

        except Exception as e:
            vprint(f"[QualityPredictor] 加载失败: {e}", verbose)
            self.enabled = False

    @torch.no_grad()
    def predict(self, audio_path: str) -> tuple[bool, float]:
        if not self.enabled:
            return True, 1.0

        try:
            audio_np, sr = sf.read(audio_path, always_2d=True)
            audio_np = audio_np.mean(axis=1)

            if sr != 16000:
                g = math.gcd(sr, 16000)
                audio_np = resample_poly(audio_np, 16000 // g, sr // g)

            waveform = torch.from_numpy(audio_np).float().unsqueeze(0)

            max_samples = 30 * 16000
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

            waveform = waveform.to(self.device)

            with maybe_suppress_stderr(self.verbose):
                mel = self.mel_extractor(waveform)
                mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0)
                logits = self.model(mel, mask=None)
                probs = torch.softmax(logits, dim=-1)

            dirty_prob = probs[0, 1].item()
            is_dirty = dirty_prob >= self.threshold
            return is_dirty, dirty_prob

        except Exception as e:
            vprint(f"[QualityPredictor] 预测失败 {audio_path}: {e}", self.verbose)
            return True, 1.0


class LoRADeltaSwitch:
    def __init__(self, keep_delta_on_gpu: bool = True, verbose: bool = False):
        self.keep_delta_on_gpu = keep_delta_on_gpu
        self.verbose = verbose
        self.items = []
        self.active = False

    def _load_adapter_state(self, adapter_dir: str) -> dict[str, torch.Tensor]:
        safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_dir, "adapter_model.bin")

        if os.path.exists(safetensors_path):
            return safe_load_file(safetensors_path)

        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu")

        raise FileNotFoundError(
            f"Cannot find adapter_model.safetensors or adapter_model.bin under {adapter_dir}"
        )

    def _load_adapter_config(self, adapter_dir: str) -> dict[str, Any]:
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Cannot find adapter_config.json under {adapter_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_module_name(self, name: str) -> str:
        for prefix in ["base_model.model.", "model."]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    def _split_lora_key(self, key: str):
        key = self._normalize_module_name(key)

        for marker in [".lora_A.", ".lora_B."]:
            if marker in key:
                module_name = key.split(marker)[0]
                kind = "A" if marker == ".lora_A." else "B"
                return module_name, kind

        return None, None

    def add_adapter(
        self,
        parent_module: torch.nn.Module,
        adapter_dir: str,
        name: str,
        strip_prefixes: list[str] | None = None,
    ):
        config = self._load_adapter_config(adapter_dir)
        state = self._load_adapter_state(adapter_dir)

        lora_alpha = config.get("lora_alpha", 1)
        r = config.get("r", None)
        fan_in_fan_out = bool(config.get("fan_in_fan_out", False))

        module_dict = dict(parent_module.named_modules())
        grouped = {}

        for key, tensor in state.items():
            module_name, kind = self._split_lora_key(key)
            if module_name is None:
                continue

            if strip_prefixes:
                for p in strip_prefixes:
                    if module_name.startswith(p):
                        module_name = module_name[len(p):]

            grouped.setdefault(module_name, {})[kind] = tensor.cpu()

        loaded = 0
        missing = []

        for module_name, pair in grouped.items():
            if "A" not in pair or "B" not in pair:
                continue

            if module_name not in module_dict:
                missing.append(module_name)
                continue

            module = module_dict[module_name]

            if not hasattr(module, "weight"):
                missing.append(module_name)
                continue

            A = pair["A"].float()
            B = pair["B"].float()

            rank = r if r is not None else A.shape[0]
            scaling = float(lora_alpha) / float(rank)

            delta = torch.matmul(B, A) * scaling
            weight = module.weight

            if fan_in_fan_out:
                delta = delta.T

            if delta.shape != weight.shape:
                try:
                    delta = delta.reshape(weight.shape)
                except Exception:
                    missing.append(
                        f"{module_name}: delta shape {tuple(delta.shape)} != weight shape {tuple(weight.shape)}"
                    )
                    continue

            delta = delta.to(dtype=weight.dtype)

            if self.keep_delta_on_gpu:
                delta = delta.to(device=weight.device)
            else:
                delta = delta.cpu()

            self.items.append(
                {
                    "name": name,
                    "module_name": module_name,
                    "weight": weight,
                    "delta": delta,
                }
            )

            loaded += 1

        vprint(f"[LoRADeltaSwitch] adapter={name}, loaded_delta={loaded}, missing={len(missing)}", self.verbose)

        if loaded == 0:
            raise ValueError(f"No LoRA delta loaded for adapter {name} from {adapter_dir}")

    @torch.no_grad()
    def set_active(self, active: bool):
        if self.active == active:
            return 0.0

        t0 = time.perf_counter()
        sign = 1.0 if active else -1.0

        for item in self.items:
            weight = item["weight"]
            delta = item["delta"]

            if delta.device != weight.device:
                delta = delta.to(device=weight.device)

            weight.data.add_(delta, alpha=sign)

        self.active = active
        return time.perf_counter() - t0


class Qwen3ASRRouter:
    def __init__(
        self,
        model_path: str = "ckpt/Mega-ASR/Qwen3-ASR-1.7B",
        lora_b_dir: str = "ckpt/Mega-ASR/A2S-SFT-lora/lora1",
        lora_c_dir: str = "ckpt/Mega-ASR/A2S-SFT-lora/lora2",
        lora_d_dir: str = "ckpt/Mega-ASR/A2S-SFT-lora/lora1",
        quality_model_dir: str = "ckpt/Mega-ASR/router",
        quality_checkpoint: str = "ckpt/Mega-ASR/router/runs/exp_20260211_1layer/best_acc_model.pt",
        quality_threshold: float = 0.5,
        routing_enabled: bool = True,
        device_map: str = "cuda:0",
        quality_device: str = "cuda",
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.routing_enabled = routing_enabled
        self.verbose = verbose

        with maybe_suppress_stderr(verbose):
            from qwen_asr import Qwen3ASRModel

        if self.routing_enabled:
            vprint("[Qwen3ASR] 加载音频质量预测器...", verbose)
            self.quality_predictor = AudioQualityPredictor(
                checkpoint_path=quality_checkpoint,
                model_dir=quality_model_dir,
                device=quality_device,
                threshold=quality_threshold,
                verbose=verbose,
            )
            if not self.quality_predictor.enabled:
                vprint("[Qwen3ASR] 质量预测器加载失败，将始终使用 LoRA", verbose)
                self.routing_enabled = False
        else:
            vprint("[Qwen3ASR] 路由已禁用，将始终使用 LoRA", verbose)
            self.quality_predictor = None

        vprint(f"[Qwen3ASR] 加载单个 Base 模型: {self.model_path}", verbose)

        with maybe_suppress_stderr(verbose):
            self.model = Qwen3ASRModel.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map=device_map,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
            )

        if not hasattr(self.model.model, "thinker"):
            raise ValueError("self.model.model does not have attribute `thinker`.")

        self.lora_switch = LoRADeltaSwitch(keep_delta_on_gpu=True, verbose=verbose)

        vprint("[Qwen3ASR] 预计算三段 LoRA delta...", verbose)

        with maybe_suppress_stderr(verbose):
            self.lora_switch.add_adapter(
                parent_module=self.model.model.thinker,
                adapter_dir=lora_b_dir,
                name="sft1",
            )
            self.lora_switch.add_adapter(
                parent_module=self.model.model.thinker,
                adapter_dir=lora_c_dir,
                name="sft2",
            )
            self.lora_switch.add_adapter(
                parent_module=self.model.model,
                adapter_dir=lora_d_dir,
                name="sft3",
            )

            self._set_lora(True)
            torch.cuda.empty_cache()

        print("[Qwen3ASR] 初始化完成：单实例 + 三 LoRA delta switch + 质量路由", file=sys.stderr)

    def _set_lora(self, active: bool):
        elapsed = self.lora_switch.set_active(active)
        if elapsed > 0:
            direction = "base->lora" if active else "lora->base"
            vprint(f"[LoRA Switch] {direction}: {elapsed * 1000:.3f} ms", self.verbose)

    @staticmethod
    def _result_to_dict(result: Any) -> dict[str, str]:
        language = getattr(result, "language", "")
        text = getattr(result, "text", None)
        if text is None:
            text = str(result)
        return {"language": str(language), "text": str(text)}

    @torch.no_grad()
    def transcribe_with_language(self, audio: str) -> list[dict[str, str]]:
        with maybe_suppress_stderr(self.verbose):
            results = self.model.transcribe(
                audio=audio,
                language=None,
            )

        if not isinstance(results, list):
            results = [results]

        return [self._result_to_dict(r) for r in results]

    @torch.no_grad()
    def infer_one(self, audio: str, force_base: bool = False, force_lora: bool = False) -> dict[str, Any]:
        if force_base and force_lora:
            raise ValueError("--force_base 和 --force_lora 不能同时使用")

        t0 = time.perf_counter()

        if force_base:
            use_lora = False
            dirty_prob = None
            route_source = "force_base"
        elif force_lora:
            use_lora = True
            dirty_prob = None
            route_source = "force_lora"
        elif self.routing_enabled and self.quality_predictor is not None:
            is_dirty, dirty_prob = self.quality_predictor.predict(audio)
            use_lora = is_dirty
            route_source = "quality_router"
        else:
            dirty_prob = 1.0
            use_lora = True
            route_source = "fallback_lora"

        self._set_lora(use_lora)
        items = self.transcribe_with_language(audio)

        elapsed = time.perf_counter() - t0
        vprint(
            f"[Qwen3ASR Route] use={'LoRA' if use_lora else 'Base'}, "
            f"route_source={route_source}, dirty_prob={dirty_prob}, elapsed={elapsed:.3f}s",
            self.verbose,
        )

        return {
            "audio": audio,
            "items": items,
            "use_lora": use_lora,
            "route_source": route_source,
            "dirty_prob": dirty_prob,
            "elapsed": elapsed,
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
    parser = argparse.ArgumentParser(description="Clean Qwen3-ASR router single-audio inference.")

    parser.add_argument("--audio", type=str, required=True, help="输入音频路径。")
    parser.add_argument("--model_path", type=str, default="ckpt/Mega-ASR/Qwen3-ASR-1.7B")

    parser.add_argument("--lora_b_dir", type=str, default="ckpt/Mega-ASR/A2S-SFT-lora/lora1")
    parser.add_argument("--lora_c_dir", type=str, default="ckpt/Mega-ASR/A2S-SFT-lora/lora2")
    parser.add_argument("--lora_d_dir", type=str, default="ckpt/Mega-ASR/A2S-SFT-lora/lora3")

    parser.add_argument(
        "--quality_model_dir",
        type=str,
        default="ckpt/Mega-ASR/router",
    )
    parser.add_argument(
        "--quality_checkpoint",
        type=str,
        default="ckpt/Mega-ASR/router/runs/exp_20260211_1layer/best_acc_model.pt",
    )
    parser.add_argument("--quality_threshold", type=float, default=0.5)

    parser.add_argument("--disable_quality_routing", action="store_true")
    parser.add_argument("--force_base", action="store_true")
    parser.add_argument("--force_lora", action="store_true")

    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--quality_device", type=str, default="cuda")
    parser.add_argument("--max_inference_batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument(
        "--output_format",
        type=str,
        default="tagged",
        choices=["tagged", "language_text", "text", "json"],
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="显示模型加载、路由和 warning 信息。")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    router = Qwen3ASRRouter(
        model_path=args.model_path,
        lora_b_dir=args.lora_b_dir,
        lora_c_dir=args.lora_c_dir,
        lora_d_dir=args.lora_d_dir,
        quality_model_dir=args.quality_model_dir,
        quality_checkpoint=args.quality_checkpoint,
        quality_threshold=args.quality_threshold,
        routing_enabled=not args.disable_quality_routing,
        device_map=args.device_map,
        quality_device=args.quality_device,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )

    result = router.infer_one(
        audio=args.audio,
        force_base=args.force_base,
        force_lora=args.force_lora,
    )

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
