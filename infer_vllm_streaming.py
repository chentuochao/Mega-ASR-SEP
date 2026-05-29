import sys
sys.path.append("src")

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from infer_vllm import build_vllm_kwargs, materialized_lora_dir, resolve_path, str2bool


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO = ROOT_DIR / "assets/example/F01_22GC010K_STR.wav"
DEFAULT_CKPT_DIR = ROOT_DIR / "ckpt/Mega-ASR"


def parse_args():
    parser = argparse.ArgumentParser(description="Mega-ASR vLLM streaming inference")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="audio file path")
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR, help="Mega-ASR ckpt root")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--step_ms", type=int, default=1000, help="streaming input step in milliseconds")
    parser.add_argument("--chunk_size_sec", type=float, default=2.0, help="Qwen3-ASR streaming chunk size")
    parser.add_argument("--unfixed_chunk_num", type=int, default=2)
    parser.add_argument("--unfixed_token_num", type=int, default=5)
    parser.add_argument(
        "--reset_interval_sec",
        type=float,
        default=120.0,
        help="finish and re-init streaming state after this many seconds; <=0 disables reset",
    )
    parser.add_argument(
        "--overlap_sec",
        type=float,
        default=2.0,
        help="audio overlap to replay into the next streaming state after reset",
    )
    parser.add_argument(
        "--context_chars",
        type=int,
        default=240,
        help="tail characters from committed transcript used as context for the next state",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32, help="small value is recommended for streaming")
    parser.add_argument(
        "--vllm_materialize_lora_force",
        type=str2bool,
        default=False,
        help="rebuild the materialized LoRA checkpoint even if the cache is fresh",
    )
    parser.add_argument(
        "--vllm_materialize_lora_device_map",
        default=None,
        help="device_map used only while materializing LoRA, e.g. cpu or cuda:0",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    return parser.parse_args()


def merge_transcript(prefix: str, suffix: str, max_overlap_chars: int = 240) -> str:
    """Append suffix to prefix while removing duplicated boundary text."""
    prefix = prefix or ""
    suffix = suffix or ""
    if not prefix:
        return suffix
    if not suffix:
        return prefix

    max_len = min(len(prefix), len(suffix), max_overlap_chars)
    for n in range(max_len, 0, -1):
        if prefix[-n:] == suffix[:n]:
            return prefix + suffix[n:]

    prefix_chars = [ch for ch in prefix[-max_overlap_chars:] if not ch.isspace()]
    suffix_chars = []
    suffix_cut_positions = []
    for idx, ch in enumerate(suffix):
        if ch.isspace():
            continue
        suffix_chars.append(ch)
        suffix_cut_positions.append(idx + 1)
        if len(suffix_chars) >= max_overlap_chars:
            break

    max_norm_len = min(len(prefix_chars), len(suffix_chars))
    for n in range(max_norm_len, 0, -1):
        if prefix_chars[-n:] == suffix_chars[:n]:
            return prefix + suffix[suffix_cut_positions[n - 1]:]

    return prefix + suffix


def make_context(text: str, max_chars: int) -> str:
    if max_chars <= 0 or not text:
        return ""
    return "Previous transcript context:\n" + text[-max_chars:]


def init_state(model, args, context: str = ""):
    kwargs = {
        "unfixed_chunk_num": args.unfixed_chunk_num,
        "unfixed_token_num": args.unfixed_token_num,
        "chunk_size_sec": args.chunk_size_sec,
    }
    if context:
        kwargs["context"] = context
    return model.init_streaming_state(**kwargs)


def finish_state(model, state, committed_text: str, label: Optional[str] = None) -> str:
    model.finish_streaming_transcribe(state)
    merged = merge_transcript(committed_text, state.text)
    if label:
        print(f"[{label}] language={state.language!r} text={merged!r}")
    return merged


def read_audio_16k(path: str | os.PathLike[str]) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr == 16000:
        return wav.astype(np.float32, copy=False)

    duration = wav.shape[0] / float(sr)
    target_len = int(round(duration * 16000))
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from MegaASR.model.megaASR import MegaASR

    audio = resolve_path(args.audio)
    ckpt_dir = resolve_path(args.ckpt_dir)
    vllm_kwargs = build_vllm_kwargs(args)

    model = MegaASR(
        model_path=ckpt_dir / "Qwen3-ASR-1.7B",
        lora_dir=ckpt_dir / "mega-asr-merged",
        routing_enabled=False,
        backend="vllm",
        vllm_apply_lora_on_load=True,
        vllm_materialized_lora_dir=materialized_lora_dir(ckpt_dir),
        vllm_materialize_lora_force=args.vllm_materialize_lora_force,
        vllm_materialize_lora_device_map=args.vllm_materialize_lora_device_map,
        max_new_tokens=args.max_new_tokens,
        **vllm_kwargs,
    )

    wav16k = read_audio_16k(audio)
    step = int(round(args.step_ms / 1000.0 * 16000))
    if step <= 0:
        raise ValueError("--step_ms must be positive.")
    if args.overlap_sec < 0:
        raise ValueError("--overlap_sec must be non-negative.")

    reset_samples = 0
    if args.reset_interval_sec > 0:
        reset_samples = int(round(args.reset_interval_sec * 16000))
        if reset_samples <= 0:
            raise ValueError("--reset_interval_sec is too small.")

    overlap_samples = int(round(args.overlap_sec * 16000))
    if reset_samples > 0 and overlap_samples >= reset_samples:
        raise ValueError("--overlap_sec must be smaller than --reset_interval_sec.")

    state = init_state(model, args)

    pos = 0
    call_id = 0
    segment_id = 1
    segment_start = 0
    committed_text = ""
    while pos < wav16k.shape[0]:
        seg = wav16k[pos:pos + step]
        pos += seg.shape[0]
        call_id += 1
        model.streaming_transcribe(seg, state)
        live_text = merge_transcript(committed_text, state.text)
        print(
            f"[call {call_id:03d} segment={segment_id:03d}] "
            f"language={state.language!r} text={live_text!r}"
        )

        if reset_samples > 0 and pos < wav16k.shape[0] and pos - segment_start >= reset_samples:
            committed_text = finish_state(
                model,
                state,
                committed_text,
                label=f"reset {segment_id:03d}",
            )
            segment_id += 1
            replay_start = max(0, pos - overlap_samples)
            segment_start = replay_start
            context = make_context(committed_text, args.context_chars)
            state = init_state(model, args, context=context)
            if replay_start < pos:
                model.streaming_transcribe(wav16k[replay_start:pos], state)
                replay_text = merge_transcript(committed_text, state.text)
                print(
                    f"[replay segment={segment_id:03d}] "
                    f"language={state.language!r} text={replay_text!r}"
                )

    committed_text = finish_state(model, state, committed_text)
    print(f"[final] language={state.language!r} text={committed_text!r}")


if __name__ == "__main__":
    main()
