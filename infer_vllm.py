import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))
DEFAULT_AUDIO = ROOT_DIR / "assets/example/F01_22GC010K_STR.wav"
DEFAULT_CKPT_DIR = ROOT_DIR / "ckpt/Mega-ASR"
DEFAULT_MATERIALIZED_LORA_NAME = "mega-asr-vllm-materialized"
DEFAULT_VLLM_SMALL_GPU_KWARGS = {
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 2048,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("1", "true", "yes", "y")


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def materialized_lora_dir(ckpt_dir):
    return Path(ckpt_dir) / DEFAULT_MATERIALIZED_LORA_NAME


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mega-ASR inference and transcript response inference")
    parser.add_argument("--task", choices=["asr", "respond", "serve-qa"], default="asr")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="audio file path")
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR, help="Mega-ASR ckpt root")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--qa-model-path", type=Path, help="Downloaded text model directory; separate from the ASR checkpoint")
    parser.add_argument("--qa-model-name", help="Served model name; defaults to recorded download provenance")
    parser.add_argument("--qa-backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--qa-device", choices=["cpu", "mps", "cuda"], default="cpu", help="Transformers check device")
    parser.add_argument("--qa-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qa-max-tokens", type=int, default=256)
    parser.add_argument("--qa-temperature", type=float, default=0)
    parser.add_argument("--qa-host", default="127.0.0.1")
    parser.add_argument("--qa-port", type=int, default=8000)
    parser.add_argument("--qa-ready-file", type=Path, help="New readiness JSON file for --task serve-qa")
    parser.add_argument("--qa-output", type=Path, help="New response JSON file for --task respond")
    transcript = parser.add_mutually_exclusive_group()
    transcript.add_argument("--transcript", help="ASR-generated text to answer; no reference answer is supplied")
    transcript.add_argument("--transcript-file", type=Path)
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
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="vLLM max_model_len. Lower this on small GPUs, e.g. 8192.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=None,
        help="vLLM max_num_seqs. Use 1 on small GPUs.",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help="vLLM max_num_batched_tokens. Lower this on small GPUs.",
    )
    args = parser.parse_args(argv)
    if args.task != "asr":
        if args.qa_model_path is None:
            parser.error("Response tasks require --qa-model-path")
        if args.task == "respond" and args.transcript is None and args.transcript_file is None:
            parser.error("--task respond requires --transcript or --transcript-file")
        if args.qa_max_tokens < 1 or not 0 <= args.qa_port <= 65535:
            parser.error("Invalid QA output limit or port")
    return args


def build_vllm_kwargs(args):
    kwargs = {}
    for name, default in DEFAULT_VLLM_SMALL_GPU_KWARGS.items():
        value = getattr(args, name)
        kwargs[name] = default if value is None else value
    return kwargs


def run_response(args):
    from MegaASR.model.text_response import TextResponseModel, serve_response_model
    kwargs = build_vllm_kwargs(args)
    max_model_len = kwargs.pop("max_model_len")
    if args.task == "respond" and args.qa_output and args.qa_output.exists():
        raise FileExistsError("Select a new --qa-output path")
    started = time.perf_counter()
    responder = TextResponseModel(args.qa_model_path, backend=args.qa_backend, device=args.qa_device,
        enable_thinking=args.qa_enable_thinking, model_name=args.qa_model_name,
        max_model_len=max_model_len, **kwargs)
    load_seconds = time.perf_counter() - started
    if args.task == "serve-qa":
        serve_response_model(responder, host=args.qa_host, port=args.qa_port,
                             ready_file=args.qa_ready_file, max_tokens=args.qa_max_tokens)
        return 0
    transcript = args.transcript if args.transcript is not None else args.transcript_file.read_text()
    if not transcript.strip():
        raise ValueError("The candidate transcript is empty")
    sys.path.insert(0, str(ROOT_DIR / "scripts"))
    from qa_pipeline import messages_for
    result = responder.generate(messages_for(transcript, 'unscored'),
                                max_tokens=args.qa_max_tokens, temperature=args.qa_temperature)
    result.update(configuration=responder.metadata(), transcript=transcript, load_seconds=load_seconds)
    if args.qa_output:
        args.qa_output.parent.mkdir(parents=True, exist_ok=True)
        with args.qa_output.open('x') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
    print(json.dumps(result, indent=2))
    return 0 if result['choices'][0]['finish_reason'] == 'stop' and result['choices'][0]['message']['content'].strip() else 2


def main(argv=None):
    args = parse_args(argv)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.task != "asr":
        return run_response(args)

    from MegaASR.model.megaASR import MegaASR
    import soundfile as sf

    audio = resolve_path(args.audio)
    ckpt_dir = resolve_path(args.ckpt_dir)
    vllm_kwargs = build_vllm_kwargs(args)

    audio_duration = sf.info(str(audio)).duration

    t0 = time.perf_counter()
    model = MegaASR(
        model_path=ckpt_dir / "Qwen3-ASR-1.7B",
        lora_dir=ckpt_dir / "mega-asr-merged",
        routing_enabled=False,
        backend="vllm",
        vllm_apply_lora_on_load=True,
        vllm_materialized_lora_dir=materialized_lora_dir(ckpt_dir),
        vllm_materialize_lora_force=args.vllm_materialize_lora_force,
        vllm_materialize_lora_device_map=args.vllm_materialize_lora_device_map,
        **vllm_kwargs,
    )
    t1 = time.perf_counter()
    result = model.infer(audio, return_route=True)
    t2 = time.perf_counter()

    load_time = t1 - t0
    infer_time = t2 - t1
    rtf = infer_time / audio_duration if audio_duration > 0 else 0.0
    rtfx = 1.0 / rtf if rtf > 0 else 0.0

    print(result)
    print(f"[timing] load={load_time:.2f}s  infer={infer_time:.2f}s  total={t2-t0:.2f}s")
    print(f"[rtf]    audio={audio_duration:.2f}s  RTF={rtf:.4f}  RTFx={rtfx:.2f}x")


if __name__ == "__main__":
    raise SystemExit(main())
