"""Text response inference for infer_vllm.py and the ASR/judge QA endpoint."""
import json
import math
import os
from pathlib import Path
import platform
import time


def validate_text_model_config(config):
    """Accept causal text models and the supported text-capable Qwen VL models."""
    architectures = config.get('architectures', [])
    if any('ASR' in name or 'Omni' in name for name in architectures):
        raise ValueError('QA needs a text causal LM; do not pass an ASR checkpoint')
    if any('CausalLM' in name for name in architectures):
        return False
    supported = {'qwen3_5': 'Qwen3_5ForConditionalGeneration',
                 'qwen3_5_moe': 'Qwen3_5MoeForConditionalGeneration'}
    if supported.get(config.get('model_type')) in architectures:
        return True
    raise ValueError('QA needs a text causal LM or a supported Qwen text-capable vision-language model')


class TextResponseModel:
    def __init__(self, model_path, *, backend='vllm', device='cpu', enable_thinking=False,
                 model_name=None, max_model_len=8192, **vllm_kwargs):
        self.path = Path(model_path).expanduser().resolve()
        if not (self.path/'config.json').is_file():
            raise ValueError('Use a downloaded local text model directory containing config.json')
        config = json.loads((self.path/'config.json').read_text())
        self.multimodal_architecture = validate_text_model_config(config)
        self.quantization_config = config.get('quantization_config')
        provenance = self.path/'model_provenance.json'
        self.provenance = json.loads(provenance.read_text()) if provenance.exists() else {}
        self.name = model_name or self.provenance.get('model') or self.path.name
        self.backend, self.device = backend, device
        self.enable_thinking, self.max_model_len = enable_thinking, max_model_len
        if max_model_len < 2:
            raise ValueError('max_model_len must be at least two tokens')
        if backend == 'vllm':
            # Keep auxiliary compiler/configuration files with an explicitly
            # selected runtime cache, including on shared cluster filesystems.
            runtime_cache = os.environ.get('VLLM_CACHE_ROOT')
            if runtime_cache:
                for variable, subdirectory in (
                    ('VLLM_CONFIG_ROOT', 'config'),
                    ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'),
                    ('FLASHINFER_WORKSPACE_BASE', 'flashinfer'),
                    ('TVM_FFI_CACHE_DIR', 'tvm-ffi'),
                ):
                    os.environ.setdefault(variable, str(Path(runtime_cache)/subdirectory))
            try:
                from vllm import LLM
                from vllm.platforms import current_platform
            except ImportError as exc:
                requirement = 'vLLM Metal' if platform.system() == 'Darwin' else 'vLLM'
                raise RuntimeError(f'Install {requirement} in the selected inference environment') from exc
            self.engine_platform = type(current_platform).__name__
            self.vllm_device_name = getattr(current_platform, 'device_name', 'unknown')
            # Metal uses vLLM's CPU control path while executing the model on the Apple GPU.
            self.device = 'metal' if self.engine_platform == 'MetalPlatform' else self.vllm_device_name
            self.engine_options = dict(max_model_len=max_model_len, **vllm_kwargs)
            if self.multimodal_architecture:
                self.engine_options.setdefault('language_model_only', True)
            self.model = LLM(model=str(self.path), **self.engine_options)
        elif backend == 'transformers':
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_class = AutoModelForCausalLM
            if self.multimodal_architecture:
                from transformers import AutoModelForImageTextToText
                model_class = AutoModelForImageTextToText
            dtype = torch.float32 if device == 'cpu' else torch.bfloat16 if device == 'cuda' else torch.float16
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            self.model = model_class.from_pretrained(
                self.path, local_files_only=True, dtype=dtype,
                attn_implementation='sdpa').to(device).eval()
            self.parameter_device = str(next(self.model.parameters()).device)
            self.parameter_dtype = str(next(self.model.parameters()).dtype)
            if next(self.model.parameters()).device.type != torch.device(device).type:
                raise RuntimeError('The response model did not load on the selected device')
        else:
            raise ValueError('Select vllm or transformers explicitly')

    def metadata(self):
        result = dict(model=self.name, model_path=str(self.path), backend=self.backend,
                    device=self.device,
                    model_revision=self.provenance.get('revision'), max_model_len=self.max_model_len,
                    chat_template_kwargs=dict(enable_thinking=self.enable_thinking))
        result['text_only'] = True
        result['quantization_config'] = self.quantization_config
        if self.backend == 'transformers':
            result.update(parameter_device=self.parameter_device, parameter_dtype=self.parameter_dtype)
        else:
            result['engine_platform'] = self.engine_platform
            result['engine_options'] = self.engine_options
            result['vllm_device_name'] = self.vllm_device_name
            if self.device == 'metal':
                result['metal_compute_backend'] = os.environ.get('VLLM_METAL_BACKEND', 'mlx')
        return result

    def generate(self, messages, *, max_tokens=256, temperature=0, n=1, top_p=None):
        if (not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant')
            or not isinstance(message.get('content'), str) for message in messages)):
            raise ValueError('Expected nonempty text chat messages')
        if (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1
                or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0):
            raise ValueError('Invalid response token limit or temperature')
        if self.enable_thinking and temperature == 0:
            raise ValueError('Thinking mode requires nonzero temperature; use non-thinking for the default QA recipe')
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError('n must be a positive integer')
        if top_p is not None and (not isinstance(top_p, (int, float)) or not 0 < top_p <= 1):
            raise ValueError('top_p must be in (0, 1]')
        if self.backend != 'vllm' and n != 1:
            raise ValueError('Multiple grading votes require the vLLM backend')
        started = time.perf_counter()
        template = dict(enable_thinking=self.enable_thinking)
        if self.backend == 'vllm':
            from vllm import SamplingParams
            sampling = dict(max_tokens=max_tokens, temperature=temperature)
            if n != 1:
                sampling['n'] = n
            if top_p is not None:
                sampling['top_p'] = top_p
            outputs = self.model.chat(messages, SamplingParams(**sampling),
                                      chat_template_kwargs=template, use_tqdm=False)
            request = outputs[0]
            choices = [dict(index=index, message=dict(role='assistant', content=answer.text),
                            finish_reason=answer.finish_reason) for index, answer in enumerate(request.outputs)]
            prompt_tokens = len(request.prompt_token_ids)
            completion_tokens = sum(len(answer.token_ids) for answer in request.outputs)
        else:
            import torch
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors='pt', **template).to(self.device)
            prompt_tokens = inputs['input_ids'].shape[1]
            if prompt_tokens + max_tokens > self.max_model_len:
                raise ValueError('Prompt and output budget exceed max_model_len; input was not truncated')
            options = dict(do_sample=temperature > 0, max_new_tokens=max_tokens,
                           pad_token_id=self.tokenizer.eos_token_id)
            if temperature > 0:
                options['temperature'] = temperature
                if top_p is not None:
                    options['top_p'] = top_p
            else:
                options.update(temperature=None, top_p=None, top_k=None)
            with torch.inference_mode():
                sequence = self.model.generate(**inputs, **options)[0, prompt_tokens:]
            tokens = sequence.tolist()
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            finish = 'stop' if tokens and tokens[-1] in eos else 'length'
            content = self.tokenizer.decode(tokens, skip_special_tokens=True)
            completion_tokens = len(tokens)
            choices = [dict(index=0, message=dict(role='assistant', content=content), finish_reason=finish)]
        return dict(model=self.name, choices=choices, usage=dict(prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens, total_tokens=prompt_tokens+completion_tokens),
                    response_seconds=time.perf_counter()-started)


def serve_response_model(responder, *, host='127.0.0.1', port=8000, ready_file=None, max_tokens=256):
    """Expose the text-only chat-completions subset used by our serial QA runner."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('This QA helper binds to localhost; use an SSH tunnel for remote clients')
    ready = Path(ready_file) if ready_file else None
    if ready and ready.exists():
        raise FileExistsError('Select a new --qa-ready-file')

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, result):
            data = json.dumps(result, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.send_json(200, dict(status='ready', max_tokens=max_tokens, **responder.metadata()))
            else:
                self.send_json(404, dict(error='Unknown endpoint'))

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_json(404, dict(error='Unknown endpoint'))
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 1024*1024:
                    raise ValueError('Invalid request length')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or payload.get('model') != responder.name or payload.get('stream', False):
                    raise ValueError('Use the served model name and non-streaming text requests')
                requested_tokens = payload.get('max_tokens', max_tokens)
                if not isinstance(requested_tokens, int) or requested_tokens > max_tokens:
                    raise ValueError('Requested output exceeds the configured server token limit')
                result = responder.generate(payload.get('messages'), max_tokens=requested_tokens,
                                            temperature=payload.get('temperature', 0))
                self.send_json(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json(400, dict(error=str(exc)))
            except Exception:
                import traceback
                traceback.print_exc()
                self.send_json(500, dict(error='Model inference failed; inspect the server log'))

    server = HTTPServer((host, port), Handler)
    metadata = dict(endpoint=f'http://{host}:{server.server_port}/v1/chat/completions',
                    max_tokens=max_tokens, **responder.metadata())
    try:
        if ready:
            ready.parent.mkdir(parents=True, exist_ok=True)
            with ready.open('x') as handle:
                json.dump(metadata, handle, indent=2)
                handle.write('\n')
        print(json.dumps(dict(status='ready', **metadata)), flush=True)
        server.serve_forever()
    finally:
        server.server_close()
