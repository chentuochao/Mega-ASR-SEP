"""Check entry-point compatibility and the response backend's vLLM interface."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
import infer_vllm
from MegaASR.model.text_response import TextResponseModel, validate_text_model_config


class ResponseTests(unittest.TestCase):
    def test_text_capable_qwen_models_allow_fp8_and_asr_remains_rejected(self):
        for model_type, architecture in [('qwen3_5', 'Qwen3_5ForConditionalGeneration'),
                                         ('qwen3_5_moe', 'Qwen3_5MoeForConditionalGeneration')]:
            self.assertTrue(validate_text_model_config(dict(model_type=model_type, architectures=[architecture])))
        with self.assertRaises(ValueError):
            validate_text_model_config(dict(model_type='qwen3_asr', architectures=['Qwen3ASRForConditionalGeneration']))

    def test_original_asr_entry_point_keeps_model_and_adapter_options(self):
        calls = []

        class ASR:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def infer(self, audio, **kwargs):
                calls.append(dict(audio=audio, **kwargs))
                return dict(text='saved transcript')

        with patch.dict(sys.modules, {
            'MegaASR.model.megaASR': SimpleNamespace(MegaASR=ASR),
            'soundfile': SimpleNamespace(info=lambda path: SimpleNamespace(duration=1)),
        }), contextlib.redirect_stdout(io.StringIO()):
            infer_vllm.main(['--audio', '/tmp/audio.wav', '--ckpt_dir', '/tmp/asr-models'])
        self.assertEqual(calls[0]['model_path'], Path('/tmp/asr-models/Qwen3-ASR-1.7B'))
        self.assertEqual(calls[0]['lora_dir'], Path('/tmp/asr-models/mega-asr-merged'))
        self.assertEqual(calls[0]['backend'], 'vllm')
        self.assertTrue(calls[0]['vllm_apply_lora_on_load'])
        self.assertFalse(calls[0]['routing_enabled'])
        self.assertTrue(calls[1]['return_route'])

    def test_response_mode_requires_explicit_model_and_transcript(self):
        for arguments in (['--task', 'respond'], ['--task', 'respond', '--qa-model-path', '/tmp/model']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                infer_vllm.parse_args(arguments)
        args = infer_vllm.parse_args(['--task', 'respond', '--qa-model-path', '/tmp/model', '--transcript', 'Question'])
        self.assertFalse(args.qa_enable_thinking)
        self.assertEqual(args.qa_backend, 'vllm')

    def test_vllm_call_contract_without_gpu_inference(self):
        # This verifies API forwarding only. A GPU vLLM run is a separate acceptance check.
        calls = {}

        class LLM:
            def __init__(self, **kwargs):
                calls['load'] = kwargs

            def chat(self, messages, sampling, **kwargs):
                calls['chat'] = dict(messages=messages, sampling=sampling, **kwargs)
                return [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(
                    text='4', finish_reason='stop', token_ids=[3, 4])])]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'config.json').write_text(json.dumps(dict(architectures=['Qwen3ForCausalLM'])))
            (path/'model_provenance.json').write_text(json.dumps(dict(model='Qwen/Qwen3-1.7B', revision='a'*40)))
            with patch('platform.system', return_value='Linux'), patch.dict(sys.modules, {
                'vllm': SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs),
                'vllm.platforms': SimpleNamespace(current_platform=SimpleNamespace(device_name='cuda')),
            }):
                model = TextResponseModel(path, gpu_memory_utilization=.3)
                messages = [dict(role='user', content='What is two plus two?')]
                result = model.generate(messages, max_tokens=16)
            self.assertEqual(calls['load']['model'], str(path.resolve()))
            self.assertEqual(calls['load']['gpu_memory_utilization'], .3)
            self.assertEqual(calls['chat']['messages'], messages)
            self.assertEqual(calls['chat']['chat_template_kwargs'], dict(enable_thinking=False))
            self.assertEqual(calls['chat']['sampling'], dict(max_tokens=16, temperature=0))
            self.assertEqual(result['choices'][0]['message']['content'], '4')
            self.assertEqual(result['usage']['total_tokens'], 4)
            self.assertEqual(model.metadata()['model_revision'], 'a'*40)
            for options in (dict(max_tokens=True), dict(temperature=float('nan'))):
                with self.assertRaises(ValueError):
                    model.generate(messages, **options)

    def test_asr_checkpoint_cannot_be_used_as_a_response_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'config.json').write_text(json.dumps(dict(architectures=['Qwen3ASRForConditionalGeneration'])))
            with self.assertRaisesRegex(ValueError, 'text causal LM'):
                TextResponseModel(path)


if __name__ == '__main__':
    unittest.main()
