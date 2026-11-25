"""Check grading denominators, protocol isolation, and model lifetime barriers."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from local_qa_grader import grade_records
from qa_pipeline import fingerprint
from serial_model_stages import run_model_stage, wait_gpu_idle
from test_voicebench_sdqa import fixtures, RESOURCES


class Grader:
    def __init__(self, votes=('Yes', 'No', 'Yes'), finish='stop'):
        self.votes, self.finish, self.calls = votes, finish, []

    def metadata(self):
        return dict(model='Qwen/Qwen3.8-27B-FP8', model_revision='a' * 40)

    def generate(self, messages, **kwargs):
        self.calls.append(dict(messages=messages, settings=kwargs))
        return dict(choices=[dict(message=dict(content=value), finish_reason=self.finish) for value in self.votes])


class GradingTests(unittest.TestCase):
    def test_three_votes_reuse_rejections_and_separate_official_scores(self):
        _, rows, records = fixtures()
        for record in records:
            for outcome in record['qa'].values():
                outcome['voicebench'] = dict(panda=True, gpt=None)
        model = Grader()
        with tempfile.TemporaryDirectory() as tmp:
            result = grade_records(rows, records, ['judge', 'always_respond'], model, Path(tmp) / 'votes.jsonl')
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(model.calls[0]['settings'], dict(n=3, temperature=.5, top_p=.95, max_tokens=1024))
        self.assertEqual(result['policies']['judge']['task_success_rate'], 2 / 3)
        self.assertEqual(result['policies']['judge']['accuracy_percent'], 100)
        self.assertIsNone(records[2]['qa']['judge']['local_grader']['correct'])
        self.assertIsNone(records[0]['qa']['judge']['voicebench']['gpt'])
        self.assertTrue(records[0]['qa']['judge']['voicebench']['panda'])

    def test_invalid_or_truncated_votes_do_not_receive_correct_credit(self):
        for model in (Grader(('Yes', 'No', 'Yes, correct')), Grader(finish='length')):
            _, rows, records = fixtures()
            with tempfile.TemporaryDirectory() as tmp:
                result = grade_records(rows, records, ['judge'], model, Path(tmp) / 'votes.jsonl')
                raw = json.loads((Path(tmp) / 'votes.jsonl').read_text().splitlines()[0])
                self.assertIsNotNone(raw['result']['completion'])
            self.assertEqual(result['status'], 'grading_errors')
            self.assertEqual(result['policies']['judge']['n_errors'], 2)
            self.assertIsNone(result['policies']['judge']['accuracy_percent'])

    def test_identity_mismatch_prevents_grading(self):
        _, rows, records = fixtures()
        records[0]['reference'] = 'different question'
        model = Grader()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            grade_records(rows, records, ['judge'], model, Path(tmp) / 'votes.jsonl')
        self.assertEqual(model.calls, [])

    def test_gpu_barrier_rejects_resident_model_and_failed_stage_is_reported(self):
        active = SimpleNamespace(stdout='GPU-allocated, 123, 35000\n')
        with patch('serial_model_stages.subprocess.run', return_value=active), \
             patch('serial_model_stages.time.monotonic', side_effect=[0, 31]):
            with self.assertRaisesRegex(RuntimeError, 'next model was not loaded'):
                wait_gpu_idle({'GPU-allocated'})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(subprocess.CalledProcessError):
                run_model_stage('failed', [sys.executable, '-c', 'raise SystemExit(7)'],
                                root / 'failed.log', root / 'events.jsonl')
            event = json.loads((root / 'events.jsonl').read_text())
            self.assertEqual(event['returncode'], 7)


@unittest.skipUnless(importlib.util.find_spec('qa_metrics'), 'Install the pinned VoiceBench scoring environment')
class SerialPipelineTests(unittest.TestCase):
    def test_real_worker_processes_never_overlap_response_and_grading_models(self):
        _, rows, _ = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, stage = root / 'manifest.jsonl', root / 'asr'
            stage.mkdir()
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            predictions = [dict(index=index, sample_id=row['sample_id'], reference=row['text'],
                audio_sep=row['audio_sep'], audio_mix=row['audio_mix'], generated_transcription=row['text'],
                effective_action='REJECT' if index == 2 else 'RESPOND', parsed_action='RESPOND', action_valid=True,
                wer=0., label_wer=0., stage_seconds=dict(asr=.1, judge=.1)) for index, row in enumerate(rows)]
            (stage / 'persample.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in predictions))
            (stage / 'summary.json').write_text(json.dumps(dict(mode='pipeline', n_samples=3, wer_threshold=.75,
                manifest_sha256=fingerprint(manifest), predictions_sha256=fingerprint(stage / 'persample.jsonl'))))
            for name, architecture in [('qa', 'Qwen3_5MoeForConditionalGeneration'), ('grader', 'Qwen3_5ForConditionalGeneration')]:
                folder = root / name
                folder.mkdir()
                (folder / 'config.json').write_text(json.dumps(dict(
                    model_type='qwen3_5_moe' if name == 'qa' else 'qwen3_5', architectures=[architecture])))
                (folder / 'model_provenance.json').write_text(json.dumps(dict(revision='a'*40)))
            wrapper = root / 'fake-vllm-python'
            wrapper.write_text('#!' + sys.executable + '\n' + r'''
import atexit, importlib.metadata, json, os, pathlib, runpy, sys
from types import SimpleNamespace
lock=pathlib.Path(os.environ['MODEL_TEST_LOCK'])
original=importlib.metadata.version
importlib.metadata.version=lambda name: 'test' if name in ('vllm','torch','transformers') else original(name)
class LLM:
    def __init__(self, **kwargs):
        assert not lock.exists(), 'Models overlapped'
        lock.write_text(str(os.getpid()))
        atexit.register(lambda:lock.unlink())
        assert kwargs['language_model_only'] is True
        assert kwargs['tensor_parallel_size'] == 2
        assert kwargs['dtype'] == 'float16'
        self.name=pathlib.Path(kwargs['model']).name
    def chat(self, messages, sampling, **kwargs):
        n=sampling.get('n',1)
        with pathlib.Path(os.environ['MODEL_TEST_CALLS']).open('a') as handle:
            handle.write(json.dumps(dict(model=self.name,n=n))+'\n')
        return [SimpleNamespace(prompt_token_ids=[1],outputs=[SimpleNamespace(
            text='Yes' if n==3 else '4',finish_reason='stop',token_ids=[2]) for _ in range(n)])]
sys.modules['vllm']=SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs:kwargs)
sys.modules['vllm.platforms']=SimpleNamespace(current_platform=SimpleNamespace(device_name='cuda'))
sys.path.insert(0,str(pathlib.Path(sys.argv[1]).parent))
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
''')
            wrapper.chmod(0o700)
            command = [sys.executable, str(ROOT / 'scripts/eval_qa_pipeline.py'), '--manifest', str(manifest),
                '--asr-results-dir', str(stage), '--out-dir', str(root / 'out'),
                '--qa-model-path', str(root / 'qa'), '--qa-python', str(wrapper),
                '--local-grader-model-path', str(root / 'grader'), '--local-grader-python', str(wrapper),
                '--qa-tensor-parallel-size', '2', '--qa-dtype', 'float16',
                '--local-grader-tensor-parallel-size', '2', '--local-grader-dtype', 'float16',
                '--voicebench-resources', str(RESOURCES), '--voicebench-gpt-mode', 'off',
                '--policies', 'judge', 'always_respond']
            environment = dict(os.environ, MODEL_TEST_LOCK=str(root / 'model.lock'),
                MODEL_TEST_CALLS=str(root/'calls.jsonl'), PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run(command, env=environment, text=True, capture_output=True)
            if result.returncode:
                logs = '\n'.join(p.read_text() for p in (root / 'out').glob('*.log'))
                self.fail(result.stderr + '\n' + logs)
            summary = json.loads((root / 'out/summary.json').read_text())
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['local_grading']['policies']['judge']['task_success_rate'], 2 / 3)
            self.assertIsNone(summary['voicebench']['policies']['judge']['official_answered']['gpt'])
            events = summary['model_stages']
            self.assertEqual([e['stage'] for e in events], ['qa_respond', 'qa_correctness_grade'])
            self.assertNotEqual(events[0]['pid'], events[1]['pid'])
            self.assertLessEqual(events[0]['ended_utc'], events[1]['started_utc'])
            self.assertFalse((root / 'model.lock').exists())
            prefix=root/'resume'
            prefix.mkdir()
            (prefix/'qa_worker_config.json').write_bytes((root/'out/qa_worker_config.json').read_bytes())
            saved=(root/'out/generation.jsonl').read_text().splitlines()[:2]
            (prefix/'generation.jsonl').write_text('\n'.join(saved)+'\n')
            before=len((root/'calls.jsonl').read_text().splitlines())
            resumed_command=list(command)
            resumed_command[resumed_command.index('--out-dir')+1]=str(root/'resumed-out')
            resumed_command.extend(['--qa-resume-dir',str(prefix)])
            resumed=subprocess.run(resumed_command,env=environment,text=True,capture_output=True)
            self.assertEqual(resumed.returncode,0,resumed.stderr+'\n'+''.join(
                p.read_text() for p in (root/'resumed-out').glob('*.log')))
            resumed_summary=json.loads((root/'resumed-out/summary.json').read_text())
            self.assertEqual(resumed_summary['qa_resume']['n_records'],2)
            self.assertEqual((root/'resumed-out/generation.jsonl').read_text().splitlines()[:2],saved)
            new_calls=[json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()[before:]]
            self.assertEqual(sum(call['model']=='qa' for call in new_calls),1)
            self.assertEqual(sum(call['model']=='grader' for call in new_calls),3)
            for policy in summary['qa']:
                for key in ('n_samples','n_answered','n_correct','n_errors','n_truncated','task_success_rate'):
                    self.assertEqual(resumed_summary['qa'][policy][key],summary['qa'][policy][key])


if __name__ == '__main__':
    unittest.main()
