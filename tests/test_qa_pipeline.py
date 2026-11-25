"""Exercise routing, answer isolation, metrics, real HTTP, and the pipeline CLI."""
import contextlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from qa_pipeline import (ChatClient, DirectChatClient, evaluate_qa, fingerprint, score_response,
                         summarize_qa, validate_rows, compare_rejections)
from prepare_qa_benchmark import prepare


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


@contextlib.contextmanager
def server():
    requests = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(payload)
            question = payload['messages'][-1]['content']
            if question == 'http error':
                self.send_error(500)
                return
            answer = '4' if question == 'accepted question' else 'wrong'
            finish = 'length' if question == 'truncated' else 'stop'
            data = json.dumps(dict(model='test-lm', choices=[dict(message=dict(content=answer), finish_reason=finish)])).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions', requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def fixture():
    rows, predictions = [], []
    for index, (hyp, action, parsed) in enumerate([
        ('accepted question', 'RESPOND', 'RESPOND'),
        ('rejected question', 'REJECT', 'REJECT'),
        ('invalid question', 'REJECT', None),
    ]):
        row = dict(sample_id=str(index), audio_sep=f'{index}_sep.wav', audio_mix=f'{index}_mix.wav',
                   text=f'PRIVATE_REFERENCE_{index}', qa_task='exact_match', qa_answers=['4'],
                   benchmark='test', benchmark_record={'prompt':f'PRIVATE_REFERENCE_{index}'})
        rows.append(row)
        predictions.append(dict(index=index, sample_id=str(index), audio_sep=row['audio_sep'], audio_mix=row['audio_mix'],
            reference=row['text'], generated_transcription=hyp, effective_action=action,
            parsed_action=parsed, action_valid=parsed is not None,
            stage_seconds=dict(asr=2., judge=1.)))
    return rows, predictions


class PipelineTests(unittest.TestCase):
    def test_direct_client_routing_and_completion_errors(self):
        requests = []

        class Responder:
            finish, content = 'stop', '4'

            def metadata(self):
                return dict(model='direct-test', backend='test')

            def generate(self, messages, **kwargs):
                requests.append(messages)
                return dict(model='direct-test', choices=[dict(
                    message=dict(content=self.content), finish_reason=self.finish)])

        model = Responder()
        client = DirectChatClient(model)
        rows, predictions = fixture()
        with tempfile.TemporaryDirectory() as directory:
            records = evaluate_qa(rows, predictions, client, ['judge'], Path(directory)/'direct.jsonl')
        self.assertEqual(len(requests), 1)
        self.assertNotIn('PRIVATE_REFERENCE', json.dumps(requests))
        self.assertEqual(summarize_qa(records, ['judge'])['judge']['task_success_rate'], 1/3)
        self.assertEqual(client.metadata()['transport'], 'in_process')
        model.finish = 'length'
        self.assertEqual(client.generate(requests[0])['error'], 'incomplete_or_unknown_finish_reason')
        model.content = ''
        self.assertEqual(client.generate(requests[0])['error'], 'ValueError')

    def test_loopback_service_bypasses_cluster_http_proxy(self):
        with server() as (endpoint, requests):
            proxy = 'http://127.0.0.1:1'
            with patch.dict(os.environ, dict(http_proxy=proxy, HTTP_PROXY=proxy, no_proxy='', NO_PROXY='')):
                client = ChatClient(endpoint, 'test-lm')
                result = client.generate([dict(role='user',content='accepted question')])
            self.assertIsNone(result['error'])
            self.assertEqual(result['response'],'4')
            self.assertEqual(len(requests),1)
            self.assertTrue(client.metadata()['environment_proxy_bypassed_for_loopback'])

    def test_routing_reference_isolation_denominators_and_baselines(self):
        rows, predictions = fixture()
        validate_rows(rows)
        with tempfile.TemporaryDirectory() as tmp, server() as (endpoint, requests):
            client = ChatClient(endpoint, 'test-lm')
            records = evaluate_qa(rows, predictions, client, ['judge'], Path(tmp)/'pred.jsonl')
            self.assertEqual(len(requests), 1)
            self.assertNotIn('PRIVATE_REFERENCE', json.dumps(requests))
            self.assertNotIn('qa_answers', json.dumps(requests))
            result = summarize_qa(records, ['judge'])['judge']
            self.assertEqual(result['task_success_rate'], 1/3)
            self.assertEqual(result['accuracy_on_answered'], 1)
            self.assertEqual(result['answer_coverage'], 1/3)
            self.assertAlmostEqual(result['pipeline_compute_seconds']['mean'],
                                   3 + records[0]['qa']['judge']['latency_seconds']/3)
            baseline = evaluate_qa(rows, predictions, client, ['judge', 'always_respond', 'reference_text'],
                                   Path(tmp)/'baseline.jsonl')
            self.assertTrue(baseline[0]['qa']['always_respond']['reused_request'])
            self.assertEqual(baseline[0]['qa']['always_respond']['pipeline_compute_seconds'],
                             2 + baseline[0]['qa']['always_respond']['latency_seconds'])
            result = summarize_qa(baseline, ['always_respond'])['always_respond']
            self.assertEqual(result['n_wrong_answers'], 2)
            self.assertEqual(compare_rejections(baseline, ['judge', 'always_respond'])['rejected_and_always_respond_wrong'], 2)
            self.assertEqual(result['task_success_rate'], 1/3)
            # Rejected, invalid, and unscored populations must not appear as correct answers.
            for pred in predictions:
                pred['effective_action'] = 'REJECT'
            records = evaluate_qa(rows, predictions, client, ['judge'], Path(tmp)/'reject.jsonl')
            result = summarize_qa(records, ['judge'])['judge']
            self.assertIsNone(result['accuracy_on_answered'])
            self.assertEqual(result['task_success_rate'], 0)

    def test_errors_truncation_strict_scoring_and_join(self):
        rows, predictions = fixture()
        with tempfile.TemporaryDirectory() as tmp, server() as (endpoint, requests):
            client = ChatClient(endpoint, 'test-lm')
            for index, text in enumerate(('truncated', 'http error')):
                predictions[0]['generated_transcription'] = text
                records = evaluate_qa(rows, predictions, client, ['judge'], Path(tmp)/f'{index}.jsonl')
                result = summarize_qa(records, ['judge'])['judge']
                self.assertEqual(result['n_errors'], 1)
                self.assertEqual(result['n_answered'], 0)
                self.assertEqual(result['task_success_rate'], 0)
            before = len(requests)
            predictions[2]['sample_id'] = 'different sample'
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                evaluate_qa(rows, predictions, client, ['judge'], Path(tmp)/'bad.jsonl')
            self.assertEqual(len(requests), before)
        mcq = dict(qa_task='multiple_choice', qa_choices=['A', 'B'], qa_answers=['A'])
        self.assertTrue(score_response('(A).', mcq)['correct'])
        for text in ('A or B', 'I think A', '', 'C'):
            self.assertFalse(score_response(text, mcq)['correct'])
        self.assertFalse(score_response('-1', dict(qa_task='exact_match', qa_answers=['1']))['correct'])
        self.assertIsNone(score_response('anything', dict(qa_task='unscored'))['correct'])

    def test_benchmark_join_requires_verified_text_and_answers(self):
        rows, _ = fixture()
        pairs = [dict(rows[0], source_id='0')]
        source = [dict(prompt=rows[0]['text'], reference='A')]
        converted = prepare(source, pairs, 'voicebench', 'openbookqa')
        self.assertEqual(converted[0]['qa_answers'], ['A'])
        self.assertNotIn('prompt', converted[0])
        self.assertEqual(converted[0]['benchmark_record']['prompt'], rows[0]['text'])
        with self.assertRaisesRegex(ValueError, 'differs'):
            prepare([dict(source[0], prompt='different')], pairs, 'voicebench', 'openbookqa')
        bba = [dict(id=0, official_answer='valid', category='formal_fallacies')]
        converted = prepare(bba, pairs, 'big_bench_audio', 'default', 'id')
        self.assertEqual(converted[0]['qa_answers'], ['valid'])
        del pairs[0]['text']
        with self.assertRaisesRegex(ValueError, 'text'):
            prepare(bba, pairs, 'big_bench_audio', 'default', 'id')

    def test_real_benchmark_preparation_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Metadata preparation checks file identity, not waveform decoding.
            (root/'sep.wav').touch()
            (root/'mix.wav').touch()
            write_jsonl(root/'source.jsonl', [dict(prompt='Question with spoken options', reference='A')])
            write_jsonl(root/'pairs.jsonl', [dict(sample_id='q0', source_id='0', audio_sep='sep.wav', audio_mix='mix.wav')])
            command = [sys.executable, str(ROOT/'scripts/prepare_qa_benchmark.py'), '--benchmark', 'voicebench',
                       '--source-jsonl', str(root/'source.jsonl'), '--paired-manifest', str(root/'pairs.jsonl'),
                       '--out-jsonl', str(root/'qa.jsonl')]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            row = json.loads((root/'qa.jsonl').read_text())
            self.assertEqual(row['audio_sep'], str((root/'sep.wav').resolve()))
            self.assertEqual(row['qa_answers'], ['A'])
            self.assertEqual(json.loads((root/'qa.jsonl.meta.json').read_text())['manifest_sha256'], fingerprint(root/'qa.jsonl'))
            original = (root/'qa.jsonl').read_bytes()
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual((root/'qa.jsonl').read_bytes(), original)

    def test_real_cli_replay_error_exit_and_checksum(self):
        rows, predictions = fixture()
        with tempfile.TemporaryDirectory() as tmp, server() as (endpoint, requests):
            root = Path(tmp)
            manifest, stage = root/'manifest.jsonl', root/'asr'
            stage.mkdir()
            write_jsonl(manifest, rows)
            write_jsonl(stage/'persample.jsonl', predictions)
            summary = dict(mode='pipeline', n_samples=3, manifest_sha256=fingerprint(manifest),
                           predictions_sha256=fingerprint(stage/'persample.jsonl'))
            (stage/'summary.json').write_text(json.dumps(summary))
            command = [sys.executable, str(ROOT/'scripts/eval_qa_pipeline.py'), '--manifest', str(manifest),
                       '--asr-results-dir', str(stage), '--qa-endpoint', endpoint, '--qa-model', 'test-lm']
            original = manifest.read_bytes()
            run = subprocess.run(command+['--out-dir', str(root/'out')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(result['qa']['judge']['task_success_rate'], 1/3)
            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(len((root/'out/benchmark-judge-answered.jsonl').read_text().splitlines()), 1)
            # Provider failures leave a report and a nonzero process result.
            predictions[0]['generated_transcription'] = 'http error'
            write_jsonl(stage/'persample.jsonl', predictions)
            summary['predictions_sha256'] = fingerprint(stage/'persample.jsonl')
            (stage/'summary.json').write_text(json.dumps(summary))
            run = subprocess.run(command+['--out-dir', str(root/'error')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertEqual(json.loads((root/'error/summary.json').read_text())['status'], 'qa_errors')
            before = len(requests)
            (stage/'persample.jsonl').write_text((stage/'persample.jsonl').read_text()+'\n')
            run = subprocess.run(command+['--out-dir', str(root/'tampered')], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('checksum', run.stderr)
            self.assertEqual(len(requests), before)

    def test_replay_flags_audio_without_changing_saved_inference_or_routing(self):
        rows, predictions = fixture()
        with tempfile.TemporaryDirectory() as tmp, server() as (endpoint, requests):
            root = Path(tmp)
            with wave.open(str(root/'audio.wav'), 'wb') as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b'\x00\x00' * 160)
            for row, prediction, wer in zip(rows, predictions, [.8, .75, 2.]):
                row.update(audio_sep=str(root/'audio.wav'), audio_mix=str(root/'audio.wav'))
                prediction.update(audio_sep=row['audio_sep'], audio_mix=row['audio_mix'], wer=wer, label_wer=wer)
            manifest, stage = root/'manifest.jsonl', root/'asr'
            stage.mkdir()
            write_jsonl(manifest, rows)
            write_jsonl(stage/'persample.jsonl', predictions)
            original = (stage/'persample.jsonl').read_bytes()
            (stage/'summary.json').write_text(json.dumps(dict(mode='pipeline', wer_threshold=.75,
                n_samples=3, manifest_sha256=fingerprint(manifest), predictions_sha256=fingerprint(stage/'persample.jsonl'))))
            command = [sys.executable, str(ROOT/'scripts/eval_qa_pipeline.py'), '--manifest', str(manifest),
                       '--asr-results-dir', str(stage), '--qa-endpoint', endpoint, '--qa-model', 'test-lm',
                       '--out-dir', str(root/'out'), '--flagged-dir', str(root/'flagged')]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(result['flagged_audio']['n_flagged'], 2)
            self.assertEqual(result['flagged_audio']['flagged_action_counts'], dict(RESPOND=1, REJECT=1))
            scored = [json.loads(line) for line in (root/'out/persample.jsonl').read_text().splitlines()]
            self.assertEqual([r['wer_flagged'] for r in scored], [True, False, True])
            self.assertEqual(len(requests), 1)
            self.assertEqual((stage/'persample.jsonl').read_bytes(), original)
            self.assertNotIn('PRIVATE_REFERENCE', json.dumps(requests))



if __name__ == '__main__':
    unittest.main()
