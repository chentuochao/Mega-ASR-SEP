"""Verify source-derived prompts, official grading payloads, and real PEDANT parity."""
import ast
import contextlib
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = Path(os.environ.get('VOICEBENCH_TEST_RESOURCES', ROOT/'ckpt/voicebench-eval/resources'))
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_sdqa_manifest import convert
from qa_pipeline import messages_for, completion_result
from qa_pipeline import fingerprint
from voicebench_sdqa import (UPSTREAM, OfficialGPTGrader, load_evaluator, protocol, score_records)


def fixtures():
    source = [dict(utt_id=f'sdqa-{index}', row_idx=index, audio_sep=f'{index}-sep.wav',
        audio_mix=f'{index}-mix.wav', text=question, prompt=question, reference=answer)
        for index,(question,answer) in enumerate([
            ('Who invented Hangul?', 'Sejong the Great'),
            ('When was quantum field theory developed?', '1920s'),
            ('Who won the first Nobel Prize for Literature?', 'Sully Prudhomme')])]
    rows = convert(source)
    outputs = ['Sejong the Great invented Hangul.', 'The answer is banana.', 'Sully Prudhomme']
    records = []
    for index,(row,response) in enumerate(zip(rows,outputs)):
        outcomes = {}
        for policy in ['judge','always_respond']:
            answered = policy != 'judge' or index != 2
            outcomes[policy] = dict(status='answered' if answered else 'rejected', requested=answered,
                response=response if answered else None, correct=None, error=None)
        records.append(dict(index=index,sample_id=row['sample_id'],reference=row['text'],qa=outcomes))
    return source, rows, records


class ProtocolTests(unittest.TestCase):
    def test_conversion_excludes_reference_fields_from_model_input(self):
        source,rows,_ = fixtures()
        self.assertNotIn('prompt', rows[0])
        self.assertEqual(rows[0]['qa_task'], 'voicebench_sdqa')
        messages = messages_for('ASR_CANDIDATE_ONLY', 'voicebench_sdqa')
        self.assertEqual(messages[1]['content'], 'ASR_CANDIDATE_ONLY')
        self.assertNotIn(source[0]['reference'], json.dumps(messages))
        self.assertEqual(messages[0]['content'], protocol()['audio_system'])
        oracle = messages_for('REFERENCE_QUESTION', 'voicebench_sdqa', reference_text=True)
        self.assertEqual(oracle[0]['content'], protocol()['text_system'])

    def test_grader_request_matches_unchanged_upstream_generate_function(self):
        captured = []
        item = dict(prompt='Who invented Hangul?',reference='Sejong the Great',response='King Sejong')
        function = next(node for node in ast.parse((UPSTREAM/'api_judge.py').read_text()).body
                        if isinstance(node,ast.FunctionDef) and node.name=='generate')
        def fake_generate(**kwargs):
            captured.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=v)) for v in ['Yes','No','Yes']])
        namespace = dict(client=object(),generate_text_chat=fake_generate,
                         meta_prompt_qa=protocol()['grading_template'],meta_prompt_open='unused')
        exec(compile(ast.Module(body=[function],type_ignores=[]),'upstream-generate','exec'),namespace)
        namespace['generate'](dict(item))
        requests = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                data = json.dumps(dict(model='gpt-4o-mini',choices=[dict(message=dict(content=value))
                    for value in ['Yes','No','Yes']])).encode()
                self.send_response(200)
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self,*args): pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ,OPENAI_API_KEY='TEST_SECRET'):
                grader = OfficialGPTGrader(f'http://127.0.0.1:{server.server_port}/v1/chat/completions')
                result = grader.grade(item)
            expected = dict(captured[0])
            expected.pop('client')
            self.assertEqual(requests,[expected])
            self.assertEqual(result['votes'],['Yes','No','Yes'])
            self.assertNotIn('TEST_SECRET',json.dumps(result))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_voicebench_scores_truncated_text_without_changing_strict_mode(self):
        import time
        response=dict(model='test',choices=[dict(message=dict(content='Partial answer'),finish_reason='length')])
        self.assertIsNotNone(completion_result(response,time.perf_counter())['error'])
        self.assertIsNone(completion_result(response,time.perf_counter(),True)['error'])

    def test_quota_error_type_stops_retries_without_persisting_error_body(self):
        body=json.dumps(dict(error=dict(type='insufficient_quota',code=None,message='PRIVATE_PROVIDER_BODY'))).encode()
        error=urllib.error.HTTPError('https://api.openai.com/v1/chat/completions',429,'quota',{},io.BytesIO(body))
        with patch.dict(os.environ,OPENAI_API_KEY='TEST_SECRET'):
            grader=OfficialGPTGrader()
            with patch.object(grader.opener,'open',side_effect=error) as request, patch('voicebench_sdqa.time.sleep') as sleep:
                result=grader.grade(dict(prompt='Question',reference='Answer',response='Answer'))
                again=grader.grade(dict(prompt='Another question',reference='Answer',response='Answer'))
        self.assertEqual(request.call_count,1)
        sleep.assert_not_called()
        self.assertEqual(result,again)
        self.assertEqual(result['provider_error_type'],'insufficient_quota')
        self.assertNotIn('PRIVATE_PROVIDER_BODY',json.dumps(result))
        self.assertNotIn('TEST_SECRET',json.dumps(result))


@unittest.skipUnless(importlib.util.find_spec('qa_metrics'), 'Install the pinned VoiceBench scoring environment')
class RealPEDANTTests(unittest.TestCase):
    def test_real_pipeline_cli_runs_pinned_scorer_in_separate_process(self):
        _,rows,records=fixtures()
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                question=request['messages'][-1]['content']
                answer='Sejong the Great' if 'Hangul' in question else 'Sully Prudhomme' if 'Nobel' in question else 'banana'
                data=json.dumps(dict(model='qa-test',choices=[dict(message=dict(content=answer),finish_reason='stop')])).encode()
                self.send_response(200)
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self,*args): pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                manifest=root/'input.jsonl'
                manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                stage=root/'asr'
                stage.mkdir()
                predictions=[]
                for index,row in enumerate(rows):
                    predictions.append(dict(index=index,sample_id=row['sample_id'],reference=row['text'],
                        audio_sep=row['audio_sep'],audio_mix=row['audio_mix'],generated_transcription=row['text'],
                        effective_action='REJECT' if index==2 else 'RESPOND'))
                path=stage/'persample.jsonl'
                path.write_text(''.join(json.dumps(row)+'\n' for row in predictions))
                (stage/'summary.json').write_text(json.dumps(dict(mode='pipeline',n_samples=3,
                    manifest_sha256=fingerprint(manifest),predictions_sha256=fingerprint(path))))
                command=[sys.executable,str(ROOT/'scripts/eval_qa_pipeline.py'),'--manifest',str(manifest),
                    '--asr-results-dir',str(stage),'--out-dir',str(root/'out'),
                    '--qa-endpoint',f'http://127.0.0.1:{server.server_port}/v1/chat/completions','--qa-model','qa-test',
                    '--policies','judge','always_respond','--voicebench-resources',str(RESOURCES),
                    '--voicebench-gpt-mode','off']
                result=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr+(root/'out/voicebench_scoring.log').read_text())
                summary=json.loads((root/'out/summary.json').read_text())
                self.assertEqual(summary['scoring'],'voicebench_sdqa_v1')
                self.assertEqual(summary['qa']['judge']['n_correct'],1)
                self.assertEqual(summary['qa']['always_respond']['n_correct'],2)
                self.assertEqual(summary['voicebench']['policies']['judge']['scope'],'answered_subset')
                self.assertIsNone(summary['voicebench']['policies']['always_respond']['official_answered']['gpt'])
                self.assertTrue((root/'out/generation.jsonl').is_file())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_real_classifier_matches_upstream_and_rejections_keep_denominator(self):
        resources = RESOURCES
        official,_,_ = load_evaluator(resources)
        _,rows,records = fixtures()
        expected = [bool(official.pedant.evaluate([row['benchmark_record']['reference'].lower()],
            record['qa']['always_respond']['response'].lower(),row['benchmark_record']['prompt'].lower()))
            for row,record in zip(rows,records)]
        with patch.dict(os.environ,OPENAI_API_KEY='TEST_SECRET'), patch('voicebench_sdqa.OfficialGPTGrader') as grader:
            summary=score_records(rows,records,['judge','always_respond'],resources,gpt_mode='off')
            grader.assert_not_called()
        self.assertEqual([record['qa']['always_respond']['correct'] for record in records],expected)
        self.assertEqual(summary['policies']['judge']['n_total'],3)
        self.assertEqual(summary['policies']['judge']['n_answered'],2)
        self.assertEqual(summary['policies']['judge']['pedant_task_success_rate'],sum(expected[:2])/3)
        self.assertIsNone(summary['policies']['always_respond']['official_answered']['gpt'])
        self.assertEqual(summary['policies']['always_respond']['gpt_status'],'disabled')
        data=[dict(prompt=row['benchmark_record']['prompt'],reference=row['benchmark_record']['reference'],
            response=record['qa']['always_respond']['response'],score=['Yes','No','Yes'])
            for row,record in zip(rows,records)]
        actual=official.evaluate(data)
        self.assertAlmostEqual(summary['policies']['always_respond']['official_answered']['panda'],actual['panda'])
        self.assertEqual(actual['gpt'],100)


if __name__=='__main__':
    unittest.main()
