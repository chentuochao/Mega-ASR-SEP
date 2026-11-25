"""Check reuse of completed QA rows and rejection of incompatible history."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from qa_pipeline import evaluate_qa
from qa_model_worker import load_resume
from test_qa_pipeline import fixture


class Client:
    def __init__(self):
        self.calls=[]

    def generate(self,messages):
        self.calls.append(messages)
        return dict(response='4',finish_reason='stop',error=None,latency_seconds=.1)


class ResumeTests(unittest.TestCase):
    def test_only_remaining_requests_run_and_saved_rows_stay_unchanged(self):
        rows,predictions=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            saved=evaluate_qa(rows,predictions,Client(),['judge','always_respond'],root/'first.jsonl')[:2]
            client=Client()
            actual=evaluate_qa(rows,predictions,client,['judge','always_respond'],root/'resumed.jsonl',resume_records=saved)
            self.assertEqual(actual[:2],saved)
            self.assertEqual(len(actual),3)
            self.assertEqual(len(client.calls),1)
            self.assertEqual(actual[2]['qa']['judge']['status'],'rejected')

    def test_invalid_history_fails_before_requests_or_output_creation(self):
        rows,predictions=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=evaluate_qa(rows,predictions,Client(),['judge'],root/'first.jsonl')[:1]
            for change in ('transcript','request','error','finish'):
                saved=copy.deepcopy(original)
                if change=='transcript':saved[0]['generated_transcription']='different question'
                elif change=='request':saved[0]['qa']['judge']['requested']=False
                elif change=='error':saved[0]['qa']['judge']['error']='failed'
                else:saved[0]['qa']['judge']['finish_reason']='length'
                client=Client();out=root/f'{change}.jsonl'
                with self.subTest(change=change),self.assertRaises(ValueError):
                    evaluate_qa(rows,predictions,client,['judge'],out,resume_records=saved)
                self.assertEqual(client.calls,[])
                self.assertFalse(out.exists())

    def test_config_revision_and_manifest_must_match(self):
        rows,predictions=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prefix=root/'old';prefix.mkdir();model=root/'model';model.mkdir()
            manifest=root/'manifest.jsonl';manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            old_manifest=root/'old_manifest.jsonl';old_manifest.write_bytes(manifest.read_bytes())
            saved=evaluate_qa(rows,predictions,Client(),['judge'],root/'original.jsonl')[:1]
            (prefix/'generation.jsonl').write_text(json.dumps(saved[0])+'\n')
            provenance=model/'model_provenance.json';provenance.write_text(json.dumps(dict(revision='a'*40)))
            source=dict(model_path=str(model),model_revision='a'*40,manifest=str(old_manifest),backend='vllm',device='cpu',
                model_name=None,max_model_len=8192,max_tokens=2048,temperature=0,voicebench=False,
                policies=['judge'],engine_options={})
            config=dict(source,manifest=str(manifest),resume_dir=str(prefix),
                engine_options=dict(tensor_parallel_size=1,dtype='auto'))
            source_path=prefix/'qa_worker_config.json';source_path.write_text(json.dumps(source))
            actual,receipt=load_resume(config,rows,predictions)
            self.assertEqual(actual,saved)
            self.assertEqual(receipt['n_records'],1)
            with self.assertRaisesRegex(ValueError,'max_tokens'):
                load_resume(dict(config,max_tokens=1024),rows,predictions)
            provenance.write_text(json.dumps(dict(revision='b'*40)))
            with self.assertRaisesRegex(ValueError,'revision'):
                load_resume(config,rows,predictions)
            provenance.write_text(json.dumps(dict(revision='a'*40)))
            old_manifest.write_text(old_manifest.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'manifest checksum'):
                load_resume(config,rows,predictions)
            old_manifest.write_bytes(manifest.read_bytes())
            source.pop('model_revision');source_path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError,'no model revision'):
                load_resume(config,rows,predictions)


if __name__=='__main__':
    unittest.main()
