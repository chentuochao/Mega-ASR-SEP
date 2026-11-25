"""Pinned VoiceBench sd-QA protocols and official PEDANT/GPT grading."""
import ast
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'third_party/voicebench_sdqa'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def protocol():
    provenance = json.loads((UPSTREAM/'PROVENANCE.json').read_text())
    for name, expected in provenance['sha256'].items():
        if sha(UPSTREAM/name) != expected:
            raise ValueError(f'Pinned VoiceBench source mismatch: {name}')
    naive = ast.parse((UPSTREAM/'naive.py').read_text())
    cls = next(node for node in naive.body if isinstance(node, ast.ClassDef))
    systems = {}
    for name in ('generate_audio', 'generate_text'):
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assignment = next(node for node in function.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'messages' for target in node.targets))
        systems[name] = ast.literal_eval(assignment.value.elts[0])['content']
    grading = ast.parse((UPSTREAM/'api_judge.py').read_text())
    assignment = next(node for node in grading.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'meta_prompt_qa' for target in node.targets))
    value = assignment.value
    template = ast.literal_eval(value.func.value).strip() if isinstance(value, ast.Call) else ast.literal_eval(value)
    function = next(node for node in grading.body if isinstance(node, ast.FunctionDef) and node.name == 'generate')
    call = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'generate_text_chat')
    settings = {keyword.arg:ast.literal_eval(keyword.value) for keyword in call.keywords
                if keyword.arg not in ('client', 'messages')}
    messages = next(keyword.value for keyword in call.keywords if keyword.arg == 'messages')
    system = ast.literal_eval(messages.elts[0])['content']
    return dict(revision=provenance['revision'], audio_system=systems['generate_audio'],
                text_system=systems['generate_text'], grading_template=template,
                grading_system=system, grading_settings=settings, source_sha256=provenance['sha256'])


def response_messages(transcript, reference_text=False):
    config = protocol()
    return [dict(role='system', content=config['text_system' if reference_text else 'audio_system']),
            dict(role='user', content=transcript)]


def load_evaluator(resources):
    import joblib
    import numpy as np
    from qa_metrics.pedant import PEDANT
    if importlib.metadata.version('qa_metrics') != '0.2.17':
        raise ValueError('Use qa_metrics==0.2.17 for VoiceBench parity')
    if importlib.metadata.version('scikit-learn') != '1.3.2':
        raise ValueError('Use scikit-learn==1.3.2 for VoiceBench parity')
    resources = Path(resources)
    provenance = json.loads((resources/'PROVENANCE.json').read_text())
    for name, expected in provenance['sha256'].items():
        if sha(resources/name) != expected:
            raise ValueError(f'PEDANT resource checksum mismatch: {name}')
    # Keep PEDANT methods unchanged. Its upstream constructor downloads mutable
    # files into site-packages; inject the same pinned classifiers instead.
    pedant = PEDANT.__new__(PEDANT)
    pedant.model = joblib.load(resources/'lr_classifier.pkl')
    pedant.tokenizer = joblib.load(resources/'tf-idf_vectorizer.pkl')
    protocol()  # Verify the vendored source before executing its unchanged methods.
    module = ast.parse((UPSTREAM/'qa.py').read_text())
    definitions = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = dict(Evaluator=object, np=np, PEDANT=PEDANT)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(UPSTREAM/'qa.py'), 'exec'), namespace)
    evaluator = namespace['QAEvaluator'].__new__(namespace['QAEvaluator'])
    evaluator.pedant = pedant
    return evaluator, namespace['majority_vote'], provenance


class OfficialGPTGrader:
    def __init__(self, endpoint='https://api.openai.com/v1/chat/completions', key_env='OPENAI_API_KEY', timeout=120):
        from urllib.parse import urlsplit
        url = urlsplit(endpoint)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('Use a grading endpoint without embedded credentials')
        if not os.environ.get(key_env):
            raise ValueError(f'Configure the {key_env} environment variable before GPT grading')
        self.endpoint, self.key_env, self.timeout = endpoint, key_env, timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if url.hostname in (
            'localhost', '127.0.0.1', '::1') else urllib.request.build_opener()
        self.config = protocol()
        self.fatal_error = None

    def grade(self, item):
        prompt = self.config['grading_template'].replace('{prompt}', item['prompt']).replace(
            '{reference}', item['reference']).replace('{response}', item['response'])
        payload = dict(self.config['grading_settings'], messages=[
            dict(role='system', content=self.config['grading_system']), dict(role='user', content=prompt)])
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers={
            'Content-Type':'application/json', 'Authorization':'Bearer '+os.environ[self.key_env]})
        # Match the grading payload exactly; retries are bounded and leave failures visible.
        for attempt in range(25):
            if self.fatal_error:
                return dict(self.fatal_error)
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    result = json.load(response)
                choices = result['choices']
                contents = [choice['message']['content'] for choice in choices]
                if len(contents) != 3 or any(not isinstance(value, str) or not value.strip() for value in contents):
                    raise ValueError('Expected three nonempty grading votes')
                votes = [value.strip() for value in contents]
                return dict(votes=votes, served_model=result.get('model'), usage=result.get('usage'), error=None)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
                status = getattr(exc, 'code', None)
                provider_code = None
                provider_type = None
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        provider = json.loads(exc.read(8192)).get('error', {})
                        code = provider.get('code')
                        if code in ('insufficient_quota','invalid_api_key','rate_limit_exceeded','model_not_found'):
                            provider_code = code
                        if provider.get('type') == 'insufficient_quota':
                            provider_type = 'insufficient_quota'
                    except (ValueError, AttributeError, TypeError):
                        pass
                error = dict(votes=None, error=type(exc).__name__, http_status=status,
                             provider_error_code=provider_code, provider_error_type=provider_type)
                if status in (400,401,403,404) or provider_code == 'insufficient_quota' or provider_type == 'insufficient_quota':
                    self.fatal_error = error
                    return error
                if attempt == 24:
                    return error
                time.sleep(30)


def score_records(rows, records, policies, resources, *, gpt_mode='auto', key_env='OPENAI_API_KEY',
                  endpoint='https://api.openai.com/v1/chat/completions', cache_path=None):
    if len(rows) != len(records) or not rows:
        raise ValueError('Scoring requires complete joined rows and predictions')
    for row, record in zip(rows, records):
        benchmark = row.get('benchmark_record', {})
        if (row['sample_id'] != record['sample_id'] or row['text'] != record['reference']
                or not isinstance(benchmark.get('prompt'), str) or not isinstance(benchmark.get('reference'), str)
                or not benchmark['reference'].strip()):
            raise ValueError('Missing or mismatched benchmark scoring metadata')
    evaluator, majority, resources_provenance = load_evaluator(resources)
    enabled = gpt_mode != 'off' and bool(os.environ.get(key_env))
    if gpt_mode == 'required' and not enabled:
        raise ValueError(f'GPT grading requires {key_env}')
    grader = OfficialGPTGrader(endpoint, key_env) if enabled else None
    items, row_keys = {}, {}
    for index, (row, record) in enumerate(zip(rows, records)):
        for policy in policies:
            outcome = record['qa'][policy]
            if outcome['status'] != 'answered':
                continue
            item = dict(prompt=row['benchmark_record']['prompt'], reference=row['benchmark_record']['reference'],
                        response=outcome['response'])
            key = hashlib.sha256(json.dumps(dict(item=item, revision=protocol()['revision'],
                grading_settings=protocol()['grading_settings']), sort_keys=True).encode()).hexdigest()
            items[key] = item
            row_keys[index, policy] = key
    cached = {}
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        for line in cache.read_text().splitlines():
            item = json.loads(line)
            if item['endpoint'] == endpoint and not item['result']['error']:
                cached[item['key']] = item['result']
    if grader:
        pending = [key for key in items if key not in cached]
        def perform(key):
            return key, grader.grade(items[key])
        with ThreadPoolExecutor(max_workers=4) as pool:
            for key, result in pool.map(perform, pending):
                cached[key] = result
                if cache:
                    with cache.open('a') as handle:
                        handle.write(json.dumps(dict(key=key, endpoint=endpoint, result=result))+'\n')
    pedant_results = {key:bool(evaluator.pedant.evaluate([item['reference'].lower()],
        item['response'].lower(), item['prompt'].lower())) for key,item in items.items()}
    summaries = {}
    for policy in policies:
        official_rows, gpt_complete = [], enabled
        for index, record in enumerate(records):
            outcome = record['qa'][policy]
            key = row_keys.get((index, policy))
            if key is None:
                outcome['voicebench'] = dict(panda=None, gpt=None, score=None)
                continue
            result = cached.get(key)
            votes = result.get('votes') if result and not result['error'] else None
            outcome.update(correct=pedant_results[key], answer_valid=True,
                voicebench=dict(panda=pedant_results[key], gpt=majority(votes) if votes else None,
                                score=votes, grading_error=result.get('error') if result else None,
                                http_status=result.get('http_status') if result else None,
                                provider_error_code=result.get('provider_error_code') if result else None,
                                provider_error_type=result.get('provider_error_type') if result else None))
            item = dict(items[key])
            if votes:
                item['score'] = votes
            else:
                gpt_complete = False
            official_rows.append(item)
        n_answered = len(official_rows)
        correct = sum(record['qa'][policy]['voicebench']['panda'] is True for record in records)
        gpt_correct = sum(record['qa'][policy]['voicebench']['gpt'] is True for record in records)
        if n_answered and gpt_complete:
            official = {key:float(value) for key,value in evaluator.evaluate(official_rows).items()}
        else:
            official = dict(panda=100*correct/n_answered if n_answered else None, gpt=None)
        summaries[policy] = dict(official_answered=official, n_total=len(rows), n_answered=n_answered,
            answer_coverage=n_answered/len(rows), pedant_task_success_rate=correct/len(rows),
            gpt_task_success_rate=gpt_correct/len(rows) if gpt_complete else None,
            scope='full_manifest' if n_answered == len(rows) else 'answered_subset',
            gpt_status='complete' if gpt_complete else 'disabled' if gpt_mode == 'off' else 'grading_errors' if enabled else 'not_configured',
            grading_errors=sum(bool(record['qa'][policy]['voicebench'].get('grading_error')) for record in records))
    return dict(schema='voicebench_sdqa_v1', voicebench_revision=protocol()['revision'],
        qa_metrics_resources=resources_provenance, qa_metrics_version=importlib.metadata.version('qa_metrics'),
        scikit_learn_version=importlib.metadata.version('scikit-learn'),
        protocol=protocol(), grader_endpoint=endpoint if enabled else None, grader_key_env=key_env,
        missing_key=not bool(os.environ.get(key_env)), policies=summaries,
        rejection_policy='Excluded from official answered score; not correct in full-manifest task success')
