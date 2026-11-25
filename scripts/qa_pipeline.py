"""QA requests and selective-prediction metrics. No model dependencies."""
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

POLICIES = ('judge', 'always_respond', 'reference_text')
TASKS = ('exact_match', 'multiple_choice', 'voicebench_sdqa', 'unscored')
SYSTEM_PROMPT = 'Answer the user request. Return only your final answer, without reasoning or commentary.'


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'{path}: expected nonempty JSONL objects')
    return rows


def validate_rows(rows):
    ids = set()
    for index, row in enumerate(rows):
        for field in ('sample_id', 'audio_sep', 'audio_mix', 'text', 'qa_task'):
            if not isinstance(row.get(field), str) or (field != 'text' and not row[field]):
                raise ValueError(f'Row {index}: expected string {field}')
        if row['sample_id'] in ids:
            raise ValueError(f'Duplicate sample_id: {row["sample_id"]}')
        ids.add(row['sample_id'])
        if row['qa_task'] not in TASKS:
            raise ValueError(f'Row {index}: unsupported qa_task')
        if row.get('prompt') or row.get('quality_context'):
            raise ValueError('Use text for the reference; prompt/quality_context are not judge inputs')
        if row['qa_task'] != 'unscored':
            answers = row.get('qa_answers')
            if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
                raise ValueError(f'Row {index}: qa_answers must contain nonempty strings')
            if row['qa_task'] == 'multiple_choice':
                choices = row.get('qa_choices')
                if (not isinstance(choices, list) or len(choices) < 2
                        or any(not isinstance(c, str) or not re.fullmatch('[A-Z]', c) for c in choices)
                        or len(set(choices)) != len(choices) or any(a not in choices for a in answers)):
                    raise ValueError(f'Row {index}: use distinct letter qa_choices and matching answer letters')


def messages_for(transcript, task, choices=None, *, reference_text=False):
    # The request builder accepts no reference answers or benchmark prompts.
    if task == 'voicebench_sdqa':
        from voicebench_sdqa import response_messages
        return response_messages(transcript, reference_text)
    instruction = 'Follow the user request.' if task == 'unscored' else SYSTEM_PROMPT
    if task == 'multiple_choice':
        instruction += ' Return exactly one option letter: ' + ', '.join(choices) + '.'
    return [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': transcript}]


def normalize_answer(text):
    # Preserve numbers, negation and internal punctuation (for example, -1 and 1 differ).
    text = unicodedata.normalize('NFKC', text).casefold().strip()
    return ' '.join(text.split()).rstrip('.!?')


def score_response(response, row):
    if row['qa_task'] == 'voicebench_sdqa':
        return {'parsed_answer': None, 'answer_valid': bool(response.strip()), 'correct': None}
    if row['qa_task'] == 'unscored':
        return {'parsed_answer': None, 'answer_valid': None, 'correct': None}
    if row['qa_task'] == 'multiple_choice':
        match = re.fullmatch(r'\s*(?:\(([A-Z])\)|([A-Z]))[.]?\s*', response)
        answer = next((g for g in match.groups() if g), None) if match else None
        valid = answer in row['qa_choices']
        return {'parsed_answer': answer, 'answer_valid': valid,
                'correct': valid and answer in row['qa_answers']}
    answer = normalize_answer(response)
    return {'parsed_answer': answer, 'answer_valid': bool(answer),
            'correct': bool(answer) and answer in {normalize_answer(a) for a in row['qa_answers']}}


def completion_result(result, started, allow_length=False):
    choice = result['choices'][0]
    content = choice['message']['content']
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Empty/non-text QA response')
    finish = choice.get('finish_reason')
    return dict(response=content, finish_reason=finish, usage=result.get('usage'),
                served_model=result.get('model'), latency_seconds=time.perf_counter()-started,
                error=None if finish == 'stop' or (allow_length and finish == 'length') else 'incomplete_or_unknown_finish_reason')


class DirectChatClient:
    """Call the same response model used by infer_vllm.py, without HTTP."""
    def __init__(self, responder, max_tokens=256, temperature=0, *, allow_length=False):
        if max_tokens < 1 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError('Invalid QA output limit or temperature')
        self.responder, self.max_tokens, self.temperature = responder, max_tokens, temperature
        self.allow_length = allow_length

    def metadata(self):
        return dict(transport='in_process', max_tokens=self.max_tokens, temperature=self.temperature,
                    system_prompt=SYSTEM_PROMPT, unscored_system_prompt='Follow the user request.',
                    choice_prompt='Return exactly one option letter: <qa_choices>.',
                    score_truncated_answers=self.allow_length, **self.responder.metadata())

    def generate(self, messages):
        started = time.perf_counter()
        try:
            result = self.responder.generate(messages, max_tokens=self.max_tokens, temperature=self.temperature)
            return completion_result(result, started, self.allow_length)
        except (RuntimeError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            return dict(response=None, finish_reason=None, usage=None, served_model=None,
                        latency_seconds=time.perf_counter()-started, error=type(exc).__name__)


class ChatClient:
    """A chat-completions HTTP endpoint, including local vLLM servers."""
    def __init__(self, endpoint, model, api_key_env='QA_API_KEY', timeout=120, max_tokens=256, temperature=0,
                 *, allow_length=False):
        url = urllib.parse.urlsplit(endpoint)
        if (url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError('Use an HTTP(S) endpoint without credentials, query, or fragment')
        if (not model or not math.isfinite(timeout) or timeout <= 0 or max_tokens < 1
                or not math.isfinite(temperature) or temperature < 0):
            raise ValueError('Invalid QA model, timeout, token limit, or temperature')
        self.endpoint, self.model = endpoint, model
        self.api_key_env, self.timeout = api_key_env, timeout
        self.max_tokens, self.temperature = max_tokens, temperature
        self.allow_length = allow_length
        try:
            loopback = ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            loopback = url.hostname.lower() == 'localhost'
        # Cluster-wide proxies must not receive requests intended for this node.
        self.loopback = loopback
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if loopback else urllib.request.build_opener()

    def metadata(self):
        return dict(endpoint=self.endpoint, model=self.model, max_tokens=self.max_tokens,
                    temperature=self.temperature, timeout_seconds=self.timeout,
                    api_key_env=self.api_key_env, system_prompt=SYSTEM_PROMPT,
                    environment_proxy_bypassed_for_loopback=self.loopback,
                    score_truncated_answers=self.allow_length,
                    unscored_system_prompt='Follow the user request.',
                    choice_prompt='Return exactly one option letter: <qa_choices>.')

    def generate(self, messages):
        payload = dict(model=self.model, messages=messages, max_tokens=self.max_tokens,
                       temperature=self.temperature, stream=False)
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get(self.api_key_env)
        if key:
            headers['Authorization'] = 'Bearer ' + key
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers)
        started = time.perf_counter()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
            return completion_result(result, started, self.allow_length)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            # Do not persist exception bodies: providers can echo credentials or request content.
            return dict(response=None, finish_reason=None, usage=None, served_model=None,
                        latency_seconds=time.perf_counter() - started, error=type(exc).__name__)


def validate_qa_resume(rows, predictions, policies, prior_records):
    """Require a complete, unchanged prefix before reusing saved QA responses."""
    if len(prior_records) > len(rows):
        raise ValueError('Saved QA responses exceed the selected manifest')
    for row, pred, saved in zip(rows, predictions, prior_records):
        if (any(key not in saved or saved[key] != value for key, value in pred.items())
                or saved.get('qa_task') != row['qa_task']
                or saved.get('qa_answers') != row.get('qa_answers')
                or set(saved.get('qa', {})) != set(policies)):
            raise ValueError('Saved QA prefix does not match the ASR inputs or policies')
        for policy in policies:
            outcome = saved['qa'][policy]
            requested = policy != 'judge' or pred['effective_action'] == 'RESPOND'
            if outcome.get('requested') is not requested or outcome.get('error') is not None:
                raise ValueError('Saved QA prefix contains an invalid request or error')
            if requested:
                finishes = ('stop', 'length') if row['qa_task'] == 'voicebench_sdqa' else ('stop',)
                if (outcome.get('status') != 'answered' or not isinstance(outcome.get('response'), str)
                        or not outcome['response'].strip() or outcome.get('finish_reason') not in finishes):
                    raise ValueError('Saved QA prefix contains an incomplete request')
            elif outcome.get('status') != 'rejected' or outcome.get('response') is not None:
                raise ValueError('Saved QA prefix contains an invalid rejection')


def evaluate_qa(rows, predictions, client, policies, output_path, *, resume_records=None):
    if not policies or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError('Select distinct supported policies')
    if len(rows) != len(predictions):
        raise ValueError('ASR predictions do not cover the selected manifest')
    # Validate every join before making requests. Never attach answers by positional zip alone.
    for index, (row, pred) in enumerate(zip(rows, predictions)):
        if (pred.get('index') != index or pred.get('sample_id') != row['sample_id']
                or pred.get('reference') != row['text'] or pred.get('audio_sep') != row['audio_sep']
                or pred.get('audio_mix') != row['audio_mix']
                or not isinstance(pred.get('generated_transcription'), str)
                or pred.get('effective_action') not in ('RESPOND', 'REJECT')):
            raise ValueError(f'ASR prediction identity or content mismatch at row {index}')
    records = list(resume_records or [])
    validate_qa_resume(rows, predictions, policies, records)
    with Path(output_path).open('x') as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        handle.flush()
        for row, pred in zip(rows[len(records):], predictions[len(records):]):
            outcomes = {}
            # Reuse identical transcript/task requests across policies within this sample.
            cache = {}
            for policy in policies:
                requested = policy != 'judge' or pred['effective_action'] == 'RESPOND'
                outcome = dict(requested=requested, response=None, correct=None, answer_valid=None,
                               parsed_answer=None, latency_seconds=0, error=None, status='rejected')
                if requested:
                    transcript = row['text'] if policy == 'reference_text' else pred['generated_transcription']
                    messages = messages_for(transcript, row['qa_task'], row.get('qa_choices'),
                                            reference_text=policy == 'reference_text')
                    key = json.dumps(messages, sort_keys=True)
                    reused = key in cache
                    if not reused:
                        cache[key] = client.generate(messages)
                    result = cache[key]
                    outcome.update(result, status='error' if result['error'] else 'answered',
                                   reused_request=reused)
                    if not result['error']:
                        outcome.update(score_response(result['response'], row))
                    elif row['qa_task'] != 'unscored':
                        outcome.update(correct=False, answer_valid=False)
                stage_times = pred.get('stage_seconds', {})
                stages = ('asr',) if policy == 'always_respond' else ('asr', 'judge')
                base_time = (sum(stage_times[k] for k in stages)
                             if all(k in stage_times for k in stages) else None)
                # Reference-text baseline bypasses ASR/judge. Policy latency includes a reused request's cost.
                outcome['pipeline_compute_seconds'] = (outcome['latency_seconds'] if policy == 'reference_text'
                    else base_time + outcome['latency_seconds'] if base_time is not None else None)
                outcomes[policy] = outcome
            record = dict(pred, qa_task=row['qa_task'], qa_answers=row.get('qa_answers'),
                          benchmark=row.get('benchmark'), category=row.get('category'), qa=outcomes)
            records.append(record)
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
    return records


def distribution(values):
    values = sorted(v for v in values if v is not None)
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                p50=statistics.median(values) if values else None,
                p95=values[math.ceil(.95 * len(values)) - 1] if values else None)


def summarize_policy(records, policy):
    scored = [r for r in records if r['qa_task'] != 'unscored']
    requested = [r for r in records if r['qa'][policy]['requested']]
    completed = [r for r in records if r['qa'][policy]['status'] == 'answered']
    scored_requested = [r for r in scored if r['qa'][policy]['requested']]
    scored_completed = [r for r in scored if r['qa'][policy]['status'] == 'answered']
    correct = sum(r['qa'][policy]['correct'] is True for r in scored)
    wrong = sum(r['qa'][policy]['correct'] is False for r in scored_completed)
    errors = sum(r['qa'][policy]['error'] is not None for r in records)
    return dict(n_samples=len(records), n_scored=len(scored), n_requested=len(requested),
                n_answered=len(completed), n_correct=correct, n_wrong_answers=wrong, n_errors=errors,
                n_truncated=sum(r['qa'][policy].get('finish_reason') == 'length' for r in records),
                request_rate=len(requested)/len(records), answer_coverage=len(completed)/len(records),
                scored_answer_coverage=len(scored_completed)/len(scored) if scored else None,
                task_success_rate=correct/len(scored) if scored else None,
                accuracy_on_requested=correct/len(scored_requested) if scored_requested else None,
                accuracy_on_answered=correct/len(scored_completed) if scored_completed else None,
                wrong_answer_rate=wrong/len(scored) if scored else None,
                qa_request_seconds=distribution(r['qa'][policy]['latency_seconds'] for r in requested),
                pipeline_compute_seconds=distribution(r['qa'][policy]['pipeline_compute_seconds'] for r in records))


def summarize_qa(records, policies):
    result = {p: summarize_policy(records, p) for p in policies}
    for p in policies:
        groups = {}
        for row in records:
            # Keep dataset/task metrics separate; incompatible scores must not become one headline score.
            key = json.dumps([row.get('benchmark'), row.get('category'), row['qa_task']])
            groups.setdefault(key, []).append(row)
        result[p]['groups'] = {key: summarize_policy(group, p) for key, group in groups.items()}
    return result


def compare_rejections(records, policies):
    """Describe the QA outcomes hidden by the judge's rejection policy."""
    if not {'judge', 'always_respond'}.issubset(policies):
        return None
    rejected = [r for r in records if r['effective_action'] == 'REJECT' and r['qa_task'] != 'unscored']
    return dict(n_scored_rejected=len(rejected),
                rejected_but_always_respond_correct=sum(r['qa']['always_respond']['correct'] is True for r in rejected),
                rejected_and_always_respond_wrong=sum(r['qa']['always_respond']['status'] == 'answered'
                    and r['qa']['always_respond']['correct'] is False for r in rejected),
                rejected_always_respond_errors=sum(r['qa']['always_respond']['error'] is not None for r in rejected))
