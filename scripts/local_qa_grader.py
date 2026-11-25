"""Grade QA correctness with a local model, separate from official GPT scores."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from voicebench_sdqa import protocol


def grade_records(rows, records, policies, model, votes_path, *, max_tokens=1024):
    if not rows or len(rows) != len(records):
        raise ValueError('Local grading requires complete joined rows and predictions')
    config = protocol()
    settings = dict(n=3, temperature=.5, top_p=.95, max_tokens=max_tokens)
    jobs, slots = {}, {}
    for index, (row, record) in enumerate(zip(rows, records)):
        benchmark = row.get('benchmark_record', {})
        if (row['sample_id'] != record['sample_id'] or row['text'] != record['reference']
                or not isinstance(benchmark.get('reference'), str) or not benchmark['reference'].strip()
                or not isinstance(benchmark.get('prompt'), str)):
            raise ValueError('Missing or mismatched local-grading reference metadata')
        for policy in policies:
            outcome = record['qa'][policy]
            if outcome['status'] != 'answered':
                outcome['local_grader'] = dict(correct=None, votes=None, error=None, status='not_answered')
                continue
            item = dict(prompt=benchmark['prompt'], reference=benchmark['reference'], response=outcome['response'])
            key = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
            jobs[key] = item
            slots[index, policy] = key
    results = {}
    with Path(votes_path).open('x') as handle:
        for key, item in jobs.items():
            prompt = config['grading_template'].replace('{prompt}', item['prompt']).replace(
                '{reference}', item['reference']).replace('{response}', item['response'])
            messages = [dict(role='system', content=config['grading_system']), dict(role='user', content=prompt)]
            started = time.perf_counter()
            completion = None
            try:
                completion = model.generate(messages, **settings)
                choices = completion['choices']
                votes = [choice['message']['content'].strip() for choice in choices]
                if (len(votes) != 3 or any(v.lower() not in ('yes', 'no') for v in votes)
                        or any(choice['finish_reason'] != 'stop' for choice in choices)):
                    raise ValueError('Expected three complete Yes/No grading votes')
                correct = Counter(v.lower() for v in votes).most_common(1)[0][0] == 'yes'
                result = dict(correct=correct, votes=votes, error=None, status='complete', completion=completion)
            except Exception as error:
                result = dict(correct=None, votes=None, error=type(error).__name__, status='error',
                              detail=str(error), completion=completion)
            result['latency_seconds'] = time.perf_counter() - started
            results[key] = result
            handle.write(json.dumps(dict(key=key, item=item, result=result), allow_nan=False) + '\n')
            handle.flush()
    summaries = {}
    for policy in policies:
        answered, correct, errors = 0, 0, 0
        for index, record in enumerate(records):
            key = slots.get((index, policy))
            if key is None:
                continue
            result = {key: value for key, value in results[key].items() if key != 'completion'}
            record['qa'][policy]['local_grader'] = result
            answered += 1
            correct += result['correct'] is True
            errors += result['error'] is not None
        summaries[policy] = dict(n_total=len(rows), n_answered=answered, n_correct=correct, n_errors=errors,
            accuracy_percent=100 * correct / answered if answered and not errors else None,
            task_success_rate=correct / len(rows) if not errors else None,
            status='grading_errors' if errors else 'complete')
    return dict(schema='local_qa_correctness_v1', grader=model.metadata(), settings=settings,
        prompt_protocol_revision=config['revision'], prompt=config['grading_template'],
        policies=summaries, unique_grading_requests=len(jobs),
        status='grading_errors' if any(value['n_errors'] for value in summaries.values()) else 'complete',
        metric_scope='Local Qwen grader; distinct from VoiceBench official gpt scoring',
        rejection_policy='No grading call; unanswered, not correct in full-manifest task success')
