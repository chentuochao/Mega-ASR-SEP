#!/usr/bin/env python3
"""Own one response or grading model for one serial pipeline stage."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from qa_pipeline import DirectChatClient, evaluate_qa, fingerprint, read_jsonl, validate_qa_resume

ROOT = Path(__file__).resolve().parents[1]


def load_resume(config, rows, records):
    directory = config.get('resume_dir')
    if not directory:
        return [], None
    directory = Path(directory)
    source_path = directory/'qa_worker_config.json'
    source = json.loads(source_path.read_text())
    for key in ('backend', 'device', 'model_name', 'max_model_len', 'max_tokens', 'temperature', 'voicebench', 'policies'):
        if source.get(key) != config.get(key):
            raise ValueError(f'Saved QA generation setting differs: {key}')
    if Path(source['model_path']).resolve() != Path(config['model_path']).resolve():
        raise ValueError('Saved QA responses came from another model path')
    defaults = dict(tensor_parallel_size=1, dtype='auto')
    if dict(defaults, **source['engine_options']) != dict(defaults, **config['engine_options']):
        raise ValueError('Saved QA engine settings differ')
    if fingerprint(source['manifest']) != fingerprint(config['manifest']):
        raise ValueError('Saved QA manifest checksum differs')
    original_revision = source.get('model_revision')
    if not original_revision:
        raise ValueError('Saved QA configuration has no model revision; generate new responses')
    provenance = json.loads((Path(config['model_path'])/'model_provenance.json').read_text())
    if not original_revision or provenance.get('revision') != original_revision:
        raise ValueError('Saved QA model revision does not match current weights')
    path = directory/'generation.jsonl'
    saved = read_jsonl(path)
    validate_qa_resume(rows, records, config['policies'], saved)
    return saved, dict(directory=str(directory.resolve()), n_records=len(saved),
        records_sha256=fingerprint(path), source_config_sha256=fingerprint(source_path),
        model_revision=original_revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['respond', 'grade'])
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = read_jsonl(config['manifest'])
    if config.get('limit') is not None:
        rows = rows[:config['limit']]
    records = read_jsonl(config['input_records'])
    saved, resume = load_resume(config, rows, records) if args.task == 'respond' else ([], None)
    if resume:
        with Path(config['summary']).with_name('qa_resume_validation.json').open('x') as handle:
            handle.write(json.dumps(resume, indent=2)+'\n')
    sys.path.insert(0, str(ROOT / 'src'))
    from MegaASR.model.text_response import TextResponseModel
    started = time.perf_counter()
    model = TextResponseModel(config['model_path'], backend=config['backend'], device=config.get('device', 'cuda'),
        enable_thinking=False, model_name=config.get('model_name'),
        max_model_len=config['max_model_len'], **config['engine_options'])
    load_seconds = time.perf_counter() - started
    if args.task == 'respond':
        client = DirectChatClient(model, config['max_tokens'], config['temperature'], allow_length=config['voicebench'])
        evaluate_qa(rows, records, client, config['policies'], config['output_records'], resume_records=saved)
        summary = dict(client=client.metadata(), load_seconds=load_seconds, resume=resume)
    else:
        from local_qa_grader import grade_records
        summary = grade_records(rows, records, config['policies'], model,
                                config['votes_path'], max_tokens=config['max_tokens'])
        summary['load_seconds'] = load_seconds
        with Path(config['output_records']).open('x') as handle:
            handle.writelines(json.dumps(record, allow_nan=False) + '\n' for record in records)
    summary.update(pid=os.getpid(), versions={name: importlib.metadata.version(name)
                   for name in ('vllm', 'torch', 'transformers') if args.task == 'grade' or config['backend'] == 'vllm'})
    Path(config['summary']).write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return 2 if summary.get('status') == 'grading_errors' else 0


if __name__ == '__main__':
    raise SystemExit(main())
