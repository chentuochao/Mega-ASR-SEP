#!/usr/bin/env python3
"""Join benchmark metadata to explicitly prepared separated/mixture audio pairs.

Load a local JSONL export or pinned Hugging Face revision. This script does not
create mixtures, run the separator, or treat clean audio as separated audio.
"""
import argparse
import json
from pathlib import Path

from qa_pipeline import fingerprint, read_jsonl, validate_rows


def prepare(sources, pairs, benchmark, subset, source_id_field=None, choices=None, task=None):
    lookup = {}
    for index, source in enumerate(sources):
        key = str(source[source_id_field]) if source_id_field else str(index)
        if key in lookup:
            raise ValueError(f'Duplicate benchmark source ID: {key}')
        lookup[key] = source
    rows = []
    for pair in pairs:
        key = str(pair['source_id'])
        if key not in lookup:
            raise ValueError(f'Audio source_id {key} is not in the benchmark')
        source = lookup[key]
        # Never decode/download audio merely to join benchmark text and answer metadata.
        record = {k:v for k,v in source.items() if k != 'audio'}
        if benchmark == 'voicebench':
            reference = source['prompt']
            selected_task = task or ('multiple_choice' if subset == 'openbookqa' else 'unscored')
            answer = source.get('reference')
            if 'text' in pair and pair['text'] != reference:
                raise ValueError(f'{key}: paired transcript differs from VoiceBench prompt')
        else:
            # BBA ships answer metadata but no question transcript. Require verified text.
            reference = pair.get('text')
            selected_task = task or 'exact_match'
            answer = source['official_answer']
        answers = answer if isinstance(answer, list) else [answer]
        row = dict(sample_id=pair['sample_id'], source_id=key,
                   audio_sep=pair['audio_sep'], audio_mix=pair['audio_mix'], text=reference,
                   qa_task=selected_task, benchmark=f'{benchmark}/{subset}',
                   category=source.get('category'), benchmark_record=record)
        if selected_task != 'unscored':
            row['qa_answers'] = answers
        if selected_task == 'multiple_choice':
            row['qa_choices'] = choices or ['A', 'B', 'C', 'D']
        rows.append(row)
    validate_rows(rows)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=['voicebench', 'big_bench_audio'], required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-jsonl', type=Path, help='Local benchmark metadata export')
    source.add_argument('--from-hub', action='store_true')
    p.add_argument('--revision', help='Pinned Hugging Face commit; required with --from-hub')
    p.add_argument('--cache-dir', type=Path, help='Explicit dataset cache; required with --from-hub')
    p.add_argument('--subset', default='openbookqa')
    p.add_argument('--split', help='Default: test for VoiceBench, train for BBA (use as evaluation data)')
    p.add_argument('--source-id-field', help='Default: row index for VoiceBench, id for BBA')
    p.add_argument('--paired-manifest', required=True, type=Path)
    p.add_argument('--out-jsonl', required=True, type=Path)
    p.add_argument('--qa-task', choices=['exact_match', 'multiple_choice', 'unscored'])
    p.add_argument('--choices', nargs='+')
    args = p.parse_args(argv)
    metadata = Path(str(args.out_jsonl)+'.meta.json')
    if args.out_jsonl.exists() or metadata.exists():
        p.error('Select new output and metadata paths')
    split = args.split or ('test' if args.benchmark == 'voicebench' else 'train')
    subset = args.subset if args.benchmark == 'voicebench' else 'default'
    source_id = args.source_id_field or ('id' if args.benchmark == 'big_bench_audio' else None)
    if args.source_jsonl:
        sources = read_jsonl(args.source_jsonl)
        origin = dict(source_jsonl=str(args.source_jsonl.resolve()), source_sha256=fingerprint(args.source_jsonl))
    else:
        import re
        if not args.cache_dir or not args.revision or not re.fullmatch('[0-9a-f]{40}', args.revision):
            p.error('--from-hub requires --cache-dir and a 40-character commit --revision')
        from datasets import load_dataset
        name = 'hlt-lab/voicebench' if args.benchmark == 'voicebench' else 'ArtificialAnalysis/big_bench_audio'
        data = load_dataset(name, subset, split=split, revision=args.revision, cache_dir=str(args.cache_dir))
        if 'audio' in data.column_names:
            data = data.remove_columns('audio')
        sources = list(data)
        origin = dict(dataset=name, revision=args.revision, dataset_fingerprint=data._fingerprint)
    pairs = read_jsonl(args.paired_manifest)
    # Resolve pair audio paths against the pair manifest, so the output can move independently.
    for pair in pairs:
        for field in ('audio_sep', 'audio_mix'):
            path = Path(pair[field])
            path = path if path.is_absolute() else args.paired_manifest.resolve().parent/path
            if not path.is_file():
                raise ValueError(f'Missing {field}: {path}')
            pair[field] = str(path.resolve())
    rows = prepare(sources, pairs, args.benchmark, subset, source_id, args.choices, args.qa_task)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
    metadata.write_text(json.dumps(dict(schema='qa_benchmark_manifest_v1', benchmark=args.benchmark,
        subset=subset, split=split, source_id_field=source_id or 'row_index', origin=origin,
        paired_manifest_sha256=fingerprint(args.paired_manifest),
        manifest_sha256=fingerprint(args.out_jsonl), n_samples=len(rows)), indent=2)+'\n')
    print(f'Prepared {len(rows)} samples: {args.out_jsonl}')


if __name__ == '__main__':
    main()
