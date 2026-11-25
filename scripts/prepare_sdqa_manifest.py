#!/usr/bin/env python3
"""Adapt prepared VoiceBench sd-QA audio pairs without changing the source."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from qa_pipeline import fingerprint, read_jsonl, validate_rows


def convert(rows):
    prepared = []
    for index, source in enumerate(rows):
        for key in ('audio_sep', 'audio_mix', 'text', 'prompt', 'reference'):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError(f'Row {index}: missing {key}')
        sample_id = str(source.get('sample_id', source.get('utt_id', source.get('row_idx', index))))
        metadata = {key:value for key,value in source.items() if key not in ('audio_sep', 'audio_mix')}
        prepared.append(dict(sample_id=sample_id, source_id=str(source.get('row_idx', index)),
            audio_sep=source['audio_sep'], audio_mix=source['audio_mix'], text=source['text'],
            qa_task='voicebench_sdqa', qa_answers=[source['reference']],
            benchmark='voicebench/sd-qa/usa', benchmark_record=metadata))
    validate_rows(prepared)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out-jsonl', required=True, type=Path)
    parser.add_argument('--stage-audio-dir', type=Path, help='New compute-node temporary directory')
    args = parser.parse_args()
    sidecar = Path(str(args.out_jsonl)+'.meta.json')
    if args.out_jsonl.exists() or sidecar.exists():
        parser.error('Select new manifest and metadata paths')
    sources = read_jsonl(args.source)
    rows = convert(sources)
    if args.stage_audio_dir:
        args.stage_audio_dir.mkdir(parents=True, exist_ok=False)
    checksums = []
    for index, row in enumerate(rows):
        for field in ('audio_sep', 'audio_mix'):
            original = Path(row[field])
            original = original if original.is_absolute() else args.source.resolve().parent/original
            if not original.is_file():
                raise ValueError(f'Missing {field}: {original}')
            if args.stage_audio_dir:
                destination = args.stage_audio_dir/f'{index:06d}_{field}.wav'
                shutil.copy2(original, destination)
                expected = fingerprint(original)
                if fingerprint(destination) != expected:
                    raise ValueError(f'Audio copy mismatch: {original}')
                checksums.append(dict(source=str(original), staged=str(destination), sha256=expected))
                row[field] = str(destination.resolve())
            else:
                row[field] = str(original.resolve())
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open('x') as handle:
        handle.writelines(json.dumps(row, allow_nan=False)+'\n' for row in rows)
    metadata = dict(source=str(args.source.resolve()), source_sha256=fingerprint(args.source),
        manifest_sha256=fingerprint(args.out_jsonl), n_samples=len(rows),
        task_counts=dict(Counter(row['qa_task'] for row in rows)),
        prompt_text_mismatches=sum(row['prompt'] != row['text'] for row in sources),
        audio_copies=checksums, source_preserved=True)
    sidecar.write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(n_samples=len(rows), manifest=str(args.out_jsonl),
                         copied_audio_files=len(checksums), source_sha256=metadata['source_sha256']), indent=2))


if __name__ == '__main__':
    main()
