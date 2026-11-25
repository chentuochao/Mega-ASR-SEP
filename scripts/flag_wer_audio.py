#!/usr/bin/env python3
"""Copy paired audio whose saved utterance WER exceeds an evaluation threshold.

This offline export uses reference transcripts for inspection only. It does not
change ASR, judge actions, or QA generation, and needs no model dependencies.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def add_flag_arguments(parser):
    parser.add_argument('--flagged-dir', type=Path, help='New directory for bad-WER audio copies and inspection metadata')
    parser.add_argument('--flag-wer-threshold', type=float, help='Flag WER strictly above this ratio; default: evaluation threshold')
    parser.add_argument('--flag-source-manifest', type=Path, help='Original manifest to recover audio paths after temporary storage expires')
    parser.add_argument('--flag-clean-reference', action='store_true', help='Also copy available libritts_path clean reference audio')


def validate_destination(out_dir, threshold, protected_paths=()):
    if not isinstance(threshold, (float, int)) or isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Flag WER threshold must be a finite nonnegative ratio')
    destination = Path(out_dir).resolve()
    if Path(out_dir).exists() or Path(out_dir).is_symlink():
        raise FileExistsError(f'Flagged directory already exists: {out_dir}; select a new run subdirectory')
    for path in protected_paths:
        protected = Path(path).resolve()
        if destination == protected or destination in protected.parents:
            raise ValueError(f'Flagged directory would contain an input or evaluation output: {path}')
    return destination


def check_flag_arguments(parser, args, protected_paths=()):
    if args.flagged_dir is None:
        if args.flag_wer_threshold is not None or args.flag_source_manifest or args.flag_clean_reference:
            parser.error('Flag options require --flagged-dir')
        return
    try:
        validate_destination(args.flagged_dir, args.flag_wer_threshold if args.flag_wer_threshold is not None else .75,
                             protected_paths)
        if args.flag_source_manifest and not args.flag_source_manifest.is_file():
            raise ValueError(f'Source manifest does not exist: {args.flag_source_manifest}')
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


def record_wer(record):
    value = record.get('wer', record.get('label_wer'))
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Missing or invalid saved WER for sample {record.get("sample_id")}')
    if 'label_wer' in record and record['label_wer'] != value:
        raise ValueError('Saved wer and label_wer disagree')
    return float(value)


def sample_id(row, index):
    value = row.get('sample_id', row.get('utt_id', index))
    if not isinstance(value, (str, int)) or isinstance(value, bool) or str(value) == '':
        raise ValueError('Invalid sample identifier')
    return str(value)


def index_rows(rows):
    indexed = {}
    for index, row in enumerate(rows):
        key = sample_id(row, index)
        if key in indexed:
            raise ValueError(f'Duplicate sample identifier: {key}')
        indexed[key] = row
    return indexed


def export_flagged_audio(records, out_dir, *, threshold, input_manifest=None,
                         source_manifest=None, include_clean_reference=False, records_path=None):
    """Preflight sources, then create a new collection, sorted by descending WER.

    Source rows join by sample_id/utt_id, with an exact reference-text check.
    Relative manifest audio paths resolve against that manifest's directory.
    Required mixture/separated files must all exist before copying starts.
    """
    protected = [p for p in (input_manifest, source_manifest, records_path) if p is not None]
    destination = validate_destination(out_dir, threshold, protected)
    indexed = index_rows(records)
    manifests = []
    provenance = {}
    for label, path in [('input_manifest', input_manifest), ('source_manifest', source_manifest)]:
        if path is not None:
            path = Path(path).resolve()
            manifests.append((path.parent, index_rows(read_jsonl(path))))
            provenance[label] = dict(path=str(path), sha256=fingerprint(path))
    if records_path is not None:
        provenance['records'] = dict(path=str(Path(records_path).resolve()), sha256=fingerprint(records_path))
    selected = [(key, row, record_wer(row)) for key, row in indexed.items()]
    selected = sorted((entry for entry in selected if entry[2] > threshold), key=lambda entry: (-entry[2], entry[0]))
    plans = []
    for key, record, wer in selected:
        reference = record.get('reference', record.get('ref'))
        hypothesis = record.get('generated_transcription', record.get('hyp'))
        if not isinstance(reference, str) or not isinstance(hypothesis, str):
            raise ValueError(f'Missing transcript for sample {key}')
        sources = [(Path.cwd(), record)]
        for directory, manifest_rows in manifests:
            if key not in manifest_rows:
                raise ValueError(f'Sample {key} missing from supplied manifest')
            row = manifest_rows[key]
            if row.get('text') != reference:
                raise ValueError(f'Reference transcript mismatch for sample {key}')
            sources.append((directory, row))
        audio = {}
        for role, field in [('separated', 'audio_sep'), ('mixture', 'audio_mix'), ('clean_reference', 'libritts_path')]:
            if role == 'clean_reference' and not include_clean_reference:
                continue
            candidates = []
            for directory, row in sources:
                value = row.get(field)
                if value is None and field == 'audio_sep':
                    value = row.get('audio_path')
                if value is None and field == 'libritts_path':
                    value = row.get('benchmark_record', {}).get(field)
                if value:
                    path = Path(value).expanduser()
                    candidates.append((directory / path).resolve() if not path.is_absolute() else path.resolve())
            audio[role] = next((path for path in candidates if path.is_file()), None)
            if audio[role] is None and role != 'clean_reference':
                raise FileNotFoundError(f'Missing {role} audio for sample {key}: {candidates}')
        plans.append((key, record, wer, reference, hypothesis, audio))
    # No existing output is overwritten. A failed copy leaves its partial folder for inspection.
    destination.mkdir(parents=True, exist_ok=False)
    exported = []
    total_bytes = 0
    audio_counts = Counter()
    with (destination / 'manifest.jsonl').open('x') as manifest:
        for rank, (key, record, wer, reference, hypothesis, sources) in enumerate(plans, 1):
            safe_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', key).strip('._-')[:80] or 'sample'
            folder = destination / f'{rank:04d}_{safe_id}_wer{wer:.4f}'
            folder.mkdir()
            audio = {}
            for role, source in sources.items():
                if source is None:
                    continue
                target = folder / (role + source.suffix)
                before = source.stat()
                shutil.copyfile(source, target)
                checksum = fingerprint(target)
                if fingerprint(source) != checksum or source.stat().st_mtime_ns != before.st_mtime_ns:
                    raise RuntimeError(f'Audio source changed during export: {source}')
                audio[role] = dict(source=str(source), path=str(target), sha256=checksum,
                                   bytes=target.stat().st_size)
                total_bytes += target.stat().st_size
                audio_counts[role] += 1
            item = dict(record, sample_id=key, wer=wer, wer_flagged=True,
                        flag_wer_threshold=threshold, flag_rule='wer > threshold',
                        folder=str(folder), copied_audio=audio)
            (folder / 'metadata.json').write_text(json.dumps(item, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            (folder / 'transcript.txt').write_text(
                f'Sample: {key}\nWER: {wer:.6f} ({100 * wer:.2f}%)\n'
                f'Flag rule: WER > {threshold}\nJudge action: {record.get("effective_action", "unknown")}\n\n'
                f'Reference transcript:\n{reference}\n\nASR transcript:\n{hypothesis}\n')
            manifest.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
            manifest.flush()
            exported.append(item)
    summary = dict(schema='flagged_wer_audio_v1', generated_utc=datetime.now(timezone.utc).isoformat(),
                   directory=str(destination), n_samples=len(records), n_flagged=len(exported),
                   flag_wer_threshold=threshold, flag_rule='wer > threshold',
                   order='descending WER, then sample identifier',
                   flagged_action_counts=dict(Counter(row.get('effective_action', 'unknown') for row in exported)),
                   copied_audio_counts=dict(audio_counts), copied_audio_bytes=total_bytes,
                   include_clean_reference=include_clean_reference,
                   missing_clean_reference=len(exported) - audio_counts['clean_reference'] if include_clean_reference else None,
                   provenance=provenance, manifest_sha256=fingerprint(destination / 'manifest.jsonl'))
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True, type=Path, help='Saved eval_judge persample.jsonl')
    parser.add_argument('--out-dir', required=True, type=Path, help='New collection directory')
    parser.add_argument('--wer-threshold', required=True, type=float, help='WER ratio, for example 0.75')
    parser.add_argument('--manifest', type=Path, help='Manifest supplied to the evaluator')
    parser.add_argument('--source-manifest', type=Path, help='Original manifest with permanent audio paths')
    parser.add_argument('--include-clean-reference', action='store_true')
    args = parser.parse_args(argv)
    summary = export_flagged_audio(read_jsonl(args.records), args.out_dir, threshold=args.wer_threshold,
        input_manifest=args.manifest, source_manifest=args.source_manifest,
        include_clean_reference=args.include_clean_reference, records_path=args.records)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
