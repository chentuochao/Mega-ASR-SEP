#!/usr/bin/env python3
"""Label production ASR candidates for transcript-conditioned judge training.

Read candidates from source-row fields or evaluator predictions, recalculate
normalized WER, and write action labels with checkpoint and source provenance.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/MegaASR/A2S-SFT'))
from judge_data import SCHEMA, NORMALIZATION, candidate_wer, fingerprint, normalize, validate_judge_manifest


def read_rows(path):
    return [(index, json.loads(line)) for index, line in enumerate(path.read_text().splitlines())
            if line.strip()]


def load_predictions(path, summary_path, source_rows, checkpoint):
    summary = json.loads(summary_path.read_text())
    if os.path.normpath(summary.get('checkpoint_dir', '')) != os.path.normpath(checkpoint):
        raise ValueError('Prediction summary checkpoint does not match --checkpoint')
    predictions = {}
    for _, prediction in read_rows(path):
        index = prediction.get('index')
        if type(index) is not int or index in predictions:
            raise ValueError(f'Missing or duplicate prediction index: {index!r}')
        predictions[index] = prediction
    expected = {index for index, _ in source_rows}
    if set(predictions) != expected or summary.get('n_samples') != len(expected):
        raise ValueError('Predictions and summary must cover every source sample exactly once')
    for index, row in source_rows:
        prediction = predictions[index]
        if normalize(row['text']) != normalize(prediction['ref']):
            raise ValueError(f'Prediction reference mismatch at source index {index}')
        if os.path.realpath(row['audio_sep']) != os.path.realpath(prediction['audio_path']):
            raise ValueError(f'Prediction audio path mismatch at source index {index}')
    return predictions, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--generation-config', required=True, help='JSON with known decoding settings; retain unknown settings explicitly')
    p.add_argument('--candidate-field', default='label_hypothesis')
    p.add_argument('--predictions', help='Optional eval_checkpoint.py prediction JSONL')
    p.add_argument('--prediction-summary', help='Required with --predictions; evaluator summary JSON')
    p.add_argument('--wer-threshold', type=float, default=0.75)
    args = p.parse_args()
    if bool(args.predictions) != bool(args.prediction_summary):
        p.error('--predictions and --prediction-summary must be supplied together')
    source, output = Path(args.source), Path(args.output)
    sidecar = Path(str(output) + '.meta.json')
    if source.resolve() == output.resolve() or output.exists() or sidecar.exists():
        raise ValueError('Write to a new manifest path; originals must be preserved')
    if args.predictions and output.resolve() in (Path(args.predictions).resolve(), Path(args.prediction_summary).resolve()):
        raise ValueError('Output must not overwrite prediction files')
    if not math.isfinite(args.wer_threshold) or args.wer_threshold < 0:
        raise ValueError('WER threshold must be finite and nonnegative')
    source_meta_path = Path(str(source) + '.meta.json')
    source_meta = json.loads(source_meta_path.read_text()) if source_meta_path.exists() else None
    if source_meta and source_meta.get('manifest_sha256') and source_meta['manifest_sha256'] != fingerprint(source):
        raise ValueError('Source manifest does not match its recorded checksum')
    if not args.predictions:
        if not source_meta:
            raise ValueError('Recorded source candidates require a metadata sidecar')
        if os.path.normpath(source_meta.get('checkpoint_id', '')) != os.path.normpath(args.checkpoint):
            raise ValueError('Source checkpoint provenance does not match --checkpoint')
    config = json.loads(Path(args.generation_config).read_text())
    if not isinstance(config, dict) or not config:
        raise ValueError('A nonempty generation configuration is required')
    source_rows = read_rows(source)
    if not source_rows:
        raise ValueError('Empty source manifest')
    predictions, summary = None, None
    if args.predictions:
        predictions, summary = load_predictions(Path(args.predictions), Path(args.prediction_summary),
                                                source_rows, args.checkpoint)
        settings = dict(language='language', sampling_rate='sample_rate', use_fusion='use_fusion',
                        fusion_type='fusion_type', collator='collator', max_new_tokens='max_new_tokens',
                        do_sample='do_sample')
        for summary_key, config_key in settings.items():
            if summary_key in summary and config_key in config and summary[summary_key] != config[config_key]:
                raise ValueError(f'Generation setting disagrees with the prediction summary: {config_key}')
    rows = []
    for index, source_row in source_rows:
        row = dict(source_row)
        candidate = predictions[index]['hyp'] if predictions is not None else row[args.candidate_field]
        wer = candidate_wer(row['text'], candidate)
        if predictions is not None and not math.isclose(wer, float(predictions[index]['wer']), abs_tol=1e-10):
            raise ValueError(f'Prediction WER mismatch at source index {index}')
        row.update(generated_transcription=candidate, wer=wer, label_wer=wer,
                   action='RESPOND' if wer <= args.wer_threshold else 'REJECT')
        rows.append(row)
    metadata = dict(manifest_schema=SCHEMA, normalization=NORMALIZATION,
                    checkpoint_id=args.checkpoint, generation_config=config,
                    wer_threshold=args.wer_threshold, total_rows=len(rows),
                    counts=dict(Counter(r['action'] for r in rows)),
                    source_manifest=str(source.resolve()), source_sha256=fingerprint(source),
                    source_metadata=source_meta,
                    source_candidate_field='predictions.hyp' if predictions is not None else args.candidate_field)
    if predictions is not None:
        metadata.update(predictions=str(Path(args.predictions).resolve()),
                        predictions_sha256=fingerprint(args.predictions),
                        prediction_summary=summary,
                        prediction_summary_sha256=fingerprint(args.prediction_summary))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows))
    metadata['manifest_sha256'] = fingerprint(output)
    sidecar.write_text(json.dumps(metadata, indent=2)+'\n')
    validate_judge_manifest(output, args.checkpoint, args.wer_threshold)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
