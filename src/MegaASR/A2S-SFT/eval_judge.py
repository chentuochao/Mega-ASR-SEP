#!/usr/bin/env python3
"""Evaluate saved candidates or production transcription followed by judging.

Pipeline mode accepts an ordinary audio/reference manifest. Cached mode requires
a prepared, checksummed candidate manifest. References are used only for metrics.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

import torch
import jiwer

import action_format
import modeling
from dataloader import read_audio
from judge_data import candidate_wer, fingerprint, normalize, validate_judge_manifest
from judge_inference import judge_candidate, transcribe_and_judge

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from flag_wer_audio import add_flag_arguments, check_flag_arguments, export_flagged_audio, validate_destination


def summarize(records, threshold):
    if not records:
        raise ValueError('Cannot evaluate an empty manifest')
    accepted = [r for r in records if r['effective_action'] == 'RESPOND']
    wers = [r['label_wer'] for r in records]
    recalls = {
        action: sum(r['effective_action'] == action for r in records if r['target_action'] == action)
        / sum(r['target_action'] == action for r in records)
        for action in action_format.ACTIONS if any(r['target_action'] == action for r in records)
    }
    accuracy = sum(r['effective_action'] == r['target_action'] for r in records)/len(records)
    return dict(
        n_samples=len(records), wer_threshold=threshold,
        action_valid_rate=sum(r['action_valid'] for r in records)/len(records),
        accuracy=accuracy, overall_action_accuracy=accuracy,
        generated_action_accuracy=sum(r['parsed_action'] == r['target_action'] for r in records)/len(records),
        corpus_wer=jiwer.wer([normalize(r['reference']) for r in records],
                             [normalize(r['generated_transcription']) for r in records]),
        mean_utterance_wer=statistics.mean(wers), median_utterance_wer=statistics.median(wers),
        response_count=len(accepted), response_rate=len(accepted)/len(records),
        respond_mean_utterance_wer=statistics.mean(r['label_wer'] for r in accepted) if accepted else None,
        respond_median_utterance_wer=statistics.median(r['label_wer'] for r in accepted) if accepted else None,
        respond_corpus_wer=jiwer.wer([normalize(r['reference']) for r in accepted],
                                     [normalize(r['generated_transcription']) for r in accepted]) if accepted else None,
        per_class_recall=recalls,
        balanced_accuracy=statistics.mean(recalls.values()) if len(recalls) == 2 else None,
        confusion_matrix={truth: {pred: sum(r['target_action'] == truth and r['effective_action'] == pred
                                           for r in records) for pred in action_format.ACTIONS}
                          for truth in action_format.ACTIONS},
    )


def evaluate(*, checkpoint, manifest, out_json, summary_json, mode='pipeline',
             decoding_mode='unconstrained', max_new_tokens=256, device=None, limit=None,
             base_model_path=None, production_adapter_path=None, use_fusion=None,
             fusion_type=None, language=None, sr=None, wer_threshold=None, quiet=False,
             measure_time=False, flagged_dir=None, flag_wer_threshold=None,
             flag_source_manifest=None, flag_clean_reference=False):
    if mode not in ('cached', 'pipeline') or decoding_mode not in action_format.DECODING_MODES:
        raise ValueError('Unknown evaluation or decoding mode')
    if max_new_tokens < 1 or (limit is not None and limit < 1):
        raise ValueError('Token and sample limits must be positive')
    destinations = [Path(out_json).resolve(), Path(summary_json).resolve()]
    if destinations[0] == destinations[1] or Path(manifest).resolve() in destinations:
        raise ValueError('Prediction, summary, and input manifest paths must be different')
    meta = action_format.read_checkpoint_metadata(checkpoint)
    if not meta or meta['action_format'] != action_format.ACTION_JUDGE_V1:
        raise ValueError('Expected an action_judge_v1 checkpoint')
    config = meta['run_config']
    threshold = config['action_wer_threshold'] if wer_threshold is None else wer_threshold
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('WER threshold must be finite and nonnegative')
    flag_threshold = threshold if flag_wer_threshold is None else flag_wer_threshold
    if flagged_dir is not None:
        validate_destination(flagged_dir, flag_threshold, [manifest, out_json, summary_json])
    elif flag_wer_threshold is not None or flag_source_manifest or flag_clean_reference:
        raise ValueError('Flag options require flagged_dir')
    for name, override in [('use_fusion', use_fusion), ('fusion_type', fusion_type)]:
        if override is not None and override != config[name]:
            raise ValueError(f'{name} differs from the trained checkpoint configuration')
    if mode == 'cached':
        validate_judge_manifest(manifest, meta['production_adapter_path'], threshold)
    rows = [(index, json.loads(line)) for index, line in enumerate(Path(manifest).read_text().splitlines())
            if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError('No samples selected for evaluation')
    for index, row in rows:
        for field in ('audio_sep', 'audio_mix', 'text'):
            if not isinstance(row.get(field), str) or (field != 'text' and not row[field]):
                raise ValueError(f'{manifest}:{index + 1}: missing or invalid {field}')
        if row.get('prompt') or row.get('quality_context'):
            raise ValueError('Judge inputs do not accept prompt or quality hints')
    base = base_model_path or meta['base_model_path']
    production = production_adapter_path or meta['production_adapter_path']
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    language = config['language'] if language is None else language
    sr = config.get('sr', 16000) if sr is None else sr
    if sr < 1:
        raise ValueError('Sample rate must be positive')
    model, processor, _ = modeling.load_action_special_adapter(
        checkpoint, meta, config['use_fusion'], config['fusion_type'], base, production)
    model.to(device).eval()
    if device == 'cpu':
        model.float()
    processor.tokenizer.padding_side = 'left'
    production_processor = modeling.load_processor(base) if mode == 'pipeline' else None
    records = []
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as handle:
        for index, row in rows:
            sep, mix = (read_audio(row[k], sr) for k in ('audio_sep', 'audio_mix'))
            if mode == 'cached':
                result = judge_candidate(model, processor, sep, mix, row['generated_transcription'],
                                         language, decoding_mode)
            else:
                result = transcribe_and_judge(model, processor, production_processor, sep, mix,
                                              language, max_new_tokens, decoding_mode,
                                              **({'measure_time': True} if measure_time else {}))
            wer = candidate_wer(row['text'], result['generated_transcription'])
            truth = 'RESPOND' if wer <= threshold else 'REJECT'
            record = dict(result, index=index, sample_id=row.get('sample_id', index),
                          audio_sep=row['audio_sep'], audio_path=row['audio_sep'],
                          audio_mix=row['audio_mix'],
                          reference=row['text'], ref=normalize(row['text']),
                          hyp=normalize(result['generated_transcription']), wer=wer,
                          label_wer=wer, target_action=truth,
                          wer_flagged=wer > flag_threshold, flag_wer_threshold=flag_threshold)
            records.append(record)
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
            if not quiet:
                print(f'[{index:04d}] WER={wer:.3f} action={record["effective_action"]} target={truth}')
    summary = dict(summarize(records, threshold), checkpoint=str(checkpoint),
                   manifest=str(manifest), manifest_sha256=fingerprint(manifest),
                   mode=mode, decoding_mode=decoding_mode,
                   base_model_path=str(base), production_adapter_path=str(production),
                   language=language, sampling_rate=sr, use_fusion=config['use_fusion'],
                   fusion_type=config['fusion_type'], do_sample=False,
                   generated_utc=datetime.now(timezone.utc).isoformat(),
                   invalid_action_policy='REJECT',
                   max_new_tokens=max_new_tokens if mode == 'pipeline' else None)
    summary['stage_timing'] = ('synchronized_asr_and_judge_excluding_audio_io_and_model_load'
                               if measure_time and mode == 'pipeline' else None)
    summary['predictions_sha256'] = fingerprint(out_json)
    if flagged_dir is not None:
        summary['flagged_audio'] = export_flagged_audio(records, flagged_dir, threshold=flag_threshold,
            input_manifest=manifest, source_manifest=flag_source_manifest,
            include_clean_reference=flag_clean_reference, records_path=out_json)
    path = Path(summary_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--mode', choices=['cached', 'pipeline'], default='cached')
    p.add_argument('--decoding-mode', choices=action_format.DECODING_MODES, default='unconstrained')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--device', default=None)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--base-model-path', default=None)
    p.add_argument('--production-adapter-path', default=None)
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--measure-time', action='store_true')
    add_flag_arguments(p)
    args = p.parse_args(argv)
    check_flag_arguments(p, args, [args.manifest, args.out_dir])
    evaluate(checkpoint=args.checkpoint, manifest=args.manifest,
             out_json=str(Path(args.out_dir)/'persample.jsonl'),
             summary_json=str(Path(args.out_dir)/'summary.json'), mode=args.mode,
             decoding_mode=args.decoding_mode, max_new_tokens=args.max_new_tokens,
             device=args.device, limit=args.limit, base_model_path=args.base_model_path,
             production_adapter_path=args.production_adapter_path, quiet=args.quiet,
             measure_time=args.measure_time, flagged_dir=args.flagged_dir,
             flag_wer_threshold=args.flag_wer_threshold, flag_source_manifest=args.flag_source_manifest,
             flag_clean_reference=args.flag_clean_reference)


if __name__ == '__main__':
    main()
