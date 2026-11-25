#!/usr/bin/env python3
"""Run frozen production ASR, binary judge, then a connected text LM."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from qa_pipeline import (ChatClient, POLICIES, evaluate_qa, fingerprint, read_jsonl,
                         summarize_qa, validate_rows, distribution, compare_rejections)
from flag_wer_audio import add_flag_arguments, check_flag_arguments, export_flagged_audio, record_wer
from serial_model_stages import run_model_stage

ROOT = Path(__file__).resolve().parents[1]
ASR = ROOT / 'src/MegaASR/A2S-SFT'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', help='action_judge_v1 checkpoint')
    source.add_argument('--asr-results-dir', type=Path, help='Replay checksummed eval_judge pipeline results')
    p.add_argument('--out-dir', required=True, type=Path, help='New directory; existing results are never overwritten')
    qa_source = p.add_mutually_exclusive_group(required=True)
    qa_source.add_argument('--qa-endpoint', help='Full chat-completions URL')
    qa_source.add_argument('--qa-model-path', type=Path, help='Direct inference from local text-model weights; no server')
    p.add_argument('--qa-model', help='Required for HTTP; optional name override for direct inference')
    p.add_argument('--qa-backend', choices=['vllm', 'transformers'], default='vllm', help='Direct inference backend')
    p.add_argument('--qa-device', choices=['cpu', 'mps', 'cuda'], default='cpu', help='Direct Transformers device')
    p.add_argument('--qa-max-model-len', type=int, default=8192, help='Direct model context limit')
    p.add_argument('--qa-gpu-memory-utilization', type=float, default=.30, help='Direct vLLM memory budget')
    p.add_argument('--qa-enforce-eager', action='store_true', help='Skip vLLM graph compilation for initial evaluation')
    p.add_argument('--qa-tensor-parallel-size', type=int, default=1, help='GPUs used by the response model')
    p.add_argument('--qa-dtype', choices=['auto', 'float16', 'bfloat16'], default='auto', help='Direct vLLM compute dtype')
    p.add_argument('--qa-api-key-env', default='QA_API_KEY')
    p.add_argument('--qa-max-tokens', type=int, default=256)
    p.add_argument('--qa-temperature', type=float, default=0)
    p.add_argument('--qa-timeout', type=float, default=120)
    p.add_argument('--policies', nargs='+', choices=POLICIES, default=['judge'])
    p.add_argument('--limit', type=int)
    p.add_argument('--decoding-mode', choices=['constrained', 'unconstrained'], default='unconstrained')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--device')
    p.add_argument('--base-model-path')
    p.add_argument('--production-adapter-path')
    p.add_argument('--asr-python', type=Path, default=Path(sys.executable), help='Production ASR environment interpreter')
    p.add_argument('--qa-eval-python', type=Path, default=Path(sys.executable), help='Separate VoiceBench scoring environment')
    p.add_argument('--qa-python', type=Path, default=Path(sys.executable), help='Response worker interpreter or container wrapper')
    p.add_argument('--qa-resume-dir', type=Path, help='Reuse a validated prefix from an interrupted direct QA stage')
    p.add_argument('--local-grader-model-path', type=Path, help='Local QA correctness model; loaded after the response worker exits')
    p.add_argument('--local-grader-python', type=Path, help='Grader worker interpreter; default: --qa-python')
    p.add_argument('--local-grader-max-model-len', type=int, default=8192)
    p.add_argument('--local-grader-gpu-memory-utilization', type=float, default=.90)
    p.add_argument('--local-grader-max-tokens', type=int, default=1024)
    p.add_argument('--local-grader-enforce-eager', action='store_true')
    p.add_argument('--local-grader-tensor-parallel-size', type=int, default=1)
    p.add_argument('--local-grader-dtype', choices=['auto', 'float16', 'bfloat16'], default='auto')
    p.add_argument('--gpu-audit', action='store_true', help='Require the allocated GPU to be idle between model processes')
    p.add_argument('--voicebench-resources', type=Path)
    p.add_argument('--voicebench-gpt-mode', choices=['auto', 'required', 'off'], default='auto')
    p.add_argument('--grader-endpoint', default='https://api.openai.com/v1/chat/completions')
    p.add_argument('--grader-key-env', default='OPENAI_API_KEY')
    add_flag_arguments(p)
    args = p.parse_args(argv)
    if args.qa_tensor_parallel_size < 1 or args.local_grader_tensor_parallel_size < 1:
        p.error('Model tensor-parallel sizes must be positive')
    if args.qa_backend != 'vllm' and (args.qa_tensor_parallel_size != 1 or args.qa_dtype != 'auto'):
        p.error('Explicit response dtype and tensor parallelism require the vLLM backend')
    check_flag_arguments(p, args, [args.manifest, args.out_dir])
    rows = read_jsonl(args.manifest)
    validate_rows(rows)
    if args.limit is not None:
        if args.limit < 1:
            p.error('--limit must be positive')
        rows = rows[:args.limit]
    if len(set(args.policies)) != len(args.policies):
        p.error('Select each policy only once')
    if args.qa_endpoint and not args.qa_model:
        p.error('--qa-endpoint requires --qa-model')
    if args.qa_resume_dir and (not args.qa_model_path or not args.asr_results_dir):
        p.error('--qa-resume-dir requires managed direct QA and checksummed --asr-results-dir')
    voicebench = any(row['qa_task'] == 'voicebench_sdqa' for row in rows)
    if voicebench and (not all(row['qa_task'] == 'voicebench_sdqa' for row in rows) or not args.voicebench_resources):
        p.error('VoiceBench sd-QA needs uniform task rows and --voicebench-resources')
    if args.local_grader_model_path and (not voicebench or not args.qa_model_path or args.voicebench_gpt_mode != 'off'):
        p.error('Local grading requires VoiceBench rows, a managed --qa-model-path, and --voicebench-gpt-mode off')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    results = args.asr_results_dir or args.out_dir / 'asr_action'
    if args.checkpoint:
        command = [str(args.asr_python), str(ASR/'eval_judge.py'), '--checkpoint', args.checkpoint,
                   '--manifest', str(args.manifest.resolve()), '--out-dir', str(results.resolve()),
                   '--mode', 'pipeline', '--decoding-mode', args.decoding_mode,
                   '--max-new-tokens', str(args.max_new_tokens), '--measure-time', '--quiet']
        for flag, value in [('--limit', args.limit), ('--device', args.device),
                            ('--base-model-path', args.base_model_path),
                            ('--production-adapter-path', args.production_adapter_path)]:
            if value is not None:
                command.extend([flag, str(value)])
        # A separate process releases ASR GPU memory before QA requests start.
        run_model_stage('asr_action', command, args.out_dir/'asr_action.log', args.out_dir/'model_stages.jsonl',
                        gpu_audit=args.gpu_audit)
    asr_summary = json.loads((results/'summary.json').read_text())
    if asr_summary.get('mode') != 'pipeline' or asr_summary.get('manifest_sha256') != fingerprint(args.manifest):
        raise ValueError('ASR results must be pipeline results for this exact manifest checksum')
    predictions = read_jsonl(results/'persample.jsonl')
    if asr_summary.get('predictions_sha256') != fingerprint(results/'persample.jsonl'):
        raise ValueError('ASR predictions checksum mismatch')
    if len(predictions) != len(rows) or asr_summary.get('n_samples') != len(rows):
        raise ValueError('ASR results and selected manifest must cover exactly the same rows')
    flagged_audio = None
    if args.flagged_dir is not None:
        flag_threshold = args.flag_wer_threshold if args.flag_wer_threshold is not None else asr_summary['wer_threshold']
        flagged_audio = export_flagged_audio(predictions, args.flagged_dir, threshold=flag_threshold,
            input_manifest=args.manifest, source_manifest=args.flag_source_manifest,
            include_clean_reference=args.flag_clean_reference, records_path=results/'persample.jsonl')
        for prediction in predictions:
            prediction.update(wer_flagged=record_wer(prediction) > flag_threshold, flag_wer_threshold=flag_threshold)
    # The parent never loads a model. The response process exits before grading.
    qa_load_started = time.perf_counter()
    generation_path = args.out_dir/('generation.jsonl' if voicebench else 'persample.jsonl')
    if args.qa_model_path:
        predictions_path = args.out_dir/'qa_input_predictions.jsonl'
        predictions_path.write_text(''.join(json.dumps(record, allow_nan=False)+'\n' for record in predictions))
        config = dict(manifest=str(args.manifest.resolve()), limit=args.limit, input_records=str(predictions_path.resolve()),
            model_path=str(args.qa_model_path.resolve()), model_name=args.qa_model,
            model_revision=json.loads((args.qa_model_path/'model_provenance.json').read_text()).get('revision')
                if (args.qa_model_path/'model_provenance.json').exists() else None,
            backend=args.qa_backend, device=args.qa_device, max_model_len=args.qa_max_model_len,
            max_tokens=args.qa_max_tokens, temperature=args.qa_temperature, voicebench=voicebench,
            policies=args.policies, output_records=str(generation_path.resolve()),
            resume_dir=str(args.qa_resume_dir.resolve()) if args.qa_resume_dir else None,
            summary=str((args.out_dir/'qa_worker_summary.json').resolve()),
            engine_options=dict(gpu_memory_utilization=args.qa_gpu_memory_utilization, max_num_seqs=1,
                tensor_parallel_size=args.qa_tensor_parallel_size, dtype=args.qa_dtype,
                max_num_batched_tokens=min(2048, args.qa_max_model_len), enforce_eager=args.qa_enforce_eager,
                enable_prefix_caching=False))
        config_path = args.out_dir/'qa_worker_config.json'
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        run_model_stage('qa_respond', [args.qa_python, ROOT/'scripts/qa_model_worker.py',
            '--task', 'respond', '--config', config_path.resolve()], args.out_dir/'qa_worker.log',
            args.out_dir/'model_stages.jsonl', gpu_audit=args.gpu_audit)
        worker_summary = json.loads((args.out_dir/'qa_worker_summary.json').read_text())
        client_metadata = worker_summary['client']
        qa_load_seconds = worker_summary['load_seconds']
        records = read_jsonl(generation_path)
    else:
        client = ChatClient(args.qa_endpoint, args.qa_model, args.qa_api_key_env,
                            args.qa_timeout, args.qa_max_tokens, args.qa_temperature, allow_length=voicebench)
        qa_load_seconds = time.perf_counter()-qa_load_started
        records = evaluate_qa(rows, predictions, client, args.policies, generation_path)
        client_metadata = client.metadata()
    voicebench_summary = None
    grading_exit = 0
    scoring_seconds = None
    if voicebench:
        scoring_started = time.perf_counter()
        scored_path = args.out_dir/('pedant.jsonl' if args.local_grader_model_path else 'persample.jsonl')
        scoring = [str(args.qa_eval_python), str(ROOT/'scripts/eval_voicebench_qa.py'),
            '--manifest', str(args.manifest.resolve()), '--records', str(generation_path.resolve()),
            '--policies', *args.policies, '--resources', str(args.voicebench_resources.resolve()),
            '--out-records', str(scored_path.resolve()),
            '--out-summary', str((args.out_dir/'voicebench_summary.json').resolve()),
            '--gpt-mode', args.voicebench_gpt_mode, '--grader-endpoint', args.grader_endpoint,
            '--grader-key-env', args.grader_key_env, '--cache', str((args.out_dir/'gpt_cache.jsonl').resolve())]
        if args.limit is not None:
            scoring.extend(['--limit', str(args.limit)])
        with (args.out_dir/'voicebench_scoring.log').open('x') as log:
            process = subprocess.run(scoring, stdout=log, stderr=subprocess.STDOUT)
        grading_exit = process.returncode
        if grading_exit not in (0, 2):
            raise RuntimeError('VoiceBench scoring failed; inspect voicebench_scoring.log')
        records = read_jsonl(scored_path)
        voicebench_summary = json.loads((args.out_dir/'voicebench_summary.json').read_text())
        scoring_seconds = time.perf_counter()-scoring_started
    local_grading = None
    if args.local_grader_model_path:
        config = dict(manifest=str(args.manifest.resolve()), limit=args.limit, input_records=str(scored_path.resolve()),
            model_path=str(args.local_grader_model_path.resolve()), backend='vllm',
            max_model_len=args.local_grader_max_model_len, max_tokens=args.local_grader_max_tokens,
            policies=args.policies, output_records=str((args.out_dir/'persample.jsonl').resolve()),
            votes_path=str((args.out_dir/'local_grader_votes.jsonl').resolve()),
            summary=str((args.out_dir/'local_grader_summary.json').resolve()),
            engine_options=dict(gpu_memory_utilization=args.local_grader_gpu_memory_utilization, max_num_seqs=1,
                tensor_parallel_size=args.local_grader_tensor_parallel_size, dtype=args.local_grader_dtype,
                max_num_batched_tokens=min(2048, args.local_grader_max_model_len),
                enforce_eager=args.local_grader_enforce_eager, enable_prefix_caching=False))
        config_path = args.out_dir/'local_grader_config.json'
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        stage = run_model_stage('qa_correctness_grade',
            [args.local_grader_python or args.qa_python, ROOT/'scripts/qa_model_worker.py', '--task', 'grade',
             '--config', config_path.resolve()], args.out_dir/'local_grader.log',
            args.out_dir/'model_stages.jsonl', gpu_audit=args.gpu_audit, accepted_codes=(0, 2))
        local_grading = json.loads((args.out_dir/'local_grader_summary.json').read_text())
        records = read_jsonl(args.out_dir/'persample.jsonl')
        if stage['returncode']:
            grading_exit = 2
    if voicebench:
        from voicebench_sdqa import protocol
        client_metadata.update(system_prompt=protocol()['audio_system'],
            reference_text_system_prompt=protocol()['text_system'],
            unscored_system_prompt=None, choice_prompt=None, protocol='voicebench_naive_cascade')
    qa_summary = summarize_qa(records, args.policies)
    summary = dict(schema='asr_judge_qa_v1', generated_utc=datetime.now(timezone.utc).isoformat(),
                   manifest=str(args.manifest.resolve()), manifest_sha256=fingerprint(args.manifest),
                   asr_results_dir=str(results.resolve()),
                   asr_predictions_sha256=fingerprint(results/'persample.jsonl'),
                   source_sha256={str(path.relative_to(ROOT)):fingerprint(path) for path in
                       [Path(__file__).resolve(), ROOT/'scripts/qa_pipeline.py', ASR/'eval_judge.py',
                        ASR/'judge_inference.py', ROOT/'src/MegaASR/model/text_response.py',
                        ROOT/'scripts/voicebench_sdqa.py', ROOT/'scripts/eval_voicebench_qa.py',
                        ROOT/'scripts/flag_wer_audio.py', ROOT/'scripts/serial_model_stages.py',
                        ROOT/'scripts/qa_model_worker.py', ROOT/'scripts/local_qa_grader.py']},
                   qa_client=client_metadata, policies=args.policies, limit=args.limit,
                   qa_resume=worker_summary.get('resume') if args.qa_model_path else None,
                   qa_load_seconds=qa_load_seconds,
                   qa_evaluation_seconds=scoring_seconds,
                   asr_action=asr_summary, qa=qa_summary, flagged_audio=flagged_audio,
                   rejection_comparison=compare_rejections(records, args.policies),
                   stage_seconds={stage:distribution(r.get('stage_seconds', {}).get(stage) for r in records)
                                  for stage in ('asr', 'judge')},
                   run_wall_seconds=time.perf_counter()-started,
                   status='qa_errors' if any(q['n_errors'] for q in qa_summary.values()) else 'grading_errors' if grading_exit else 'complete',
                   scoring='voicebench_sdqa_v1' if voicebench else 'strict_final_answer_v1; not the official VoiceBench evaluator',
                   voicebench=voicebench_summary, local_grading=local_grading,
                   model_stages=read_jsonl(args.out_dir/'model_stages.jsonl') if (args.out_dir/'model_stages.jsonl').exists() else [],
                   model_loading='Separate serial processes; each process group exits before the next model loads',
                   rejection_policy='no QA call; unanswered, not a correct answer',
                   latency_scope='serial ASR/judge compute plus direct QA generation or QA HTTP round trip; '
                                 'excludes separation, audio IO, model loading, and user clarification')
    (args.out_dir/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    # Retain original benchmark metadata for official evaluators, accepted rows only.
    for policy in args.policies:
        with (args.out_dir/f'benchmark-{policy}-answered.jsonl').open('x') as handle:
            for row, record in zip(rows, records):
                if record['qa'][policy]['status'] == 'answered':
                    exported = dict(row.get('benchmark_record', {}), sample_id=row['sample_id'],
                                    response=record['qa'][policy]['response'])
                    votes = record['qa'][policy].get('voicebench', {}).get('score')
                    if votes is not None:
                        exported['score'] = votes
                    handle.write(json.dumps(exported, allow_nan=False)+'\n')
    print(json.dumps(dict(status=summary['status'], qa=qa_summary), indent=2))
    return 2 if summary['status'] != 'complete' else 0


if __name__ == '__main__':
    raise SystemExit(main())
