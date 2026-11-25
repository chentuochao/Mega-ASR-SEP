#!/usr/bin/env python3
"""Score pipeline responses with pinned VoiceBench sd-QA rules."""
import argparse
import json
from pathlib import Path
from qa_pipeline import read_jsonl
from voicebench_sdqa import score_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--policies', nargs='+', required=True)
    parser.add_argument('--resources', required=True, type=Path)
    parser.add_argument('--out-records', required=True, type=Path)
    parser.add_argument('--out-summary', required=True, type=Path)
    parser.add_argument('--gpt-mode', choices=['auto','required','off'], default='auto')
    parser.add_argument('--grader-endpoint', default='https://api.openai.com/v1/chat/completions')
    parser.add_argument('--grader-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.out_records.exists() or args.out_summary.exists():
        parser.error('Select new output paths')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[:args.limit]
    records = read_jsonl(args.records)
    summary = score_records(rows, records, args.policies, args.resources, gpt_mode=args.gpt_mode,
        key_env=args.grader_key_env, endpoint=args.grader_endpoint, cache_path=args.cache)
    with args.out_records.open('x') as handle:
        handle.writelines(json.dumps(record, allow_nan=False)+'\n' for record in records)
    with args.out_summary.open('x') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(summary['policies'], indent=2))
    return 2 if any(value['grading_errors'] for value in summary['policies'].values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
