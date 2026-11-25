#!/usr/bin/env python3
"""Download pinned response-model weights and record transfer checksums."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from MegaASR.model.text_response import validate_text_model_config


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--revision', help='Optional full commit; otherwise resolve main once')
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.revision and not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('--revision must be a full commit identifier')
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(output / '.cache/hf_home'))
    os.environ.setdefault('HF_XET_CACHE', str(output / '.cache/xet'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    info = HfApi().model_info(args.model, revision=args.revision or 'main', files_metadata=True)
    revision = info.sha
    provenance = output / 'model_provenance.json'
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        if (previous['model'], previous['revision']) != (args.model, revision):
            raise ValueError('Select a new model directory; existing provenance differs')
    # Validate the small configuration before downloading model weights.
    config_path = hf_hub_download(args.model, 'config.json', revision=revision, local_dir=str(output))
    validate_text_model_config(json.loads(Path(config_path).read_text()))
    names = [item.rfilename for item in info.siblings if (
        item.rfilename.endswith(('.json', '.safetensors', '.model', '.jinja'))
        or item.rfilename in ('merges.txt', 'vocab.txt', 'README.md', 'LICENSE', 'LICENSE.txt'))
        and '/' not in item.rfilename]
    if not any(name.endswith('.safetensors') for name in names):
        raise ValueError('Expected safetensors model weights')
    print(json.dumps(dict(status='downloading', model=args.model, revision=revision,
                          out_dir=str(output), files=names)), flush=True)
    snapshot_download(args.model, revision=revision, local_dir=str(output), allow_patterns=names)
    files = {name: sha256(output/name) for name in sorted(names)}
    for item in info.siblings:
        if item.rfilename in files and item.lfs:
            expected = item.lfs.sha256 if hasattr(item.lfs, 'sha256') else item.lfs['sha256']
            if files[item.rfilename] != expected:
                raise ValueError(f'Upstream weight checksum mismatch: {item.rfilename}')
    record = dict(schema='response_model_download_v1', model=args.model, revision=revision,
                  downloaded_utc=datetime.now(timezone.utc).isoformat(),
                  source=f'https://huggingface.co/{args.model}/tree/{revision}',
                  files_sha256=files, total_bytes=sum((output/name).stat().st_size for name in names))
    provenance.write_text(json.dumps(record, indent=2)+'\n')
    checksums = dict(files)
    checksums['model_provenance.json'] = sha256(provenance)
    (output/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(checksums.items())))
    print(json.dumps(dict(status='verified', model=args.model, revision=revision,
                          files=len(files), total_bytes=record['total_bytes'], out_dir=str(output))), flush=True)


if __name__ == '__main__':
    main()
