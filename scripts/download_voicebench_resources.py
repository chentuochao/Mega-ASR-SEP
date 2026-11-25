#!/usr/bin/env python3
"""Download and verify the fixed PEDANT classifier files used by the evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT/'third_party/voicebench_sdqa/PEDANT_RESOURCES.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True, type=Path, help='New resource directory')
    args = parser.parse_args()
    if args.out_dir.exists() or args.out_dir.is_symlink():
        parser.error('Select a new resource directory; existing files are never overwritten')
    provenance = json.loads(PIN.read_text())
    base = f"https://raw.githubusercontent.com/zli12321/qa_metrics/{provenance['revision']}/"
    resources = {}
    for name, source in provenance['upstream_paths'].items():
        with urlopen(base+source, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != provenance['sha256'][name]:
            raise ValueError(f'PEDANT download checksum mismatch: {name}')
        resources[name] = content
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, content in resources.items():
        with (args.out_dir/name).open('xb') as handle:
            handle.write(content)
    (args.out_dir/'PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(status='verified', revision=provenance['revision'],
                          out_dir=str(args.out_dir.resolve()), files=list(resources)), indent=2))


if __name__ == '__main__':
    main()
