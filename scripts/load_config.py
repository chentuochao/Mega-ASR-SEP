#!/usr/bin/env python3
"""Print `export KEY=value` lines for a YAML config so a calling shell script
can `eval` them. Generic and reusable across the finetune_*.sh scripts: every
top-level YAML key maps to KEY_UPPER (e.g. train_jsonl -> TRAIN_JSONL), and
any key already present in the environment is left untouched (explicit env
vars always win over the config file).

A top-level key whose value is a dict (e.g. a nested "ft_params" block, see
scripts/configs/example_ft_params.yaml) is JSON-serialized into ONE env var
instead of str()'d -- str(dict) produces Python repr (single-quoted, True/
False/None) which downstream JSON parsing (arguments.py's --ft_params,
consumed by modeling.py's parse_ft_params) can't read. This does mean a
nested block can only be overridden as a whole via env var (e.g.
FT_PARAMS='{"encoder": ...}'), not field-by-field like the flat scalar keys
below -- appropriate here since the region config is one coherent unit, not
independent knobs."""
import json
import os
import shlex
import sys

import yaml

config_path = sys.argv[1]
with open(config_path) as f:
    config = yaml.safe_load(f) or {}

for key, value in config.items():
    env_key = key.upper()
    if env_key in os.environ:
        continue
    if isinstance(value, dict):
        print(f"export {env_key}={shlex.quote(json.dumps(value))}")
        continue
    value = os.path.expandvars(str(value))
    print(f"export {env_key}={shlex.quote(value)}")
