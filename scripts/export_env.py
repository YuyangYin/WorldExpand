#!/usr/bin/env python3
"""Print safely quoted shell exports from an external path configuration."""
import argparse
import os
from pathlib import Path
import re
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldexpand.config import load_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', required=True)
args = parser.parse_args()
cfg = load_config(args.config)
aliases = {'python': 'PY', 'accelerate': 'ACCELERATE', 'model': 'MODEL_PATH',
           'checkpoint': 'INFER_CKPT', 'output': 'INFER_OUT',
           'train_json': 'TRAIN_JSON_4VIEW', 'data_root': 'DATA_ROOT'}
for key, value in cfg['paths'].items():
    if not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', key):
        raise ValueError(f'Invalid path configuration key: {key!r}')
    print(f'export WE_{key.upper()}={shlex.quote(value)}')
    if key in aliases:
        # Explicit shell overrides have precedence over JSON defaults.
        name = aliases[key]
        print(f'export {name}={shlex.quote(os.environ.get(name, value))}')
if 'python' not in cfg['paths'] and 'PY' not in os.environ:
    print(f'export PY={shlex.quote(sys.executable)}')
print(f'export WE_PATH_CONFIG={shlex.quote(cfg["_config_file"])}')
