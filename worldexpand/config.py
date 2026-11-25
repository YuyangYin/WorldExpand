"""External JSON path configuration; this module uses the standard library only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str, base: Path, *, executable: bool = False) -> str:
    value = os.path.expanduser(os.path.expandvars(str(value)))
    if re.search(r'\$(?:\w+|\{[^}]+\})', value):
        raise ValueError(f'Unresolved environment variable in path: {value}')
    path = Path(value)
    path = path if path.is_absolute() else base / path
    # The executable path is what tells Python where to find pyvenv.cfg.
    return os.path.abspath(path) if executable else str(path.resolve())


def load_config(filename: str | Path) -> dict:
    filename = Path(filename).expanduser().resolve()
    with filename.open(encoding='utf-8') as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict) or not isinstance(cfg.get('paths'), dict):
        raise ValueError('External config must contain a paths object')
    cfg['paths'] = {
        key: resolve_path(os.environ.get('WE_' + key.upper(), value), filename.parent,
                          executable=key.endswith('python') or key == 'accelerate')
        for key, value in cfg['paths'].items()
        if value is not None or 'WE_' + key.upper() in os.environ
    }
    cfg['_config_file'] = str(filename)
    return cfg


def validate_case_id(value: str) -> str:
    if not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError(f'Case ID must be a single directory name: {value!r}')
    return value


def expected_outputs(output: Path, case_ids: list[str], num_views: int = 4) -> list[Path]:
    return [output / validate_case_id(case) / f'aux_{view}' / 'output.png'
            for case in case_ids for view in range(num_views)]
