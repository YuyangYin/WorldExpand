"""Resolve source defaults through an external prefix map.

WE_PATH_CONFIG is the same JSON used by scripts/run.py. Its optional
path_mappings object maps absolute prefixes to local mounts. The longest
matching prefix wins. No filesystem monkey-patching or symlinks are used.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path

from .config import REPO_ROOT, resolve_path


def map_prefix(value: str, mapping: dict) -> str:
    for prefix in sorted(mapping, key=len, reverse=True):
        if value == prefix or value.startswith(prefix.rstrip('/') + '/'):
            return str(mapping[prefix]).rstrip('/') + value[len(prefix):]
    return value


@lru_cache(maxsize=8)
def _mappings(filename: str):
    if not filename:
        return {}
    path = Path(filename).expanduser().resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    return {key.rstrip('/'): resolve_path(value, path.parent)
            for key, value in cfg.get('path_mappings', {}).items()}


def resolve_resource(value: str) -> str:
    value = str(value)
    if value == '@repo' or value.startswith('@repo/'):
        value = str(REPO_ROOT) + value[5:]
    return os.path.expanduser(os.path.expandvars(map_prefix(
        value, _mappings(os.environ.get('WE_PATH_CONFIG', '')))))
