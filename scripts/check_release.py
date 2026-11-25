#!/usr/bin/env python3
"""Check Python syntax, release isolation, and the focused file inventory."""
import ast
import json
from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parents[1]
errors = []
python_files = []
excluded = {'.cache', '.git', '__pycache__', 'external', 'outputs', 'weights'}
paths = []
for directory, dirs, files in os.walk(root):
    dirs[:] = [name for name in dirs if name not in excluded and not name.startswith('.venv')]
    paths.extend(Path(directory) / name for name in (*dirs, *files))
for path in paths:
    if path.is_symlink():
        errors.append(f'Symlink: {path.relative_to(root)}')
    if not path.is_file() or '__pycache__' in path.parts or '.cache' in path.parts:
        continue
    if path.suffix in {'.png', '.jpg', '.jpeg', '.pyc'}:
        continue
    try:
        source = path.read_text(encoding='utf-8')
    except UnicodeError:
        continue
    forbidden = '/apdcephfs/' + 'private_'
    if forbidden in source:
        errors.append(f'Private mount reference: {path.relative_to(root)}')
    if path.suffix == '.py':
        python_files.append(str(path.relative_to(root)))
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))
report = {'status': 'passed' if not errors else 'failed',
          'python_files': len(python_files), 'errors': errors}
print(json.dumps(report, indent=2))
if errors:
    sys.exit(1)
