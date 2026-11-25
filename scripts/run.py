#!/usr/bin/env python3
"""Run from any working directory; use --config to load external resource paths."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'expand':
        from reconstruction.pipeline import main
        main(sys.argv[2:])
    else:
        from worldexpand.cli import main
        main()
