#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"
exec "$PY" "$ROOT/scripts/run.py" train --config "$WE_CONFIG_FILE" "$@"
