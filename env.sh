#!/usr/bin/env bash
# Source this helper; resources come from external JSON, with WE_* overrides.
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WE_CONFIG_FILE="${WE_CONFIG_FILE:-${REPO_ROOT}/configs/release/local.json}"
_we_exports=$(python3 "${REPO_ROOT}/scripts/export_env.py" --config "$WE_CONFIG_FILE") || return 1
# export_env.py uses shlex.quote for every value.
eval "$_we_exports"
unset _we_exports
export PYTHONPATH="${REPO_ROOT}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1
