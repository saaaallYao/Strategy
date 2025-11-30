#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo root
CODE_DIR="${PROJECT_ROOT}/code"
# All dependencies live under code/.
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mpl_cache"

exec python3 "${PROJECT_ROOT}/scripts/paper_trader_fixed_pairs.py" "$@"
