#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mpl_cache"

# Python script puts needed paths on sys.path; no PYTHONPATH required.
exec python3 "${PROJECT_ROOT}/scripts/paper_trader_fixed_pairs_win.py" "$@"
