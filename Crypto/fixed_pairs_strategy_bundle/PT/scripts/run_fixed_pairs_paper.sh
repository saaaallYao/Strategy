#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_ROOT="$(cd "${ROOT_DIR}/../fixed_pairs_strategy_bundle" && pwd)"  # contains pairs_strategy
# We no longer rely on fixed_pairs_pt for data; keep bundle for pairs_strategy.
export PYTHONPATH="${ROOT_DIR}:${BUNDLE_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${ROOT_DIR}/.mpl_cache"

exec python3 "${ROOT_DIR}/scripts/paper_trader_fixed_pairs.py" "$@"
