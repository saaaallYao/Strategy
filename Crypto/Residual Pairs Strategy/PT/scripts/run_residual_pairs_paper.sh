#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${ROOT_DIR}/.mpl_cache"

UNIVERSE="${RESIDUAL_UNIVERSE:-}"
POLL_INTERVAL="${RESIDUAL_POLL_INTERVAL:-60}"
HISTORY_MINUTES="${RESIDUAL_HISTORY_MINUTES:-120000}"
FEE_PER_SIDE="${RESIDUAL_FEE_PER_SIDE:-0.0003}"
LOOKBACK_DAYS="${RESIDUAL_LOOKBACK_DAYS:-20}"
ZWIN_DAYS="${RESIDUAL_ZWIN_DAYS:-45}"
K_PER_SIDE="${RESIDUAL_K_PER_SIDE:-3}"
Z_THRESHOLD="${RESIDUAL_Z_THRESHOLD:-0.8}"

exec python3 "${ROOT_DIR}/scripts/paper_trader_residual.py" \
  ${UNIVERSE:+--universe "${UNIVERSE}"} \
  --poll-interval "${POLL_INTERVAL}" \
  --history-minutes "${HISTORY_MINUTES}" \
  --fee-per-side "${FEE_PER_SIDE}" \
  --lookback-days "${LOOKBACK_DAYS}" \
  --zwin-days "${ZWIN_DAYS}" \
  --k-per-side "${K_PER_SIDE}" \
  --z-threshold "${Z_THRESHOLD}"
