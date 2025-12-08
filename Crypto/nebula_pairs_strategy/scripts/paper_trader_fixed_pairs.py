#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
CODE_DIR = PROJECT_ROOT / "code"

# Ensure the code directory (where all modules live) is on sys.path.
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from live_trading_fixed_pairs import LiveFixedPairsTrader, configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nebula Pairs paper trader (KuCoin spot klines).")
    parser.add_argument("--base-symbol", default=os.environ.get("FP_BASE_SYMBOL", "BTC-USDT"), help="Base asset symbol (default: BTC-USDT)")
    parser.add_argument("--pairs", nargs="+", default=None, help="Pairs to trade (KuCoin symbols). Defaults to strategy core PAIRS.")
    parser.add_argument("--resample-rule", default=os.environ.get("FP_RESAMPLE_RULE", "15min"), help="Resample rule (default: 15min)")
    parser.add_argument("--seed-days", type=int, default=int(os.environ.get("FP_SEED_DAYS", "200")), help="History days to bootstrap (default: 200)")
    parser.add_argument("--initial-capital", type=float, default=float(os.environ.get("FP_INITIAL_CAPITAL", "1000000")), help="Initial equity (default: 1,000,000)")
    parser.add_argument("--cushion-bars", type=int, default=int(os.environ.get("FP_CUSHION_BARS", "5")), help="Bars cushion when refetching (default: 5)")
    parser.add_argument("--bar-grace-seconds", type=int, default=int(os.environ.get("FP_BAR_GRACE_SECONDS", "5")), help="Seconds after bar close before polling (default: 5)")
    parser.add_argument("--max-bars", type=int, default=int(os.environ.get("FP_MAX_BARS", "0")), help="Max trailing bars to keep (0 = keep all)")
    parser.add_argument("--log-prefix", default=os.environ.get("FP_LOG_PREFIX", "nebula_pairs"), help="Prefix for log filenames (default: nebula_pairs)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    trader = LiveFixedPairsTrader(
        base_symbol=args.base_symbol,
        pairs=args.pairs,
        resample_rule=args.resample_rule,
        seed_days=args.seed_days,
        initial_capital=args.initial_capital,
        cushion_bars=args.cushion_bars,
        poll_grace_seconds=args.bar_grace_seconds,
        max_bars=args.max_bars,
        output_prefix=args.log_prefix,
    )
    trader.run_forever()


if __name__ == "__main__":
    main()
