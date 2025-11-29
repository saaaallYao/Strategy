#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from code.live_trading_residual import (
    LiveResidualPairsTrader,
    configure_logging,
    default_universe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run residual pairs market-neutral paper trader (KuCoin spot).")
    parser.add_argument(
        "--universe",
        type=str,
        default=os.environ.get("RESIDUAL_UNIVERSE", ""),
        help="Comma-separated KuCoin symbols (e.g., BTC-USDT,ETH-USDT,SOL-USDT). Default uses built-in list.",
    )
    parser.add_argument("--poll-interval", type=int, default=int(os.environ.get("RESIDUAL_POLL_INTERVAL", "60")), help="Polling interval in seconds (default: 60)")
    parser.add_argument("--history-minutes", type=int, default=int(os.environ.get("RESIDUAL_HISTORY_MINUTES", "120000")), help="Minutes of 1m history to keep (default: 120000 ~83 days)")
    parser.add_argument("--fee-per-side", type=float, default=float(os.environ.get("RESIDUAL_FEE_PER_SIDE", "0.0003")), help="Fee per side (fraction, default: 0.0003)")
    parser.add_argument("--lookback-days", type=int, default=int(os.environ.get("RESIDUAL_LOOKBACK_DAYS", "20")), help="Beta lookback in days (default: 20)")
    parser.add_argument("--zwin-days", type=int, default=int(os.environ.get("RESIDUAL_ZWIN_DAYS", "45")), help="Z-score window in days (default: 45)")
    parser.add_argument("--k-per-side", type=int, default=int(os.environ.get("RESIDUAL_K_PER_SIDE", "3")), help="Max longs/shorts per side (default: 3)")
    parser.add_argument("--z-threshold", type=float, default=float(os.environ.get("RESIDUAL_Z_THRESHOLD", "0.8")), help="|Z| entry threshold (default: 0.8)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    universe = [s.strip() for s in args.universe.split(",") if s.strip()] if args.universe else default_universe()
    trader = LiveResidualPairsTrader(
        universe=universe,
        poll_interval=args.poll_interval,
        history_minutes=args.history_minutes,
        fee_per_side=args.fee_per_side,
        lookback_days=args.lookback_days,
        zwin_days=args.zwin_days,
        k_per_side=args.k_per_side,
        z_threshold=args.z_threshold,
    )
    trader.run_forever()


if __name__ == "__main__":
    main()
