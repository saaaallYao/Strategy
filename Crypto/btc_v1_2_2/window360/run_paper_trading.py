#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper trading runner using Alpaca data via the live_monitor logic.
"""
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fina.strategies.crypto.btc_eth_v1.live_monitor import LiveMonitor
from fina.strategies.crypto.btc_eth_v1.strategy_engine import StatArbEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper trading with live_monitor logic.")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run a single update/execute cycle")
    parser.add_argument("--fee", type=float, default=5e-4, help="Trading fee (per side)")
    parser.add_argument("--min-edge", type=float, default=0.0014, help="Minimum edge return for entry")
    parser.add_argument("--dyn-edge", action="store_true", default=None, help="Enable dynamic fee-aware edge filter")
    parser.add_argument("--dyn-edge-fee-mult", type=float, default=5.0, help="Fee multiplier for dyn edge")
    parser.add_argument("--dyn-edge-vol-mult", type=float, default=0.5, help="Vol multiplier for dyn edge")
    parser.add_argument("--fee-exit", action="store_true", default=None, help="Enable fee-aware exit")
    parser.add_argument("--fee-exit-mult", type=float, default=2.0, help="Fee multiplier for exit hold")
    parser.add_argument("--stop-loss", type=float, default=0.010, help="Stop-loss pct (e.g., 0.010 for 1%%)")
    parser.add_argument("--z-enter", type=float, default=1.2, help="Z-score entry threshold")
    parser.add_argument("--z-exit", type=float, default=0.4, help="Z-score exit threshold")
    parser.add_argument("--signal-persistence", type=int, default=3, help="Bars beyond threshold before entry")
    parser.add_argument("--min-hold-bars", type=int, default=30, help="Minimum hold time in minutes")
    parser.add_argument("--cooldown-bars", type=int, default=30, help="Cooldown time in minutes")
    return parser.parse_args()


def apply_config(monitor: LiveMonitor, args: argparse.Namespace) -> None:
    cfg = dict(monitor.strategy_config)
    dyn_edge_enabled = args.dyn_edge if args.dyn_edge is not None else True
    fee_exit_enabled = args.fee_exit if args.fee_exit is not None else True
    cfg.update({
        "fee": args.fee,
        "min_edge_return": args.min_edge,
        "dyn_edge_enabled": dyn_edge_enabled,
        "dyn_edge_fee_mult": args.dyn_edge_fee_mult,
        "dyn_edge_vol_mult": args.dyn_edge_vol_mult,
        "fee_exit_enabled": fee_exit_enabled,
        "fee_exit_mult": args.fee_exit_mult,
        "stop_loss_pct": args.stop_loss,
        "z_enter": args.z_enter,
        "z_exit": args.z_exit,
        "signal_persistence": args.signal_persistence,
        "min_hold_bars": args.min_hold_bars,
        "cooldown_bars": args.cooldown_bars,
    })
    monitor.strategy_config = cfg
    monitor.engine = StatArbEngine(cfg)


def main() -> int:
    args = parse_args()
    monitor = LiveMonitor()
    apply_config(monitor, args)

    try:
        while True:
            px, etf_px = monitor.update_data()
            if px is None or px.empty:
                time.sleep(args.interval)
                if args.once:
                    break
                continue
            monitor.execute_strategy(px, etf_px)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
