#!/usr/bin/env python3
"""
Paper-trading runner that reuses the shared signal engine to ensure the live
loop matches the offline backtest.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

# Ensure repository root is importable when executed standalone and avoid masking stdlib modules.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if sys.path and sys.path[0] == str(SCRIPT_DIR):
    sys.path[0] = str(REPO_ROOT)
elif str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from fixed_pairs_pt.binanceus_client import (
    ensure_resample,
    fetch_history,
    fetch_incremental,
    seconds_until_next_bar,
)
from fixed_pairs_pt.broker import PairsPaperBroker
from pairs_strategy.core import PAIRS, StrategyConfig
from pairs_strategy.signal import PairsSignalEngine

ARTIFACT_DIR = Path("pairs_strategy") / "artifacts_paper_trading"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper trade the fixed pairs strategy using the shared signal engine.")
    parser.add_argument("--base-symbol", default="BTCUSD", help="Base asset symbol (maps to BTC_USD column).")
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS), help="Pairs to trade (Binance US symbols).")
    parser.add_argument("--resample-rule", default="15min", help="Resample rule for live bars.")
    parser.add_argument("--seed-days", type=int, default=120, help="History window (days) to bootstrap the strategy.")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0, help="Initial equity for the paper broker.")
    parser.add_argument("--output-dir", type=str, default=str(ARTIFACT_DIR), help="Directory for PT artifacts.")
    parser.add_argument("--train-split", type=str, default="2025-12-31", help="Training cut date for beta estimation.")
    parser.add_argument("--iterations", type=int, default=0, help="Number of live iterations to run after warmup (0 = infinite).")
    parser.add_argument("--cushion-bars", type=int, default=5, help="Refetch this many bars before the latest timestamp.")
    parser.add_argument("--bar-grace-seconds", type=int, default=5, help="Extra seconds to wait after the bar close before polling.")
    parser.add_argument("--max-bars", type=int, default=0, help="Keep only this many trailing bars in memory (0 = keep all).")
    return parser.parse_args()


def build_strategy_config(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        base_asset="BTC_USD",
        pairs=tuple(args.pairs),
        resample_rule=args.resample_rule,
        train_split=args.train_split,
    )


def bootstrap_history(symbols: List[str], args: argparse.Namespace) -> pd.DataFrame:
    history = fetch_history(
        symbols=symbols,
        resample_rule=args.resample_rule,
        lookback_days=args.seed_days,
    )
    history = ensure_resample(history, args.resample_rule, base_column="BTC_USD")
    return history


def init_artifacts(output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pnl_path = output_dir / "pnl.csv"
    trades_path = output_dir / "trades.csv"
    positions_path = output_dir / "positions.csv"
    with open(pnl_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "base_price", "equity", "cash", "position_value", "realized_pnl"])
    with open(trades_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trade_id", "timestamp", "symbol", "side", "qty", "price", "notional", "fee", "comment"])
    with open(positions_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "target_weight", "price"])
    return pnl_path, trades_path, positions_path


def log_pnl(pnl_path: Path, timestamp: pd.Timestamp, base_price: float, broker: PairsPaperBroker) -> None:
    snap = broker.snapshot_equity()
    with open(pnl_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                timestamp,
                round(base_price, 6),
                round(snap["equity"], 2),
                round(snap["cash"], 2),
                round(snap["position_value"], 2),
                round(snap["realized_pnl"], 2),
            ]
        )


def append_new_trades(broker: PairsPaperBroker, trades_path: Path, start_index: int) -> int:
    new_trades = broker.trade_records_since(start_index)
    if not new_trades:
        return start_index
    with open(trades_path, "a", newline="") as f:
        writer = csv.writer(f)
        for tr in new_trades:
            writer.writerow(
                [
                    tr.trade_id,
                    tr.timestamp,
                    tr.symbol,
                    tr.side,
                    round(tr.qty, 8),
                    round(tr.price, 6),
                    round(tr.notional, 2),
                    round(tr.fee, 2),
                    tr.comment,
                ]
            )
    return start_index + len(new_trades)


def log_positions(
    positions_path: Path,
    timestamp: pd.Timestamp,
    prices: pd.Series,
    target_weights: pd.Series,
) -> None:
    with open(positions_path, "a", newline="") as f:
        writer = csv.writer(f)
        for symbol, weight in target_weights.items():
            writer.writerow(
                [
                    timestamp,
                    symbol,
                    round(weight, 6),
                    round(prices.get(symbol, float("nan")), 6),
                ]
            )


def rebalance_once(
    history: pd.DataFrame,
    cfg: StrategyConfig,
    engine: PairsSignalEngine,
    broker: PairsPaperBroker,
    pnl_path: Path,
    trades_path: Path,
    positions_path: Path,
    last_trade_idx: int,
) -> int:
    latest_weights, _ = engine.latest_weights(history)
    ts = history.index[-1]
    prices = history.loc[ts, latest_weights.index]
    broker.rebalance_to_weights(latest_weights, prices, timestamp=ts)
    base_price = float(history.loc[ts, cfg.base_asset])
    log_pnl(pnl_path, ts, base_price, broker)
    log_positions(positions_path, ts, prices, latest_weights)
    new_idx = append_new_trades(broker, trades_path, last_trade_idx)
    return new_idx


def live_loop(args: argparse.Namespace) -> None:
    cfg = build_strategy_config(args)
    symbols = [args.base_symbol] + list(args.pairs)

    history = bootstrap_history(symbols, args)
    history = history[[cfg.base_asset, *cfg.pairs]].dropna()

    engine = PairsSignalEngine(cfg)
    broker = PairsPaperBroker(fee_rate=cfg.cost_rate, initial_equity=args.initial_capital)

    pnl_path, trades_path, positions_path = init_artifacts(Path(args.output_dir))

    last_trade_idx = 0
    last_trade_idx = rebalance_once(history, cfg, engine, broker, pnl_path, trades_path, positions_path, last_trade_idx)
    print(f"[bootstrap] ts={history.index[-1]} equity={broker.total_equity():,.2f}", flush=True)

    iterations_run = 1
    target_iterations = args.iterations

    while target_iterations == 0 or iterations_run < target_iterations:
        wait_seconds = seconds_until_next_bar(args.resample_rule, args.bar_grace_seconds)
        time.sleep(wait_seconds)

        last_ts = history.index[-1]
        recent = fetch_incremental(
            symbols=symbols,
            resample_rule=args.resample_rule,
            last_timestamp=last_ts,
            cushion_bars=args.cushion_bars,
        )
        if recent.empty:
            continue

        history = pd.concat([history, recent], axis=0)
        history = history[~history.index.duplicated(keep="last")]
        history = ensure_resample(history, args.resample_rule, base_column=cfg.base_asset)
        history = history[[cfg.base_asset, *cfg.pairs]].dropna()
        if args.max_bars and args.max_bars > 0:
            history = history.tail(args.max_bars)

        last_trade_idx = rebalance_once(
            history,
            cfg,
            engine,
            broker,
            pnl_path,
            trades_path,
            positions_path,
            last_trade_idx,
        )
        iterations_run += 1

        ts = history.index[-1]
        print(f"[live] ts={ts} equity={broker.total_equity():,.2f}", flush=True)


def main() -> None:
    args = parse_args()
    live_loop(args)


if __name__ == "__main__":
    main()
