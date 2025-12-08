#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from kucoin_client import fetch_history, symbol_to_column, to_kucoin_symbol
from pairs_strategy_core import StrategyConfig, compute_metrics
from pairs_strategy_signal import PairsSignalEngine

DEFAULT_PAIRS = ["ETC-USDT", "APT-USDT", "ARB-USDT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest Nebula Pairs using KuCoin spot klines.")
    parser.add_argument("--base-symbol", default=os.environ.get("FP_BASE_SYMBOL", "BTC-USDT"), help="Base asset symbol (default: BTC-USDT)")
    parser.add_argument("--pairs", nargs="+", default=None, help="Pairs to trade (KuCoin symbols). Defaults to strategy core PAIRS.")
    parser.add_argument("--resample-rule", default=os.environ.get("FP_RESAMPLE_RULE", "15min"), help="Resample rule for bars (default: 15min)")
    parser.add_argument("--lookback-days", type=int, default=int(os.environ.get("FP_LOOKBACK_DAYS", "400")), help="History window to download (days, default: 400)")
    parser.add_argument("--train-split", default=os.environ.get("FP_TRAIN_SPLIT"), help="Optional ISO timestamp to split training data (default: use full history)")
    parser.add_argument("--output-prefix", default=os.environ.get("FP_LOG_PREFIX", "nebula_pairs"), help="Prefix for output files (default: nebula_pairs)")
    return parser.parse_args()


def save_frame(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def main() -> None:
    args = parse_args()

    base_symbol = to_kucoin_symbol(args.base_symbol)
    pair_symbols = [to_kucoin_symbol(p) for p in (args.pairs if args.pairs else DEFAULT_PAIRS)]
    symbols = [base_symbol, *pair_symbols]

    history = fetch_history(symbols, resample_rule=args.resample_rule, lookback_days=args.lookback_days)
    if history.empty:
        raise RuntimeError("No history returned from KuCoin; check symbols or lookback window.")

    base_col = symbol_to_column(base_symbol)
    pair_cols = tuple(symbol_to_column(p) for p in pair_symbols)
    panel = history[[base_col, *pair_cols]].dropna()

    cfg = StrategyConfig(
        base_asset=base_col,
        pairs=pair_cols,
        resample_rule=args.resample_rule,
        train_split=args.train_split,
    )
    engine = PairsSignalEngine(cfg)
    state = engine.compute_state(panel)

    out_dir = PROJECT_ROOT / "data"
    prefix = args.output_prefix
    save_frame(state.results, out_dir / f"backtest_results_{prefix}.csv")
    save_frame(state.weights, out_dir / f"backtest_weights_{prefix}.csv")
    save_frame(state.zscores, out_dir / f"backtest_zscores_{prefix}.csv")
    save_frame(state.raw_positions, out_dir / f"backtest_raw_positions_{prefix}.csv")
    save_frame(state.scaled_positions, out_dir / f"backtest_scaled_positions_{prefix}.csv")
    save_frame(state.results[["equity"]], out_dir / f"backtest_equity_{prefix}.csv")

    metrics = compute_metrics(state.results["net_return"], cfg, turnover=state.results["turnover"])
    metrics_path = out_dir / f"backtest_metrics_{prefix}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[backtest] saved outputs to {out_dir} with prefix '{prefix}'")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
