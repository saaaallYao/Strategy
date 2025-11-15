#!/usr/bin/env python3
"""
Backtesting entrypoint that reuses the shared signal engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is importable when executed directly and avoid name clashes.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if sys.path and sys.path[0] == str(SCRIPT_DIR):
    sys.path[0] = str(REPO_ROOT)
elif str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pairs_strategy.core import StrategyConfig, compute_metrics, load_prices
from pairs_strategy.signal import PairsSignalEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the fixed pairs strategy using the shared signal engine.")
    parser.add_argument("--dataset", type=Path, default=Path("fixed_pairs_strategy_bundle/crypto_data.zip"), help="Zip file containing historical prices.")
    parser.add_argument(
        "--dataset-member",
        default="crypto_data.csv",
        help="CSV filename inside the dataset zip.",
    )
    parser.add_argument("--base-asset", default="BTC_USD", help="Column name for the hedge asset.")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["ETCUSDT", "APTUSDT", "ARBUSDT"],
        help="Residual spread columns to trade.",
    )
    parser.add_argument("--resample-rule", default="15min", help="Pandas resample rule to aggregate the data.")
    parser.add_argument("--train-split", default="2024-06-30", help="Timestamp that separates train and test regimes.")
    parser.add_argument("--z-entry", type=float, default=2.0, help="Z-score entry threshold.")
    parser.add_argument("--z-exit", type=float, default=0.8, help="Z-score exit threshold.")
    parser.add_argument(
        "--spread-window",
        type=int,
        default=144,
        help="Rolling window (bars) for spread mean/std and vol target.",
    )
    parser.add_argument("--base-weight", type=float, default=0.18, help="Per-spread nominal weight when a signal fires.")
    parser.add_argument("--vol-target", type=float, default=0.25, help="Annualized volatility target for the aggregate book.")
    parser.add_argument("--max-gross-leverage", type=float, default=1.2, help="Upper cap on leverage after targeting.")
    parser.add_argument("--cost-rate", type=float, default=0.0005, help="Per-leg transaction cost assumption.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pairs_strategy/artifacts_backtest"),
        help="Directory to store CSV/JSON outputs.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        dataset=args.dataset,
        dataset_member=args.dataset_member,
        base_asset=args.base_asset,
        pairs=tuple(args.pairs),
        resample_rule=args.resample_rule,
        train_split=args.train_split,
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        spread_window=args.spread_window,
        base_weight=args.base_weight,
        vol_target=args.vol_target,
        max_gross_leverage=args.max_gross_leverage,
        cost_rate=args.cost_rate,
    )


def save_outputs(output_dir: Path, state) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state.results.to_csv(output_dir / "backtest_results.csv", index=True)
    state.raw_positions.to_csv(output_dir / "raw_positions.csv", index=True)
    state.scaled_positions.to_csv(output_dir / "scaled_positions.csv", index=True)
    state.weights.to_csv(output_dir / "weights.csv", index=True)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    prices = load_prices(cfg)

    engine = PairsSignalEngine(cfg)
    state = engine.compute_state(prices)
    metrics = compute_metrics(state.results["net_return"], cfg, turnover=state.results["turnover"])

    save_outputs(args.output_dir, state)
    summary_path = args.output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
