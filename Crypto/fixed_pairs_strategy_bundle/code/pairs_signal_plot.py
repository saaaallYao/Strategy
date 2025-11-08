"""
Generate diagnostic plots for the pairs reversion strategy showing how z-score
signals map to trade entries/exits over different lookback windows.

The script produces stacked price/z-score panels for each requested window
highlighting thresholds and buy/sell markers so it is easy to see what drives
engagement.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategy_pairs_reversion_1 import (
    StrategyConfig,
    build_spreads,
    compute_betas,
    generate_positions,
    load_prices,
)


def compute_pair_state(cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = load_prices(cfg)
    log_prices = np.log(prices)

    split_ts = pd.Timestamp(cfg.train_split)
    betas = compute_betas(log_prices, cfg.base_asset, cfg.pairs, split_ts)
    spreads = build_spreads(log_prices, cfg.base_asset, betas)
    spread_ma = spreads.rolling(cfg.spread_window).mean()
    spread_std = spreads.rolling(cfg.spread_window).std().replace(0.0, np.nan)
    zscores = (spreads - spread_ma) / spread_std

    positions = generate_positions(zscores, cfg)
    return prices, zscores, positions


def subset_window(df: pd.DataFrame, window: pd.Timedelta, end_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    if window <= pd.Timedelta(0):
        return df
    if end_ts is None:
        end_ts = df.index.max()
    start_ts = end_ts - window
    return df.loc[start_ts:end_ts]


def annotate_trades(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    prev = series.shift(1).fillna(0.0)
    long_entries = (series > 0.0) & (prev == 0.0)
    short_entries = (series < 0.0) & (prev == 0.0)
    exits = (series == 0.0) & (prev != 0.0)
    return long_entries, short_entries, exits


def plot_window(window_df: pd.DataFrame, cfg: StrategyConfig, pair: str, label: str, output_path: Path) -> None:
    if window_df.empty:
        print(f"[warn] No data available for window {label}. Skipping plot.")
        return

    long_entries, short_entries, exits = annotate_trades(window_df["position"])
    ts = window_df.index

    fig, (ax_price, ax_z) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_price.plot(ts, window_df["price"], color="#1b9e77", linewidth=1.3, label=f"{pair} price")
    ax_price.scatter(ts[long_entries], window_df["price"][long_entries], color="#0c7c59", marker="^", s=35, label="Buy (long spread)")
    ax_price.scatter(ts[short_entries], window_df["price"][short_entries], color="#d95f02", marker="v", s=35, label="Sell (short spread)")
    ax_price.scatter(ts[exits], window_df["price"][exits], color="#e41a1c", marker="x", s=40, label="Exit")
    ax_price.set_ylabel("Price")
    ax_price.set_title(f"{pair} price & signals ({label})")
    ax_price.grid(alpha=0.3)
    ax_price.legend(loc="upper left")

    ax_z.plot(ts, window_df["zscore"], color="#333333", linewidth=1.3, label="Z-score")
    ax_z.axhline(cfg.z_entry, color="#d95f02", linestyle="--", linewidth=1.0, label="+/- entry")
    ax_z.axhline(-cfg.z_entry, color="#d95f02", linestyle="--", linewidth=1.0)
    ax_z.axhline(cfg.z_exit, color="#7570b3", linestyle=":", linewidth=1.0, label="+/- exit")
    ax_z.axhline(-cfg.z_exit, color="#7570b3", linestyle=":", linewidth=1.0)
    ax_z.scatter(ts[long_entries], window_df["zscore"][long_entries], color="#0c7c59", marker="^", s=30)
    ax_z.scatter(ts[short_entries], window_df["zscore"][short_entries], color="#d95f02", marker="v", s=30)
    ax_z.scatter(ts[exits], window_df["zscore"][exits], color="#e41a1c", marker="x", s=35)
    ax_z.set_ylabel("Z-score")
    ax_z.set_xlabel("Timestamp")
    ax_z.legend(loc="upper left")
    ax_z.grid(alpha=0.3)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_z.xaxis.set_major_locator(locator)
    ax_z.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[info] Saved {label} plot to {output_path}")


def parse_windows(values: Sequence[str]) -> list[tuple[str, pd.Timedelta]]:
    parsed: list[tuple[str, pd.Timedelta]] = []
    for value in values:
        td = pd.Timedelta(value)
        parsed.append((value, td))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pairs reversion z-score/price signals for diagnostics.")
    parser.add_argument("--data", type=Path, default=Path("crypto_data.zip"))
    parser.add_argument("--train-end", type=str, default="2024-06-30")
    parser.add_argument("--pair", type=str, default="ETCUSDT", help="Pair to visualise (must be in StrategyConfig.pairs).")
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["48h", "7d"],
        help="List of windows (e.g. 48h 7d 30d) to render, parsed via pandas Timedelta.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs"), help="Directory for generated plots.")
    parser.add_argument(
        "--anchors",
        nargs="+",
        default=None,
        help="Optional ISO timestamps to anchor window end points (e.g. 2024-07-01T00:00).",
    )
    parser.add_argument(
        "--random-anchors",
        type=int,
        default=0,
        help="Sample N random timestamps from available data to use as window end points.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for random anchor sampling.")
    args = parser.parse_args()

    cfg = StrategyConfig(dataset=args.data, train_split=args.train_end)
    if args.pair not in cfg.pairs:
        raise ValueError(f"Pair {args.pair} not in configured pairs {cfg.pairs}")

    prices, zscores, positions = compute_pair_state(cfg)
    pair_df = pd.DataFrame(
        {
            "price": prices[args.pair],
            "zscore": zscores[args.pair],
            "position": positions[args.pair],
        }
    ).dropna(subset=["zscore"])

    windows = parse_windows(args.windows)

    anchors: list[pd.Timestamp | None] = []
    if args.anchors:
        anchors.extend(pd.to_datetime(args.anchors))
    if args.random_anchors > 0:
        rng = np.random.default_rng(args.seed)
        idx = pair_df.index.unique()
        if len(idx) == 0:
            raise RuntimeError("No index values available for random sampling.")
        choice = rng.choice(idx.values, size=min(args.random_anchors, len(idx)), replace=False)
        anchors.extend(pd.to_datetime(choice))
    if not anchors:
        anchors = [None]

    for anchor in anchors:
        anchor_label = "latest" if anchor is None else anchor.strftime("%Y%m%dT%H%M")
        for label, window in windows:
            window_df = subset_window(pair_df, window, anchor)
            sanitized = label.replace(" ", "_")
            output_path = args.output_dir / f"pairs_signal_{args.pair}_{sanitized}_{anchor_label}.png"
            plot_window(window_df, cfg, args.pair, f"{label} @ {anchor_label}", output_path)


if __name__ == "__main__":
    main()
