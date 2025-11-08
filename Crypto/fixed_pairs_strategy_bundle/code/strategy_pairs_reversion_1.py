"""
Market-neutral spread-reversion strategy built on cointegrated crypto pairs.

The strategy trades three altcoin/BTC spreads. We estimate hedge ratios on the
training window, monitor the residual spread with a rolling z-score, and enter
long/short positions when the residual deviates beyond configurable bands.
Positions revert to flat once the spread mean reverts. Portfolio risk is
handled through position sizing, portfolio-level volatility targeting, and
transaction cost modelling (5 bps per leg).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COST_RATE = 0.0005
PAIRS: Tuple[str, ...] = ("ETCUSDT", "APTUSDT", "ARBUSDT")


@dataclass
class StrategyConfig:
    dataset: Path = Path("crypto_data.zip")
    dataset_member: str = "crypto_data.csv"
    base_asset: str = "BTC_USD"
    pairs: Tuple[str, ...] = PAIRS
    resample_rule: str = "15min"
    train_split: str = "2024-06-30"
    z_entry: float = 2.0
    z_exit: float = 0.8
    spread_window: int = 144        # 36 hours on 15-minute bars
    base_weight: float = 0.18
    vol_target: float = 0.25        # annualised target for the aggregate book
    max_gross_leverage: float = 1.2
    cost_rate: float = COST_RATE

    @property
    def periods_per_year(self) -> int:
        return int(pd.Timedelta(days=365) / pd.Timedelta(self.resample_rule))


def load_prices(cfg: StrategyConfig) -> pd.DataFrame:
    with zipfile.ZipFile(cfg.dataset) as zf:
        with zf.open(cfg.dataset_member) as fh:
            df = pd.read_csv(fh, parse_dates=["open_time"])
    df = df.set_index("open_time").sort_index()
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df.resample(cfg.resample_rule).last().ffill().dropna()
    return df[[cfg.base_asset, *cfg.pairs]]


def compute_betas(log_prices: pd.DataFrame, base_asset: str, pair_assets: Iterable[str], split_ts: pd.Timestamp) -> Dict[str, Dict[str, float]]:
    train_mask = log_prices.index <= split_ts
    x = np.column_stack((log_prices[base_asset].loc[train_mask], np.ones(train_mask.sum())))
    xx_inv = np.linalg.inv(x.T @ x)
    betas: Dict[str, Dict[str, float]] = {}
    for asset in pair_assets:
        y = log_prices[asset].loc[train_mask].values
        coeff = xx_inv @ (x.T @ y)
        betas[asset] = {"slope": float(coeff[0]), "intercept": float(coeff[1])}
    return betas


def build_spreads(log_prices: pd.DataFrame, base_asset: str, betas: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    spreads = {}
    base_series = log_prices[base_asset]
    for asset, beta in betas.items():
        spreads[asset] = log_prices[asset] - (beta["slope"] * base_series + beta["intercept"])
    return pd.DataFrame(spreads)


def generate_positions(zscores: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    pos = pd.DataFrame(0.0, index=zscores.index, columns=zscores.columns)
    for asset in zscores.columns:
        current = 0.0
        for i, z in enumerate(zscores[asset].values):
            if math.isnan(z):
                pos.iat[i, pos.columns.get_loc(asset)] = current
                continue
            if current == 0.0:
                if z <= -cfg.z_entry:
                    current = cfg.base_weight
                elif z >= cfg.z_entry:
                    current = -cfg.base_weight
            else:
                if current > 0.0 and z >= -cfg.z_exit:
                    current = 0.0
                elif current < 0.0 and z <= cfg.z_exit:
                    current = 0.0
            pos.iat[i, pos.columns.get_loc(asset)] = current
    return pos.ffill().fillna(0.0)


def volatility_target(ann_vol: float, cfg: StrategyConfig) -> float:
    if ann_vol <= 1e-8 or not math.isfinite(ann_vol):
        return 0.0
    leverage = cfg.vol_target / ann_vol
    leverage = min(leverage, cfg.max_gross_leverage)
    return float(leverage)


def run_backtest(cfg: StrategyConfig) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], pd.DataFrame]:
    prices = load_prices(cfg)
    log_prices = np.log(prices)
    returns = log_prices.diff().fillna(0.0)

    split_ts = pd.Timestamp(cfg.train_split)
    betas = compute_betas(log_prices, cfg.base_asset, cfg.pairs, split_ts)
    spreads = build_spreads(log_prices, cfg.base_asset, betas)
    spread_ma = spreads.rolling(cfg.spread_window).mean()
    spread_std = spreads.rolling(cfg.spread_window).std().replace(0.0, np.nan)
    zscores = (spreads - spread_ma) / spread_std

    positions = generate_positions(zscores, cfg)
    lagged_positions = positions.shift(1).fillna(0.0)

    base_returns = returns[cfg.base_asset]
    pair_returns = returns[list(cfg.pairs)]
    hedge_adjusted = pair_returns.sub(
        base_returns.values.reshape(-1, 1) * np.array([betas[a]["slope"] for a in cfg.pairs]),
        axis=1,
    )

    raw_gross = (lagged_positions * hedge_adjusted).sum(axis=1)
    base_turnover = positions.diff().abs().sum(axis=1)
    hedge_turnover = positions.diff().abs().mul(np.abs([betas[a]["slope"] for a in cfg.pairs])).sum(axis=1)
    turnover = base_turnover + hedge_turnover
    costs = turnover * cfg.cost_rate

    # Volatility targeting on aggregate returns
    periods_per_year = cfg.periods_per_year
    rolling_vol = raw_gross.ewm(span=cfg.spread_window, adjust=False).std()
    ann_vol = rolling_vol * math.sqrt(periods_per_year)
    leverage_series = ann_vol.shift(1).apply(lambda v: volatility_target(v, cfg))
    leverage_series = leverage_series.clip(lower=0.0, upper=cfg.max_gross_leverage).fillna(0.0)

    scaled_gross = raw_gross * leverage_series
    scaled_costs = costs * leverage_series
    net = scaled_gross - scaled_costs

    equity = (1.0 + net).cumprod()
    result = pd.DataFrame(
        {
            "gross_return": scaled_gross,
            "net_return": net,
            "raw_gross": raw_gross,
            "cost": scaled_costs,
            "turnover": turnover * leverage_series,
            "leverage": leverage_series,
            "equity": equity,
        }
    )
    result["basket_return"] = base_returns
    result["active_pairs"] = (positions.abs() > 0).sum(axis=1)
    result["signal_strength"] = positions.abs().sum(axis=1)
    return result, betas, positions


def compute_metrics(series: pd.Series, cfg: StrategyConfig, turnover: pd.Series | None = None) -> Dict[str, float]:
    series = series.dropna()
    if series.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "mar": 0.0,
            "monthly_win_rate": 0.0,
            "avg_turnover": 0.0,
        }

    cumulative = (1.0 + series).cumprod()
    total_return = cumulative.iloc[-1] - 1.0
    periods_per_year = cfg.periods_per_year
    annual_return = (1.0 + total_return) ** (periods_per_year / len(series)) - 1.0
    std = series.std()
    sharpe = series.mean() / std * math.sqrt(periods_per_year) if std > 0 else 0.0
    drawdown = 1.0 - cumulative / cumulative.cummax()
    max_dd = drawdown.max()
    mar = annual_return / max_dd if max_dd != 0 else 0.0
    monthly = series.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else 0.0
    avg_turnover = float(turnover.loc[series.index].mean()) if turnover is not None else 0.0

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "mar": float(mar),
        "monthly_win_rate": win_rate,
        "avg_turnover": avg_turnover,
    }


def segment_metrics(series: pd.Series, cfg: StrategyConfig, turnover: pd.Series | None = None) -> Dict[str, Dict[str, float]]:
    split_ts = pd.Timestamp(cfg.train_split)
    train = series.loc[:split_ts]
    test = series.loc[split_ts + pd.Timedelta(cfg.resample_rule):]
    return {
        "train": compute_metrics(train, cfg, turnover),
        "test": compute_metrics(test, cfg, turnover),
        "overall": compute_metrics(series, cfg, turnover),
    }


def plot_equity(results: pd.DataFrame, cfg: StrategyConfig, output_path: Path) -> None:
    net_equity = results["equity"]
    buy_hold = (1.0 + results["basket_return"]).cumprod()
    split_ts = pd.Timestamp(cfg.train_split)

    gross_activity = results["signal_strength"]
    entries = (gross_activity > 0.0) & (gross_activity.shift(1, fill_value=0.0) == 0.0)
    exits = (gross_activity == 0.0) & (gross_activity.shift(1, fill_value=0.0) > 0.0)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax_top.plot(net_equity, label="Pairs strategy (net)", color="#1b9e77", linewidth=1.5)
    ax_top.plot(buy_hold, label="BTC buy & hold", color="#7570b3", linewidth=1.2)
    ax_top.scatter(net_equity.index[entries], net_equity[entries], color="green", marker="^", s=25, label="Entry")
    ax_top.scatter(net_equity.index[exits], net_equity[exits], color="red", marker="v", s=25, label="Exit")
    ax_top.axvline(split_ts, linestyle="--", color="gray", linewidth=1.0, alpha=0.7)
    ax_top.set_ylabel("Equity (× start capital)")
    ax_top.legend(loc="upper left")
    ax_top.grid(alpha=0.3)

    ax_bottom.plot(results["active_pairs"], label="# active pairs", color="#d95f02", linewidth=1.2)
    ax_bottom.axvline(split_ts, linestyle="--", color="gray", linewidth=1.0, alpha=0.7)
    ax_bottom.set_ylabel("Active spreads")
    ax_bottom.set_xlabel("Timestamp")
    ax_bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_metrics_csv(strategy_metrics: Dict[str, Dict[str, float]], baseline_metrics: Dict[str, Dict[str, float]], output_csv: Path) -> None:
    rows = []
    for label, metrics in (("strategy", strategy_metrics), ("buy_hold", baseline_metrics)):
        for segment, values in metrics.items():
            row = {"model": label, "segment": segment}
            row.update(values)
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Market-neutral pairs reversion strategy backtest.")
    parser.add_argument("--data", type=Path, default=Path("crypto_data.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument("--train-end", type=str, default="2024-06-30")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StrategyConfig(dataset=args.data, train_split=args.train_end)

    results, betas, positions = run_backtest(cfg)
    strategy_metrics = segment_metrics(results["net_return"], cfg, results["turnover"])
    baseline_metrics = segment_metrics(results["basket_return"], cfg)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics_pairs_reversion_1.csv"
    chart_path = output_dir / "pnl_pairs_reversion_1.png"
    summary_path = output_dir / "strategy_summary_pairs_reversion_1.json"

    save_metrics_csv(strategy_metrics, baseline_metrics, metrics_path)
    plot_equity(results, cfg, chart_path)

    summary = {
        "betas": betas,
        "strategy_metrics": strategy_metrics,
        "baseline_metrics": baseline_metrics,
        "config": {
            "pairs": cfg.pairs,
            "z_entry": cfg.z_entry,
            "z_exit": cfg.z_exit,
            "spread_window": cfg.spread_window,
            "base_weight": cfg.base_weight,
            "vol_target": cfg.vol_target,
            "max_gross_leverage": cfg.max_gross_leverage,
            "cost_rate": cfg.cost_rate,
            "resample_rule": cfg.resample_rule,
            "train_split": cfg.train_split,
        },
        "trade_summary": {
            "avg_active_pairs": float(results["active_pairs"].mean()),
            "max_active_pairs": int(results["active_pairs"].max()),
            "avg_turnover": float(results["turnover"].mean()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
