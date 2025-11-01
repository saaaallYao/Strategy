"""
Intraday relative-strength pairs strategy.

Idea:
- Resample the minute data to 30-minute bars.
- Rank alt-coins by their relative momentum versus BTC using ratio returns.
- Go long the strongest alt-coins and hedge each allocation with an equal notional
  short in BTC to stay close to market-neutral.
- Size positions by inverse volatility of the alt/BTC spread and target a modest
  annualised volatility, rebalancing each bar (~48 decisions per day).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-darkgrid")


CONFIG_PATH = Path("config.json")
OUTPUT_DIR = Path("docs")
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StrategyParams:
    freq: str = "30min"
    rel_momentum_lookback: int = 8   # 4 hours
    spread_vol_lookback: int = 48    # 1 day
    entry_threshold: float = 0.0015
    decay_threshold: float = 0.0004
    top_n: int = 3
    target_vol: float = 0.22
    max_leverage: float = 1.6
    transaction_cost: float = 0.0005
    train_end: str = "2024-06-30"
    cooldown_bars: int = 4

    @property
    def periods_per_year(self) -> int:
        minutes = pd.Timedelta(self.freq).total_seconds() / 60.0
        per_day = int(24 * 60 / minutes)
        return 365 * per_day


def load_config() -> Dict[str, str]:
    with CONFIG_PATH.open() as fh:
        return json.load(fh)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if dataset_path.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(dataset_path) as zf:
            csv_name = next(name for name in zf.namelist() if name.endswith(".csv"))
            with zf.open(csv_name) as fh:
                df = pd.read_csv(fh, parse_dates=["open_time"])
    else:
        df = pd.read_csv(dataset_path, parse_dates=["open_time"])
    return df.set_index("open_time").sort_index()


def resample_prices(data: pd.DataFrame, assets: List[str], freq: str) -> pd.DataFrame:
    resampled = (
        data[assets]
        .ffill()
        .resample(freq)
        .last()
        .dropna(how="all")
    )
    return resampled


def compute_relative_scores(prices: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    btc = prices["BTC_USD"]
    ratio = prices.div(btc, axis=0)
    rel_returns = ratio.pct_change(params.rel_momentum_lookback)
    return rel_returns


def compute_strategy(prices: pd.DataFrame, params: StrategyParams) -> Dict[str, pd.Series | pd.DataFrame]:
    returns = prices.pct_change().fillna(0)
    rel_scores = compute_relative_scores(prices, params).shift(1)

    btc = "BTC_USD"
    alt_assets = [col for col in prices.columns if col != btc]

    spread_returns = returns[alt_assets].sub(returns[btc], axis=0)
    spread_vol = spread_returns.rolling(params.spread_vol_lookback).std().shift(1)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    cooldown = pd.DataFrame(0, index=prices.index, columns=alt_assets)

    for ts in prices.index:
        score_row = rel_scores.loc[ts, alt_assets].dropna()
        if score_row.empty:
            continue

        prev_weights = weights.shift(1).loc[ts] if ts != prices.index[0] else pd.Series(0.0, index=prices.columns)
        prev_cooldown = cooldown.shift(1).loc[ts] if ts != prices.index[0] else pd.Series(0, index=alt_assets)
        current_cooldown = prev_cooldown.clip(lower=0)

        # update cooldown for existing positions
        for asset in alt_assets:
            prev_score = score_row.get(asset, 0.0)
            if prev_weights.get(asset, 0.0) != 0.0:
                if prev_score >= -params.decay_threshold:
                    current_cooldown[asset] = min(params.cooldown_bars, prev_cooldown.get(asset, 0) + 1)
                else:
                    current_cooldown[asset] = 0
            else:
                current_cooldown[asset] = max(prev_cooldown.get(asset, 0) - 1, 0)

        mask = current_cooldown < params.cooldown_bars
        eligible_idx = mask[mask].index.intersection(score_row.index)
        eligible = score_row.loc[eligible_idx]
        selected = eligible[eligible <= -params.entry_threshold].nsmallest(params.top_n)
        if selected.empty:
            cooldown.loc[ts] = current_cooldown
            continue

        vol_slice = spread_vol.loc[ts, selected.index].replace(0, np.nan).dropna()
        if vol_slice.empty:
            cooldown.loc[ts] = current_cooldown
            continue

        inv_vol = 1.0 / vol_slice
        base_alloc = inv_vol / inv_vol.sum()

        btc_weight = 0.0
        for asset, weight in base_alloc.items():
            weights.loc[ts, asset] = weight
            btc_weight -= weight

        weights.loc[ts, btc] = btc_weight
        cooldown.loc[ts] = current_cooldown

    raw_turnover = weights.diff().abs().sum(axis=1).fillna(0)
    gross_returns = (weights * returns).sum(axis=1)
    net_returns = gross_returns - raw_turnover * params.transaction_cost

    ann_factor = params.periods_per_year
    inst_vol = returns.rolling(params.spread_vol_lookback).std().shift(1) * np.sqrt(ann_factor)
    proxy_vol = inst_vol[btc].replace(0, np.nan).ffill().bfill()
    inst_vol = inst_vol.apply(lambda col: col.fillna(proxy_vol))
    portfolio_vol = np.sqrt(((weights ** 2) * (inst_vol ** 2)).sum(axis=1))
    leverage = (params.target_vol / portfolio_vol.replace(0, np.nan)).clip(upper=params.max_leverage).fillna(0)

    effective_weights = weights.mul(leverage, axis=0)
    turnover = effective_weights.diff().abs().sum(axis=1).fillna(0)
    strategy_returns = (effective_weights * returns).sum(axis=1) - turnover * params.transaction_cost

    trade_flags = (turnover > 1e-6).astype(int)

    return {
        "returns": strategy_returns,
        "raw_returns": net_returns,
        "weights": effective_weights,
        "base_weights": weights,
        "leverage": leverage,
        "turnover": turnover,
        "trade_flags": trade_flags,
    }


def to_daily(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return returns
    return (1.0 + returns).resample("1D").prod() - 1.0


def performance_metrics(returns: pd.Series, ann_factor: float) -> Dict[str, float]:
    ser = returns.dropna()
    if ser.empty or ser.std() == 0:
        return {k: np.nan for k in ["ann_return", "sharpe", "max_dd", "mar", "ann_vol", "cum_return"]}

    equity = (1.0 + ser).cumprod()
    total_return = equity.iloc[-1] - 1.0
    ann_return = (1.0 + total_return) ** (ann_factor / len(ser)) - 1.0
    ann_vol = ser.std() * np.sqrt(ann_factor)
    sharpe = (ser.mean() / ser.std()) * np.sqrt(ann_factor)
    drawdown = equity / equity.cummax() - 1.0
    max_dd = drawdown.min()
    mar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "mar": mar,
        "ann_vol": ann_vol,
        "cum_return": total_return,
    }


def monthly_stats(returns: pd.Series) -> Dict[str, float]:
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    monthly = monthly.dropna()
    if monthly.empty:
        return {"win_rate": np.nan, "max_dd_monthly": np.nan}
    equity = (1.0 + monthly).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {
        "win_rate": (monthly > 0).mean(),
        "max_dd_monthly": dd.min(),
    }


def attach_additional_metrics(perf: Dict[str, float], monthly: Dict[str, float]) -> Dict[str, float]:
    merged = perf.copy()
    merged.update(
        {
            "max_dd_monthly": monthly.get("max_dd_monthly"),
            "win_rate_monthly": monthly.get("win_rate"),
        }
    )
    return merged


def partition_series(series: pd.Series, split_ts: pd.Timestamp) -> Dict[str, pd.Series]:
    after_split = split_ts + pd.Timedelta(seconds=1)
    return {
        "train": series.loc[:split_ts],
        "test": series.loc[after_split:],
        "full": series,
    }


def summarise_metrics(returns: pd.Series, split_ts: pd.Timestamp) -> pd.DataFrame:
    parts = partition_series(returns, split_ts)
    records = []
    for period, segment in parts.items():
        daily = to_daily(segment)
        perf = performance_metrics(daily, 365)
        month = monthly_stats(daily)
        record = attach_additional_metrics(perf, month)
        record["period"] = period
        record["length_days"] = len(daily)
        record["bars"] = len(segment)
        records.append(record)
    return pd.DataFrame(records).set_index("period")


def compute_activity_metrics(
    turnover: pd.Series,
    trade_flags: pd.Series,
    split_ts: pd.Timestamp,
) -> Dict[str, Dict[str, float]]:
    after_split = split_ts + pd.Timedelta(seconds=1)
    segments = {
        "train": (turnover.loc[:split_ts], trade_flags.loc[:split_ts]),
        "test": (turnover.loc[after_split:], trade_flags.loc[after_split:]),
        "full": (turnover, trade_flags),
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, (turn_seg, flag_seg) in segments.items():
        trades_per_day = flag_seg.resample("1D").sum()
        out[name] = {
            "avg_turnover": turn_seg.mean(),
            "median_turnover": turn_seg.median(),
            "avg_trades_per_day": trades_per_day.mean(),
            "median_trades_per_day": trades_per_day.median(),
        }
    return out


def plot_equity_curves(strategy_returns: pd.Series, benchmark_returns: pd.Series, split_ts: pd.Timestamp, output_path: Path) -> None:
    strat_eq = (1.0 + to_daily(strategy_returns)).cumprod()
    bench_eq = (1.0 + to_daily(benchmark_returns)).cumprod()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(strat_eq.index, strat_eq, label="Strategy")
    ax.plot(bench_eq.index, bench_eq, label="BTC Buy & Hold", alpha=0.7)
    ax.axvline(split_ts, color="grey", linestyle="--", linewidth=1, label="Train/Test Split")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Equity Curves (Daily Aggregation)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_trades(strategy_returns: pd.Series, weights: pd.DataFrame, output_path: Path) -> None:
    equity = (1.0 + to_daily(strategy_returns)).cumprod()
    exposure = weights.abs().sum(axis=1)
    invested = (exposure > 1e-6).astype(bool)
    entries = invested & (~invested.shift(1, fill_value=False))
    exits = (~invested) & invested.shift(1, fill_value=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(equity.index, equity, color="#1f77b4", label="Strategy Equity")
    ax.scatter(equity.index[entries.resample("1D").max().astype(bool)], equity[entries.resample("1D").max().astype(bool)], color="green", marker="^", label="Entry", s=18)
    ax.scatter(equity.index[exits.resample("1D").max().astype(bool)], equity[exits.resample("1D").max().astype(bool)], color="red", marker="v", label="Exit", s=18)
    ax.legend()
    ax.set_ylabel("Equity")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_monthly_returns(strategy_returns: pd.Series, output_path: Path) -> None:
    monthly = (1.0 + to_daily(strategy_returns)).resample("ME").prod() - 1.0
    if monthly.empty:
        return
    colors = ["#2ca02c" if val > 0 else "#d62728" for val in monthly]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(monthly.index, monthly.values, color=colors)
    ax.set_ylabel("Monthly Return")
    ax.set_title("Monthly Returns")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    config = load_config()
    dataset_path = Path(config["dataset_path"])
    df = load_dataset(dataset_path)

    assets = [
        "BTC_USD",
        "ETH_USD",
        "BNBUSDT",
        "SOL_USD",
        "XRPUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "LINKUSDT",
        "AVAXUSDT",
        "MATICUSDT",
        "DOTUSDT",
        "ATOMUSDT",
        "APTUSDT",
        "ARBUSDT",
        "OPUSDT",
        "SUIUSDT",
        "NEARUSDT",
        "ETCUSDT",
    ]

    params = StrategyParams()
    prices = resample_prices(df, assets, params.freq)
    prices = prices.dropna(axis=1, how="all")

    results = compute_strategy(prices, params)
    strategy_returns = results["returns"]
    leverage = results["leverage"]
    effective_weights = results["weights"]
    turnover = results["turnover"]
    trade_flags = results["trade_flags"]

    btc_returns = prices["BTC_USD"].pct_change().fillna(0)

    split_ts = pd.Timestamp(params.train_end, tz=prices.index.tz)

    metrics = summarise_metrics(strategy_returns, split_ts)
    btc_metrics = summarise_metrics(btc_returns, split_ts)

    activity = compute_activity_metrics(turnover, trade_flags, split_ts)
    for period, stats in activity.items():
        for key, value in stats.items():
            metrics.loc[period, key] = value

    metrics.loc["train", "avg_leverage"] = leverage.loc[:split_ts].mean()
    metrics.loc["test", "avg_leverage"] = leverage.loc[split_ts + pd.Timedelta(seconds=1):].mean()
    metrics.loc["full", "avg_leverage"] = leverage.mean()

    metrics_path = OUTPUT_DIR / "03_strategy_metrics.csv"
    metrics.to_csv(metrics_path)

    btc_metrics_path = OUTPUT_DIR / "03_benchmark_metrics.csv"
    btc_metrics.to_csv(btc_metrics_path)

    plot_equity_curves(strategy_returns, btc_returns, split_ts, FIG_DIR / "03_equity.png")
    plot_trades(strategy_returns, effective_weights, FIG_DIR / "03_trades.png")
    plot_monthly_returns(strategy_returns, FIG_DIR / "03_monthly_returns.png")

    summary = {
        "strategy_metrics_csv": str(metrics_path),
        "benchmark_metrics_csv": str(btc_metrics_path),
        "figures": [
            str(FIG_DIR / "03_equity.png"),
            str(FIG_DIR / "03_trades.png"),
            str(FIG_DIR / "03_monthly_returns.png"),
        ],
        "params": {
            "freq": params.freq,
            "rel_momentum_lookback": params.rel_momentum_lookback,
            "spread_vol_lookback": params.spread_vol_lookback,
            "entry_threshold": params.entry_threshold,
            "decay_threshold": params.decay_threshold,
            "top_n": params.top_n,
            "target_vol": params.target_vol,
            "max_leverage": params.max_leverage,
            "cooldown_bars": params.cooldown_bars,
        },
    }
    summary_path = OUTPUT_DIR / "03_outputs.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("Strategy metrics saved to:", metrics_path)
    print("Benchmark metrics saved to:", btc_metrics_path)
    print("Figures saved under:", FIG_DIR)


if __name__ == "__main__":
    main()
