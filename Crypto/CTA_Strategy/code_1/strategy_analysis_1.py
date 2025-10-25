#!/usr/bin/env python3
"""
CTA-style altcoin momentum sleeve built from training data heuristics.
Selects a momentum asset and a convex (high-skew) asset, combines them via
volatility-targeted risk parity, and benchmarks against buy-and-hold.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.json"
DATA_DIR = ROOT_DIR / "data_2"
OUTPUT_DIR = ROOT_DIR / "docs_3"
PLOTS_DIR = OUTPUT_DIR

RESAMPLE_RULE = "60min"
PERIODS_PER_YEAR = 24 * 365  # hourly bars
COST_RATE = 0.0005
VOL_LOOKBACK = 96
VOL_TARGET = 0.0
SMOOTH_ALPHA = 0.1
MOMENTUM_SELECT_DAYS = 60


@dataclass
class StrategySelection:
    momentum_asset: str
    momentum_score: float
    skew_asset: str
    skew_score: float
    core_asset: str
    core_sharpe: float


@dataclass
class StrategyOutputs:
    selection: StrategySelection
    positions: pd.DataFrame
    net_returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    buy_and_hold_returns: pd.Series
    buy_and_hold_equity: pd.Series
    net_position: pd.Series


def resolve_dataset_path() -> Path:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)
    preferred = Path(config.get("dataset_path", "")).expanduser()
    if preferred.exists():
        return preferred
    fallback_csv = DATA_DIR / "crypto_data.csv"
    if fallback_csv.exists():
        return fallback_csv
    fallback_zip = ROOT_DIR / "crypto_data.zip"
    if fallback_zip.exists():
        raise FileNotFoundError(
            "CSV not extracted. Please unzip crypto_data.zip into data_2 directory."
        )
    raise FileNotFoundError("Dataset path not found. Check config.json.")


def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="open_time", parse_dates=["open_time"])
    df = df.sort_index()
    df = df.replace("", np.nan).astype(float)
    df = df.ffill()
    if RESAMPLE_RULE:
        df = df.resample(RESAMPLE_RULE).last().ffill()
    return df


def select_assets(prices: pd.DataFrame) -> StrategySelection:
    returns = prices.pct_change().fillna(0.0)
    split_point = prices.index.min() + pd.DateOffset(months=12)

    train_returns = returns.loc[:split_point]
    skew_scores = train_returns.skew().sort_values(ascending=False)
    skew_asset = skew_scores.index[0]

    momentum_window = 24 * MOMENTUM_SELECT_DAYS
    momentum_series = prices.pct_change(momentum_window)
    momentum_at_split = momentum_series.loc[:split_point].iloc[-1]
    momentum_at_split = momentum_at_split.dropna().sort_values(ascending=False)
    momentum_asset = momentum_at_split.index[0]
    if momentum_asset == skew_asset and len(momentum_at_split) > 1:
        momentum_asset = momentum_at_split.index[1]

    core_asset = "BTC_USD"
    core_returns = train_returns[core_asset]
    core_mean = core_returns.mean()
    core_std = core_returns.std()
    core_sharpe = (
        float(core_mean / core_std * math.sqrt(PERIODS_PER_YEAR))
        if core_std > 0
        else float("nan")
    )

    selection = StrategySelection(
        momentum_asset=momentum_asset,
        momentum_score=float(momentum_at_split.loc[momentum_asset]),
        skew_asset=skew_asset,
        skew_score=float(skew_scores.loc[skew_asset]),
        core_asset=core_asset,
        core_sharpe=core_sharpe,
    )
    return selection


def construct_positions(
    prices: pd.DataFrame, selection: StrategySelection
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assets: List[str] = [
        selection.momentum_asset,
        selection.skew_asset,
        selection.core_asset,
    ]
    assets = list(dict.fromkeys(assets))

    returns = prices.pct_change().fillna(0.0)
    returns_subset = returns[assets]
    vol = returns_subset.ewm(span=VOL_LOOKBACK, adjust=False).std()
    vol_annual = vol * math.sqrt(PERIODS_PER_YEAR)

    inv_vol = 1.0 / (vol_annual + 1e-6)
    base_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
    smoothed_weights = base_weights.ewm(alpha=SMOOTH_ALPHA, adjust=False).mean()

    short_momentum = prices[assets].pct_change(24)
    portfolio_momentum = (short_momentum * smoothed_weights).sum(axis=1)
    risk_scale = np.where(portfolio_momentum < 0, 0.85, 1.0)
    risk_scale = pd.Series(risk_scale, index=smoothed_weights.index)
    risk_scale = risk_scale.ewm(alpha=0.2, adjust=False).mean()
    smoothed_weights = smoothed_weights.mul(risk_scale, axis=0)

    if VOL_TARGET > 0:
        vol_period_target = VOL_TARGET / math.sqrt(PERIODS_PER_YEAR)
        port_vol = np.sqrt((smoothed_weights.pow(2) * vol.pow(2)).sum(axis=1))
        scaler = (vol_period_target / port_vol.replace(0.0, np.nan)).clip(upper=4.0)
        scaler = scaler.fillna(0.0)
        positions = smoothed_weights.mul(scaler, axis=0)
    else:
        positions = smoothed_weights
    return positions, returns_subset


def compute_pnl(
    positions: pd.DataFrame, returns: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    shifted_positions = positions.shift(1).fillna(0.0)
    gross_returns = (shifted_positions * returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    costs = COST_RATE * turnover
    net_returns = gross_returns - costs
    return net_returns, gross_returns, turnover, costs


def run_backtest(prices: pd.DataFrame, selection: StrategySelection) -> StrategyOutputs:
    positions, returns_subset = construct_positions(prices, selection)
    net_returns, gross_returns, turnover, costs = compute_pnl(positions, returns_subset)

    equity_curve = (1.0 + net_returns).cumprod()
    gross_equity_curve = (1.0 + gross_returns).cumprod()

    buy_and_hold_weights = pd.Series(
        1.0 / prices.shape[1], index=prices.columns, name="w"
    )
    buy_and_hold_returns = (prices.pct_change().fillna(0.0) * buy_and_hold_weights).sum(
        axis=1
    )
    buy_and_hold_equity = (1.0 + buy_and_hold_returns).cumprod()

    net_position = positions.sum(axis=1)

    return StrategyOutputs(
        selection=selection,
        positions=positions,
        net_returns=net_returns,
        gross_returns=gross_returns,
        costs=costs,
        turnover=turnover,
        equity_curve=equity_curve,
        gross_equity_curve=gross_equity_curve,
        buy_and_hold_returns=buy_and_hold_returns,
        buy_and_hold_equity=buy_and_hold_equity,
        net_position=net_position,
    )


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def evaluate_slice(
    returns: pd.Series,
    gross_returns: pd.Series,
    equity: pd.Series,
    monthly_equity: pd.Series,
) -> Dict[str, float]:
    if returns.empty:
        return {
            "annual_return": float("nan"),
            "annual_gross_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "max_monthly_drawdown": float("nan"),
            "mar_ratio": float("nan"),
            "monthly_win_rate": float("nan"),
        }

    mean = returns.mean()
    std = returns.std()
    sharpe = (mean / std * math.sqrt(PERIODS_PER_YEAR)) if std > 0 else float("nan")
    annual_return = (1.0 + returns).prod() ** (PERIODS_PER_YEAR / len(returns)) - 1.0
    annual_gross_return = (1.0 + gross_returns).prod() ** (
        PERIODS_PER_YEAR / len(gross_returns)
    ) - 1.0
    max_dd = _max_drawdown(equity)

    monthly_returns = monthly_equity.pct_change().dropna()
    monthly_cum = (1.0 + monthly_returns).cumprod()
    monthly_max_dd = (monthly_cum / monthly_cum.cummax() - 1.0).min()
    monthly_win_rate = (monthly_returns > 0).mean()

    mar_ratio = annual_return / abs(max_dd) if max_dd != 0 else float("inf")
    return {
        "annual_return": annual_return,
        "annual_gross_return": annual_gross_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_monthly_drawdown": monthly_max_dd,
        "mar_ratio": mar_ratio,
        "monthly_win_rate": monthly_win_rate,
    }


def collect_metrics(outputs: StrategyOutputs) -> Dict[str, Dict[str, float]]:
    equity = outputs.equity_curve
    gross_equity = outputs.gross_equity_curve
    buy_hold_equity = outputs.buy_and_hold_equity

    monthly_equity = equity.resample("ME").last().dropna()
    monthly_equity_bh = buy_hold_equity.resample("ME").last().dropna()

    split_point = equity.index.min() + pd.DateOffset(months=12)
    train_slice = slice(None, split_point)
    test_slice = slice(split_point + pd.Timedelta("1s"), None)
    train_month_slice = slice(None, split_point)
    test_month_slice = slice(split_point + pd.Timedelta("1D"), None)

    metrics = {
        "train": evaluate_slice(
            outputs.net_returns.loc[train_slice],
            outputs.gross_returns.loc[train_slice],
            equity.loc[train_slice],
            monthly_equity.loc[train_month_slice],
        ),
        "test": evaluate_slice(
            outputs.net_returns.loc[test_slice],
            outputs.gross_returns.loc[test_slice],
            equity.loc[test_slice],
            monthly_equity.loc[test_month_slice],
        ),
        "full": evaluate_slice(
            outputs.net_returns,
            outputs.gross_returns,
            equity,
            monthly_equity,
        ),
        "buy_and_hold_full": evaluate_slice(
            outputs.buy_and_hold_returns,
            outputs.buy_and_hold_returns,
            buy_hold_equity,
            monthly_equity_bh,
        ),
    }

    avg_daily_turnover = (
        outputs.turnover.resample("D").sum().mean()
        if not outputs.turnover.empty
        else float("nan")
    )
    metrics["full"]["average_daily_turnover"] = avg_daily_turnover
    return metrics


def format_pct(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def save_plot(outputs: StrategyOutputs, plot_path: Path) -> None:
    equity = outputs.equity_curve
    buy_hold = outputs.buy_and_hold_equity
    net_pos = outputs.net_position

    net_sign = np.sign(net_pos)
    signal_change = net_sign.diff().fillna(0.0)
    buys = equity.loc[signal_change > 0]
    sells = equity.loc[signal_change < 0]

    plt.figure(figsize=(14, 7))
    plt.plot(equity.index, equity.values, label="CTA Strategy (Net)", linewidth=2.0)
    plt.plot(
        buy_hold.index, buy_hold.values, label="Buy & Hold (Equal Weight)", alpha=0.7
    )
    if not buys.empty:
        plt.scatter(
            buys.index, buys.values, marker="^", color="green", s=35, label="Net Long"
        )
    if not sells.empty:
        plt.scatter(
            sells.index, sells.values, marker="v", color="red", s=35, label="Net Short"
        )
    plt.title("CTA Momentum Sleeve vs Buy & Hold")
    plt.ylabel("Capital (Starting = 1.0)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def write_report(
    metrics: Dict[str, Dict[str, float]],
    outputs: StrategyOutputs,
    plot_path: Path,
    report_path: Path,
) -> None:
    sel = outputs.selection
    report_lines = [
        "# CTA Momentum Sleeve Report",
        "",
        f"- Performance Plot: {plot_path.name}",
        "",
        "## Asset Selection (training-driven)",
        f"- Momentum asset (highest {MOMENTUM_SELECT_DAYS}-day return): {sel.momentum_asset} ({format_pct(sel.momentum_score)})",
        f"- Convex asset (highest training skew): {sel.skew_asset} ({sel.skew_score:.2f})",
        f"- Core asset (liquid benchmark with strong training Sharpe): {sel.core_asset} (Sharpe {sel.core_sharpe:.2f})",
        "",
        "## Key Metrics",
    ]
    for label in ["train", "test", "full"]:
        data = metrics[label]
        sharpe_str = (
            f"{data['sharpe']:.2f}" if not math.isnan(data["sharpe"]) else "n/a"
        )
        mar_str = f"{data['mar_ratio']:.2f}" if not math.isnan(data["mar_ratio"]) else "n/a"
        report_lines.extend(
            [
                f"### {label.capitalize()}",
                f"- Annualized Net Return: {format_pct(data['annual_return'])}",
                f"- Annualized Gross Return: {format_pct(data['annual_gross_return'])}",
                f"- Sharpe Ratio: {sharpe_str}",
                f"- Max Drawdown (equity): {format_pct(data['max_drawdown'])}",
                f"- Max Drawdown (monthly): {format_pct(data['max_monthly_drawdown'])}",
                f"- MAR Ratio: {mar_str}",
                f"- Monthly Win Rate: {format_pct(data['monthly_win_rate'])}",
            ]
        )
        if label == "full" and "average_daily_turnover" in data:
            report_lines.append(
                f"- Average Daily Turnover (abs weight change): {data['average_daily_turnover']:.2f}"
            )
        report_lines.append("")

    bh = metrics["buy_and_hold_full"]
    sharpe_bh = f"{bh['sharpe']:.2f}" if not math.isnan(bh["sharpe"]) else "n/a"
    mar_bh = f"{bh['mar_ratio']:.2f}" if not math.isnan(bh["mar_ratio"]) else "n/a"
    report_lines.extend(
        [
            "## Buy & Hold Benchmark (Full Period)",
            f"- Annualized Return: {format_pct(bh['annual_return'])}",
            f"- Sharpe Ratio: {sharpe_bh}",
            f"- Max Drawdown (equity): {format_pct(bh['max_drawdown'])}",
            f"- MAR Ratio: {mar_bh}",
            f"- Monthly Win Rate: {format_pct(bh['monthly_win_rate'])}",
        ]
    )

    report_lines.extend(
        [
            "",
            "## Reproduction",
            "- Activate environment if needed: `conda activate myenv`",
            "- Run: `python code_1/strategy_analysis_1.py`",
        ]
    )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path()
    prices = load_price_data(dataset_path)

    selection = select_assets(prices)
    outputs = run_backtest(prices, selection)
    metrics = collect_metrics(outputs)

    plot_path = PLOTS_DIR / "performance_curve_1.png"
    save_plot(outputs, plot_path)

    report_path = OUTPUT_DIR / "strategy_report_1.md"
    write_report(metrics, outputs, plot_path, report_path)

    print("CTA Strategy Metrics (Test Period):")
    for key, val in metrics["test"].items():
        if key == "average_daily_turnover":
            continue
        if isinstance(val, float):
            if math.isnan(val):
                display = "n/a"
            elif any(term in key for term in ["return", "drawdown", "win_rate"]):
                display = format_pct(val)
            else:
                display = f"{val:.2f}"
        else:
            display = str(val)
        print(f"  {key}: {display}")

    print("\nBenchmark (Buy & Hold, Full Period):")
    for key, val in metrics["buy_and_hold_full"].items():
        if isinstance(val, float):
            if math.isnan(val):
                display = "n/a"
            elif any(term in key for term in ["return", "drawdown", "win_rate"]):
                display = format_pct(val)
            else:
                display = f"{val:.2f}"
        else:
            display = str(val)
        print(f"  {key}: {display}")


if __name__ == "__main__":
    main()
