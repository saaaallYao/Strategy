"""
Self-contained copy of the configuration and math helpers used by the pairs
strategy so that the package can operate without importing the original bundle.
"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
    train_split: str | None = None
    z_entry: float = 2.0
    z_exit: float = 0.8
    spread_window: int = 144
    base_weight: float = 0.18
    vol_target: float = 0.25
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


def compute_betas(
    log_prices: pd.DataFrame,
    base_asset: str,
    pair_assets: Iterable[str],
    split_ts: pd.Timestamp | None,
) -> Dict[str, Dict[str, float]]:
    if split_ts is None:
        train_mask = pd.Series(True, index=log_prices.index)
    else:
        train_mask = log_prices.index <= split_ts
    if train_mask.sum() < 3:
        # fallback: use full history if filtered set too small
        train_mask = pd.Series(True, index=log_prices.index)
    if train_mask.sum() < 3:
        raise ValueError("Not enough training samples for beta estimation.")
    x = np.column_stack((log_prices[base_asset].loc[train_mask], np.ones(train_mask.sum())))
    xtx = x.T @ x
    try:
        xx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse with small ridge for stability
        ridge = 1e-6 * np.eye(xtx.shape[0])
        xx_inv = np.linalg.pinv(xtx + ridge)
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
