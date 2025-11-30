"""
Shared signal generator for the fixed pairs spread strategy.

Both the offline backtester and the live paper-trading loop can import this
module to produce identical weights from an arbitrary price panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from pairs_strategy_core import StrategyConfig, build_spreads, compute_betas, generate_positions, volatility_target


@dataclass
class LiveStrategyState:
    """Container capturing the full signal state for diagnostics."""

    results: pd.DataFrame
    betas: Dict[str, Dict[str, float]]
    raw_positions: pd.DataFrame
    scaled_positions: pd.DataFrame
    weights: pd.DataFrame


class PairsSignalEngine:
    """
    Thin wrapper around the original backtest math that can be reused anywhere
    we have a price DataFrame shaped like the training data.
    """

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def _validate_columns(self, prices: pd.DataFrame) -> None:
        missing = [col for col in [self.cfg.base_asset, *self.cfg.pairs] if col not in prices.columns]
        if missing:
            raise ValueError(f"Price frame missing columns: {missing}")

    def compute_state(self, prices: pd.DataFrame) -> LiveStrategyState:
        """
        Mirrors `run_backtest` but operates on externally supplied prices.
        """
        self._validate_columns(prices)
        log_prices = np.log(prices)
        returns = log_prices.diff().fillna(0.0)

        split_ts = pd.Timestamp(self.cfg.train_split) if self.cfg.train_split is not None else None
        betas = compute_betas(log_prices, self.cfg.base_asset, self.cfg.pairs, split_ts)
        spreads = build_spreads(log_prices, self.cfg.base_asset, betas)
        spread_ma = spreads.rolling(self.cfg.spread_window).mean()
        spread_std = spreads.rolling(self.cfg.spread_window).std().replace(0.0, np.nan)
        zscores = (spreads - spread_ma) / spread_std

        positions = generate_positions(zscores, self.cfg)
        lagged_positions = positions.shift(1).fillna(0.0)

        base_returns = returns[self.cfg.base_asset]
        pair_returns = returns[list(self.cfg.pairs)]
        slopes = pd.Series(
            [betas[a]["slope"] for a in self.cfg.pairs],
            index=self.cfg.pairs,
            dtype=float,
        )

        hedge_adjusted = pair_returns.sub(
            base_returns.values.reshape(-1, 1) * slopes.values,
            axis=1,
        )

        raw_gross = (lagged_positions * hedge_adjusted).sum(axis=1)
        base_turnover = positions.diff().abs().sum(axis=1)
        hedge_turnover = positions.diff().abs().mul(np.abs(slopes), axis=1).sum(axis=1)
        turnover = base_turnover + hedge_turnover
        costs = turnover * self.cfg.cost_rate

        periods_per_year = self.cfg.periods_per_year
        rolling_vol = raw_gross.ewm(span=self.cfg.spread_window, adjust=False).std()
        ann_vol = rolling_vol * np.sqrt(periods_per_year)
        leverage_series = ann_vol.shift(1).apply(lambda v: volatility_target(v, self.cfg))
        leverage_series = leverage_series.clip(lower=0.0, upper=self.cfg.max_gross_leverage).fillna(0.0)

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
                "basket_return": base_returns,
                "active_pairs": (positions.abs() > 0).sum(axis=1),
                "signal_strength": positions.abs().sum(axis=1),
            }
        )

        scaled_positions = lagged_positions.mul(leverage_series, axis=0)
        base_weights = -scaled_positions.mul(slopes, axis=1).sum(axis=1)
        weights = scaled_positions.copy()
        weights[self.cfg.base_asset] = base_weights

        return LiveStrategyState(
            results=result,
            betas=betas,
            raw_positions=positions,
            scaled_positions=scaled_positions,
            weights=weights,
        )

    def latest_weights(self, prices: pd.DataFrame) -> Tuple[pd.Series, LiveStrategyState]:
        """Convenience helper returning the most recent weight vector."""
        state = self.compute_state(prices)
        latest_ts = state.weights.index[-1]
        latest_weights = state.weights.loc[latest_ts]
        return latest_weights, state
