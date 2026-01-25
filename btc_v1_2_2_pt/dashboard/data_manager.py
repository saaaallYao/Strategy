import os
from typing import Dict

import numpy as np
import pandas as pd

from agent.metrics import sharpe_ratio, max_drawdown, total_return
from agent.config import AgentConfig


class DataManager:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.config = AgentConfig()

    def _load_csv(self, name: str) -> pd.DataFrame:
        path = os.path.join(self.log_dir, name)
        if not os.path.isfile(path):
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            # Tolerate schema changes or partial writes.
            try:
                return pd.read_csv(path, engine="python", on_bad_lines="skip")
            except Exception:
                return pd.DataFrame()

    def load_all(self) -> Dict[str, pd.DataFrame]:
        prices = self._load_csv("prices.csv")
        signals = self._load_csv("signals.csv")
        trades = self._load_csv("trades.csv")
        equity = self._load_csv("equity.csv")
        prices = self._ensure_strategy(prices)
        signals = self._ensure_strategy(signals)
        trades = self._ensure_strategy(trades)
        equity = self._ensure_strategy(equity)
        return {
            "prices": prices,
            "signals": signals,
            "trades": trades,
            "equity": equity,
        }

    def _ensure_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        if "strategy" not in df.columns:
            df = df.copy()
            df["strategy"] = "default"
        return df

    def compute_metrics(self, equity: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, float]:
        if equity.empty:
            base = {"sharpe": 0.0, "return": 0.0, "max_drawdown": 0.0}
        else:
            equity_vals = equity["equity"].astype(float).tolist()
            returns = np.diff(equity_vals) / np.maximum(equity_vals[:-1], 1e-9)
            base = {
                "sharpe": sharpe_ratio(returns.tolist(), self.config.bars_per_year),
                "return": total_return(equity_vals),
                "max_drawdown": max_drawdown(equity_vals),
            }

        trade_metrics = self._trade_metrics(trades)
        return {**base, **trade_metrics}

    def _trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        if trades.empty:
            return {
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "avg_hold_minutes": 0.0,
                "trade_freq_per_day": 0.0,
                "trade_count": 0.0,
            }

        trades = trades.copy()
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
        trades = trades.dropna(subset=["timestamp"])
        trades = trades.sort_values("timestamp")

        sell_trades = trades[trades["side"] == "sell"].copy()
        pnl_series = sell_trades.get("pnl", pd.Series(dtype=float)).astype(float)
        total_pnl = float(pnl_series.sum()) if not pnl_series.empty else 0.0
        win_rate = float((pnl_series > 0).mean()) if not pnl_series.empty else 0.0
        avg_pnl = float(pnl_series.mean()) if not pnl_series.empty else 0.0

        avg_hold_minutes = self._average_hold_time_minutes(trades)

        trade_count = float(len(trades))
        if len(trades) >= 2:
            duration_seconds = (trades["timestamp"].iloc[-1] - trades["timestamp"].iloc[0]).total_seconds()
            duration_days = max(duration_seconds / 86400.0, 1.0 / 86400.0)
            trade_freq_per_day = float(trade_count / duration_days)
        else:
            trade_freq_per_day = 0.0

        return {
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_hold_minutes": avg_hold_minutes,
            "trade_freq_per_day": trade_freq_per_day,
            "trade_count": trade_count,
        }

    def _average_hold_time_minutes(self, trades: pd.DataFrame) -> float:
        lots = {}
        total_minutes = 0.0
        total_qty = 0.0

        for _, row in trades.iterrows():
            symbol = row.get("symbol")
            side = row.get("side")
            qty = float(row.get("qty", 0) or 0)
            ts = row.get("timestamp")
            if symbol is None or qty <= 0 or pd.isna(ts):
                continue

            if symbol not in lots:
                lots[symbol] = []

            if side == "buy":
                lots[symbol].append([qty, ts])
            elif side == "sell":
                remaining = qty
                queue = lots[symbol]
                while remaining > 0 and queue:
                    lot_qty, lot_ts = queue[0]
                    take = min(remaining, lot_qty)
                    hold_minutes = (ts - lot_ts).total_seconds() / 60.0
                    total_minutes += hold_minutes * take
                    total_qty += take
                    lot_qty -= take
                    remaining -= take
                    if lot_qty <= 1e-9:
                        queue.pop(0)
                    else:
                        queue[0][0] = lot_qty

        if total_qty <= 0:
            return 0.0
        return total_minutes / total_qty
