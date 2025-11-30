"""
Minimal paper broker that works with weight outputs from the pairs strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    trade_id: str
    timestamp: pd.Timestamp
    symbol: str
    side: str
    qty: float
    price: float
    notional: float
    fee: float
    comment: str = ""


class PairsPaperBroker:
    """
    Simple portfolio accountant that rebalances to target notionals per symbol.
    """

    def __init__(self, fee_rate: float = 0.0005, initial_equity: float = 1_000_000.0):
        self.fee_rate = float(fee_rate)
        self.initial_equity = float(initial_equity)
        self.cash = float(initial_equity)
        self.positions: Dict[str, float] = {}
        self.last_prices: Dict[str, float] = {}
        self.trades: List[TradeRecord] = []
        self.trade_counter = 0

    def mark_to_market(self, prices: pd.Series) -> None:
        for sym, px in prices.items():
            if pd.notna(px):
                self.last_prices[sym] = float(px)

    def _position_value(self) -> float:
        value = 0.0
        for sym, qty in self.positions.items():
            px = self.last_prices.get(sym, np.nan)
            if np.isnan(px):
                continue
            value += qty * px
        return value

    def total_equity(self) -> float:
        return self.cash + self._position_value()

    def rebalance_to_weights(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        timestamp: Optional[pd.Timestamp] = None,
        comment: str = "rebalance",
    ) -> None:
        """
        Convert target weights (fraction of equity) into trades.
        """
        self.mark_to_market(prices)
        equity_before = self.total_equity()
        if equity_before <= 0:
            return

        time_index = timestamp or getattr(prices, "name", None)

        for sym, weight in target_weights.items():
            price = prices.get(sym)
            if pd.isna(price) or float(price) <= 0:
                continue

            price = float(price)
            weight = float(weight)

            target_notional = weight * equity_before
            current_qty = self.positions.get(sym, 0.0)
            current_notional = current_qty * price
            diff = target_notional - current_notional
            if abs(diff) < 1e-9:
                continue

            qty = diff / price
            side = "BUY" if qty > 0 else "SELL"
            fee = abs(diff) * self.fee_rate

            # Update cash and position
            self.cash -= diff + fee
            self.positions[sym] = current_qty + qty

            self.trade_counter += 1
            trade_id = f"trade_{self.trade_counter}"
            trade = TradeRecord(
                trade_id=trade_id,
                timestamp=time_index,
                symbol=sym,
                side=side,
                qty=qty,
                price=price,
                notional=diff,
                fee=fee,
                comment=comment,
            )
            self.trades.append(trade)

    def records(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.trades])

    def trade_records_since(self, start_index: int) -> List[TradeRecord]:
        return self.trades[start_index:]

    def snapshot_equity(self) -> Dict[str, float]:
        position_value = self._position_value()
        total = self.cash + position_value
        realized = self.cash - self.initial_equity
        return {
            "equity": total,
            "cash": self.cash,
            "position_value": position_value,
            "realized_pnl": realized,
        }
