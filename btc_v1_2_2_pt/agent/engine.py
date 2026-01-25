import argparse
import os
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional, Any

from .config import AgentConfig
from .feed import AlpacaBarFeed
from .storage import CSVStorage
from .strategy_loader import load_strategy


class PaperEngine:
    def __init__(self, config: AgentConfig, strategy_path: str, mode: str = "offline"):
        self.config = config
        self.mode = mode
        self.storage = CSVStorage(config.log_dir)
        self.feed = AlpacaBarFeed(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            base_url=config.alpaca_base_url,
            symbols=config.symbols,
            timeframe=config.timeframe,
        )
        self.strategy = load_strategy(strategy_path, {"symbols": config.symbols})
        self.strategy_name = os.path.splitext(os.path.basename(strategy_path))[0]
        self.cash = config.starting_cash
        self.positions: Dict[str, float] = {s: 0.0 for s in config.symbols}
        self.avg_price: Dict[str, float] = {s: 0.0 for s in config.symbols}
        self.trade_api: Optional[Any] = None
        if self.mode == "online":
            try:
                import alpaca_trade_api as tradeapi
            except Exception as exc:
                raise RuntimeError(f"alpaca_trade_api is required for online mode: {exc}") from exc
            self.trade_api = tradeapi.REST(
                config.alpaca_api_key,
                config.alpaca_secret_key,
                config.alpaca_base_url,
                api_version="v2",
            )

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _update_position(self, symbol: str, side: str, qty: float, price: float) -> float:
        pnl = 0.0
        if side == "buy":
            cost = qty * price
            if cost > self.cash:
                return 0.0
            current_qty = self.positions[symbol]
            new_qty = current_qty + qty
            if new_qty > 0:
                self.avg_price[symbol] = (
                    (self.avg_price[symbol] * current_qty) + (price * qty)
                ) / new_qty
            self.positions[symbol] = new_qty
            self.cash -= cost
        elif side == "sell":
            current_qty = self.positions[symbol]
            sell_qty = min(qty, current_qty)
            if sell_qty <= 0:
                return 0.0
            pnl = (price - self.avg_price[symbol]) * sell_qty
            self.positions[symbol] = current_qty - sell_qty
            self.cash += sell_qty * price
            if self.positions[symbol] == 0:
                self.avg_price[symbol] = 0.0
        return pnl

    def _submit_order(self, symbol: str, side: str, qty: float) -> Optional[Any]:
        if self.trade_api is None:
            return None
        try:
            order = self.trade_api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="gtc",
            )
            order_id = getattr(order, "id", None)
            if order_id:
                try:
                    order = self.trade_api.get_order(order_id)
                except Exception:
                    pass
            return order
        except Exception as exc:
            print(f"[engine] submit_order failed: {exc}")
            return None

    def _snapshot_equity(self, prices: Dict[str, float]) -> Dict:
        exposure = sum(self.positions[s] * prices.get(s, 0.0) for s in self.positions)
        equity = self.cash + exposure
        return {
            "timestamp": self._now(),
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "exposure": round(exposure, 2),
        }

    def run(self) -> None:
        print(f"Starting paper engine... mode={self.mode}")
        if hasattr(self.strategy, "preload_prices") and self.strategy.preload_prices is not None:
            preload = self.strategy.preload_prices
            print(f"[engine] logging preload prices: {len(preload)} rows")
            preload_end = None
            for _, row in preload.iterrows():
                ts = pd.to_datetime(row["timestamp"]).isoformat()
                preload_end = ts
                self.storage.log_price({
                    "timestamp": ts,
                    "symbol": row["symbol"],
                    "strategy": self.strategy_name,
                    "open": float(row["close"]),
                    "high": float(row["close"]),
                    "low": float(row["close"]),
                    "close": float(row["close"]),
                    "volume": 0,
                })
            if preload_end:
                marker_path = os.path.join(self.config.log_dir, "preload_end.txt")
                with open(marker_path, "w") as f:
                    f.write(preload_end)
            self.strategy.preload_prices = None
        while True:
            latest_prices = {}
            for bar in self.feed.get_latest_bars():
                latest_prices[bar["symbol"]] = bar["close"]
                self.storage.log_price({
                    "timestamp": bar["timestamp"],
                    "symbol": bar["symbol"],
                    "strategy": self.strategy_name,
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "volume": bar["volume"],
                })

                state = {
                    "cash": self.cash,
                    "positions": dict(self.positions),
                }
                decision = self.strategy.on_bar(bar, state)
                action = decision.get("action", "hold")
                qty = float(decision.get("qty", 0))
                reason = decision.get("reason", "")
                zscore = decision.get("zscore")
                target_pos = decision.get("target_pos")
                blocked_reason = decision.get("blocked_reason", "")
                skip_log = bool(decision.get("skip_log", False))

                if not skip_log:
                    self.storage.log_signal({
                        "timestamp": bar["timestamp"],
                        "symbol": bar["symbol"],
                        "strategy": self.strategy_name,
                        "action": action,
                        "qty": qty,
                        "reason": reason,
                        "zscore": zscore,
                        "target_pos": target_pos,
                        "blocked_reason": blocked_reason,
                    })

                if action in {"buy", "sell"} and qty > 0:
                    order = None
                    exec_qty = qty
                    exec_price = bar["close"]
                    order_id = None
                    order_status = None
                    if self.mode == "online":
                        order = self._submit_order(bar["symbol"], action, qty)
                        if order is None:
                            continue
                        order_id = getattr(order, "id", None)
                        order_status = getattr(order, "status", None)
                        filled_qty = getattr(order, "filled_qty", None)
                        filled_price = getattr(order, "filled_avg_price", None)
                        try:
                            if filled_qty is not None:
                                exec_qty = float(filled_qty)
                            if filled_price is not None:
                                exec_price = float(filled_price)
                        except Exception:
                            exec_qty = qty
                            exec_price = bar["close"]

                    pnl = self._update_position(bar["symbol"], action, exec_qty, exec_price)
                    self.storage.log_trade({
                        "timestamp": bar["timestamp"],
                        "symbol": bar["symbol"],
                        "strategy": self.strategy_name,
                        "side": action,
                        "qty": exec_qty,
                        "price": exec_price,
                        "pnl": round(pnl, 4),
                        "position": self.positions[bar["symbol"]],
                        "reason": reason,
                        "order_id": order_id,
                        "order_status": order_status,
                        "mode": self.mode,
                    })

            equity_row = self._snapshot_equity(latest_prices)
            equity_row["strategy"] = self.strategy_name
            self.storage.log_equity(equity_row)
            self.feed.sleep(self.config.poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper trading engine")
    parser.add_argument("--strategy", required=True, help="Path to strategy file")
    parser.add_argument("--symbols", default="AAPL", help="Comma-separated symbols")
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AgentConfig(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        poll_interval=args.poll_interval,
    )
    engine = PaperEngine(config, args.strategy, mode=args.mode)
    engine.run()


if __name__ == "__main__":
    main()
