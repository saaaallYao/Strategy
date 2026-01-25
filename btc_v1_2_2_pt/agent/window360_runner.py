import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import AgentConfig
from .storage import CSVStorage


def _resolve_window360_root() -> Path:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    candidate = project_root / "btc_v1_2_2" / "window360"
    if not candidate.exists():
        raise FileNotFoundError(f"window360 not found at {candidate}")
    return candidate


WINDOW360_ROOT = _resolve_window360_root()
sys.path.insert(0, str(WINDOW360_ROOT))

from fina.strategies.crypto.btc_eth_v1.live_monitor import LiveMonitor  # noqa: E402


class Window360Runner:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.storage = CSVStorage(config.log_dir)
        self.monitor = LiveMonitor()
        self.last_trade_count = 0
        self.strategy_name = "window360"

    def _iso_ts(self, ts) -> str:
        ts = pd.to_datetime(ts, errors="coerce")
        if pd.isna(ts):
            return datetime.now(timezone.utc).isoformat()
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        return ts.isoformat()

    def _log_prices(self, px: pd.DataFrame) -> None:
        if px is None or px.empty:
            return
        last_ts = px.index[-1]
        ts = self._iso_ts(last_ts)
        for col in px.columns:
            if not col.endswith("_USD"):
                continue
            symbol = col.replace("_", "/")
            close = float(px[col].iloc[-1])
            self.storage.log_price({
                "timestamp": ts,
                "symbol": symbol,
                "strategy": self.strategy_name,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 0,
            })

    def _log_signal(self, ts: str) -> None:
        signal = getattr(self.monitor, "last_signal", 0)
        if signal > 0:
            action = "buy"
        elif signal < 0:
            action = "sell"
        else:
            action = "hold"
        qty = abs(float(getattr(self.monitor, "current_position", 0) or 0))
        self.storage.log_signal({
            "timestamp": ts,
            "symbol": "BTC/USD",
            "strategy": self.strategy_name,
            "action": action,
            "qty": qty,
            "reason": "window360_signal",
        })

    def _log_equity(self, ts: str) -> None:
        equity = self.config.starting_cash * float(getattr(self.monitor, "current_equity", 1.0))
        position = abs(float(getattr(self.monitor, "current_position", 0) or 0))
        exposure = equity * position
        cash = equity - exposure
        self.storage.log_equity({
            "timestamp": ts,
            "strategy": self.strategy_name,
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "exposure": round(exposure, 2),
        })

    def _log_trades(self) -> None:
        history = list(getattr(self.monitor, "trade_history", []))
        if len(history) <= self.last_trade_count:
            return
        new_trades = history[self.last_trade_count :]
        self.last_trade_count = len(history)

        for trade in new_trades:
            action = trade.get("action", "")
            pos = float(trade.get("position", 0) or 0)
            if action in {"open"}:
                side = "buy" if pos > 0 else "sell"
            elif action in {"close", "stop_loss"}:
                side = "sell" if pos > 0 else "buy"
            elif action in {"reverse"}:
                side = "sell" if pos > 0 else "buy"
            else:
                side = "buy" if pos > 0 else "sell"

            price = float(trade.get("price", 0) or 0)
            pnl_raw = float(trade.get("pnl", 0) or 0)
            pnl = pnl_raw * self.config.starting_cash
            ts = self._iso_ts(trade.get("timestamp"))
            self.storage.log_trade({
                "timestamp": ts,
                "symbol": "BTC/USD",
                "strategy": self.strategy_name,
                "side": side,
                "qty": abs(pos),
                "price": price,
                "pnl": round(pnl, 4),
                "position": float(getattr(self.monitor, "current_position", 0) or 0),
                "reason": action,
            })

    def run(self, interval: int) -> None:
        print("Starting window360 runner...")
        while True:
            px, etf_px = self.monitor.update_data()
            if px is None or px.empty:
                time.sleep(interval)
                continue
            self.monitor.execute_strategy(px, etf_px)
            ts = self._iso_ts(px.index[-1])
            self._log_prices(px)
            self._log_signal(ts)
            self._log_trades()
            self._log_equity(ts)
            time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run window360 strategy with dashboard logging")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AgentConfig(symbols=["BTC/USD", "ETH/USD"], poll_interval=args.interval)
    runner = Window360Runner(config)
    runner.run(args.interval)


if __name__ == "__main__":
    main()
