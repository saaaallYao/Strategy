from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from pairs_strategy.core import StrategyConfig
from pairs_strategy.signal import PairsSignalEngine
from code.kucoin_client import (
    fetch_history,
    fetch_incremental,
    seconds_until_next_bar,
    symbol_to_column,
    to_kucoin_symbol,
)
from fixed_pairs_pt.broker import PairsPaperBroker

LOGGER = logging.getLogger(__name__)


def get_log_paths(prefix: str = "fixed_pairs") -> tuple[Path, Path, Path]:
    trade_log = Path(f"data/paper_trades_{prefix}_1.csv")
    equity_log = Path(f"data/paper_equity_curve_{prefix}_1.csv")
    signal_log = Path(f"data/paper_signals_{prefix}_1.csv")
    return trade_log, equity_log, signal_log


DEFAULT_PAIRS = [
    "ETC-USDT",
    "APT-USDT",
    "ARB-USDT",
]


@dataclass
class LiveFixedPairsTrader:
    base_symbol: str = "BTC-USDT"
    pairs: Optional[List[str]] = field(default_factory=lambda: list(DEFAULT_PAIRS))
    resample_rule: str = "15min"
    seed_days: int = 200
    initial_capital: float = 1_000_000.0
    output_prefix: str = "fixed_pairs"
    poll_grace_seconds: int = 5
    cushion_bars: int = 5
    max_bars: int = 0
    trade_log_path: Optional[Path] = None
    equity_log_path: Optional[Path] = None
    signal_log_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.pairs = [to_kucoin_symbol(p) for p in (list(self.pairs) if self.pairs else list(DEFAULT_PAIRS))]
        base_sym = to_kucoin_symbol(self.base_symbol)
        self.symbols = [base_sym] + list(self.pairs)
        if self.trade_log_path is None or self.equity_log_path is None or self.signal_log_path is None:
            trade_log, equity_log, signal_log = get_log_paths(self.output_prefix)
            self.trade_log_path = self.trade_log_path or trade_log
            self.equity_log_path = self.equity_log_path or equity_log
            self.signal_log_path = self.signal_log_path or signal_log

        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.equity_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.signal_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Map KuCoin symbols to strategy columns (BTC_USD, ETCUSDT, etc.)
        base_col = symbol_to_column(base_sym)
        pair_cols = tuple(symbol_to_column(p) for p in self.pairs)

        self.cfg = StrategyConfig(
            base_asset=base_col,
            pairs=pair_cols,
            resample_rule=self.resample_rule,
            train_split=None,  # use full history for beta estimation
        )
        self.engine = PairsSignalEngine(self.cfg)
        self.broker = PairsPaperBroker(fee_rate=self.cfg.cost_rate, initial_equity=self.initial_capital)

        self.history: pd.DataFrame = pd.DataFrame()
        self.last_timestamp: Optional[pd.Timestamp] = None
        self.last_trade_idx = 0

        LOGGER.info(
            "Log paths: trades=%s equity=%s signals=%s",
            self.trade_log_path,
            self.equity_log_path,
            self.signal_log_path,
        )

    def bootstrap_history(self) -> None:
        history = fetch_history(self.symbols, resample_rule=self.resample_rule, lookback_days=self.seed_days)
        history = history[[self.cfg.base_asset, *self.cfg.pairs]].dropna()
        if history.empty:
            raise RuntimeError("Bootstrap history is empty.")
        self.history = history
        self.last_timestamp = history.index.max()
        if len(self.history) >= 30:
            self.last_trade_idx = self._rebalance_once()
        else:
            LOGGER.warning("Not enough history to rebalance (len=%d); waiting for more bars.", len(self.history))
        LOGGER.info("Bootstrapped %d bars ending at %s", len(history), self.last_timestamp)

    def _log_trade(self, trade: dict) -> None:
        header_needed = not self.trade_log_path.exists()
        with self.trade_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow(["trade_id", "timestamp", "symbol", "side", "qty", "price", "notional", "fee", "comment"])
            writer.writerow(
                [
                    trade["trade_id"],
                    trade["timestamp"],
                    trade["symbol"],
                    trade["side"],
                    f"{trade['qty']:.8f}",
                    f"{trade['price']:.6f}",
                    f"{trade['notional']:.2f}",
                    f"{trade['fee']:.2f}",
                    trade.get("comment", ""),
                ]
            )

    def _log_trades_since(self, start_idx: int) -> int:
        new_trades = self.broker.trade_records_since(start_idx)
        if not new_trades:
            return start_idx
        for tr in new_trades:
            self._log_trade(tr.__dict__)
        return start_idx + len(new_trades)

    def _log_equity(self, timestamp: pd.Timestamp) -> None:
        header_needed = not self.equity_log_path.exists()
        snap = self.broker.snapshot_equity()
        weights = self.engine.latest_weights(self.history)[0]
        with self.equity_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow(["timestamp", "equity", "cash", "position_value", "realized_pnl", *weights.index])
            writer.writerow(
                [
                    timestamp,
                    f"{snap['equity']:.2f}",
                    f"{snap['cash']:.2f}",
                    f"{snap['position_value']:.2f}",
                    f"{snap['realized_pnl']:.2f}",
                    *[f"{weights.get(sym, 0.0):.6f}" for sym in weights.index],
                ]
            )

    def _log_signal(self, timestamp: pd.Timestamp, weights: pd.Series) -> None:
        header_needed = not self.signal_log_path.exists()
        with self.signal_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow(["timestamp", *weights.index])
            writer.writerow([timestamp, *[f"{weights.get(sym, 0.0):.6f}" for sym in weights.index]])

    def _rebalance_once(self) -> int:
        if len(self.history) < 30:
            return self.last_trade_idx
        try:
            weights, _ = self.engine.latest_weights(self.history)
        except np.linalg.LinAlgError as exc:
            LOGGER.warning("Skipping rebalance due to singular matrix in beta estimation: %s", exc)
            return self.last_trade_idx
        except Exception as exc:
            LOGGER.warning("Skipping rebalance due to signal error: %s", exc)
            return self.last_trade_idx
        ts = self.history.index[-1]
        prices = self.history.loc[ts, weights.index]
        self.broker.rebalance_to_weights(weights, prices, timestamp=ts)
        self._log_signal(ts, weights)
        self._log_equity(ts)
        return self._log_trades_since(self.last_trade_idx)

    def step_once(self) -> None:
        wait = seconds_until_next_bar(self.resample_rule, grace_seconds=self.poll_grace_seconds)
        time.sleep(wait)
        last_ts = self.history.index[-1]
        recent = fetch_incremental(
            symbols=self.symbols,
            resample_rule=self.resample_rule,
            last_timestamp=last_ts,
            cushion_bars=self.cushion_bars,
        )
        if recent.empty:
            return

        history = pd.concat([self.history, recent], axis=0)
        history = history[~history.index.duplicated(keep="last")]
        history = history[[self.cfg.base_asset, *self.cfg.pairs]].dropna()
        if self.max_bars and self.max_bars > 0:
            history = history.tail(self.max_bars)
        self.history = history

        self.last_trade_idx = self._rebalance_once()
        LOGGER.info("[live] ts=%s equity=%.2f", self.history.index[-1], self.broker.total_equity())

    def run_forever(self) -> None:
        LOGGER.info("Starting fixed pairs paper trader (KuCoin spot klines).")
        if self.last_timestamp is None:
            self.bootstrap_history()
        while True:
            try:
                self.step_once()
            except Exception as exc:
                LOGGER.exception("Error in trading loop: %s", exc)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def create_trader_from_env() -> LiveFixedPairsTrader:
    base_symbol = os.environ.get("FP_BASE_SYMBOL", "BTC-USDT")
    pairs_env = os.environ.get("FP_PAIRS", "")
    pairs_raw = [p.strip() for p in pairs_env.split(",") if p.strip()] if pairs_env else list(DEFAULT_PAIRS)
    pairs = [to_kucoin_symbol(p) for p in pairs_raw]
    resample_rule = os.environ.get("FP_RESAMPLE_RULE", "15min")
    seed_days = int(os.environ.get("FP_SEED_DAYS", "200"))
    initial_capital = float(os.environ.get("FP_INITIAL_CAPITAL", "1000000"))
    cushion_bars = int(os.environ.get("FP_CUSHION_BARS", "5"))
    poll_grace_seconds = int(os.environ.get("FP_BAR_GRACE_SECONDS", "5"))
    max_bars = int(os.environ.get("FP_MAX_BARS", "0"))
    prefix = os.environ.get("FP_LOG_PREFIX", "fixed_pairs")
    return LiveFixedPairsTrader(
        base_symbol=base_symbol,
        pairs=pairs,
        resample_rule=resample_rule,
        seed_days=seed_days,
        initial_capital=initial_capital,
        cushion_bars=cushion_bars,
        poll_grace_seconds=poll_grace_seconds,
        max_bars=max_bars,
        output_prefix=prefix,
    )


def main() -> None:
    configure_logging()
    trader = create_trader_from_env()
    trader.run_forever()


if __name__ == "__main__":
    main()
