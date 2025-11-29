from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

KUCOIN_SPOT_BASE_URL = "https://api.kucoin.com"


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def to_timestamp(sec: int) -> pd.Timestamp:
    return pd.Timestamp(sec, unit="s", tz=timezone.utc)


def granularity_to_sec(granularity: str) -> int:
    if granularity == "1min":
        return 60
    raise ValueError(f"Unsupported granularity {granularity}")


class KucoinSpotClient:
    """Lightweight REST client for KuCoin spot minute candles."""

    def __init__(self, base_url: str = KUCOIN_SPOT_BASE_URL, request_timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._timeout = request_timeout

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params or {}, timeout=self._timeout)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") != "200000":
            raise RuntimeError(f"KuCoin API error: {payload}")
        return payload["data"]

    def _get_with_retry(self, path: str, params: Optional[Dict] = None, attempts: int = 3, backoff: float = 0.5) -> Dict:
        for i in range(attempts):
            try:
                return self._get(path, params=params)
            except requests.RequestException as exc:
                if i == attempts - 1:
                    raise
                time.sleep(backoff * (i + 1))

    def fetch_candles(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        granularity: str = "1min",
        limit: int = 1500,
    ) -> pd.DataFrame:
        start_sec = start_ms // 1000
        end_sec = end_ms // 1000
        cursor = start_sec
        frames: List[pd.DataFrame] = []
        step = granularity_to_sec(granularity) * limit
        while cursor <= end_sec:
            to_sec = min(cursor + step, end_sec)
            params = {
                "symbol": symbol,
                "type": granularity,
                "startAt": cursor,
                "endAt": to_sec,
            }
            data = self._get_with_retry("/api/v1/market/candles", params=params)
            if not data:
                cursor = to_sec + granularity_to_sec(granularity)
                continue
            df = self._parse_candles(data)
            frames.append(df)
            cursor = int(df.index.max().timestamp()) + granularity_to_sec(granularity)
        if not frames:
            return pd.DataFrame(columns=["close"], dtype=float)
        return pd.concat(frames).sort_index()

    def fetch_recent_minutes(self, symbol: str, minutes: int, granularity: str = "1min") -> pd.DataFrame:
        end_ms = utc_now_ms()
        start_ms = end_ms - minutes * 60_000
        return self.fetch_candles(symbol, start_ms, end_ms, granularity)

    def fetch_since(self, symbol: str, since_ts: pd.Timestamp, granularity: str = "1min") -> pd.DataFrame:
        start_ms = int(since_ts.timestamp() * 1000) + granularity_to_sec(granularity) * 1000
        end_ms = utc_now_ms()
        return self.fetch_candles(symbol, start_ms, end_ms, granularity)

    @staticmethod
    def _parse_candles(data: Iterable[Iterable]) -> pd.DataFrame:
        records = []
        for entry in data:
            # [time, open, close, high, low, volume, turnover]
            ts_sec = int(entry[0])
            close = float(entry[2])
            records.append((to_timestamp(ts_sec), close))
        if not records:
            return pd.DataFrame(columns=["close"], dtype=float)
        df = pd.DataFrame(records, columns=["datetime", "close"]).set_index("datetime")
        return df[~df.index.duplicated(keep="last")]


def default_universe() -> List[str]:
    return [
        "BTC-USDT",
        "ETH-USDT",
        "SOL-USDT",
        "XRP-USDT",
        "DOGE-USDT",
        "ADA-USDT",
        "POL-USDT",  # MATIC rebranded; KuCoin uses POL-USDT
    ]


def get_log_paths(prefix: str) -> tuple[Path, Path, Path]:
    trade_log = Path(f"data/paper_trades_{prefix}_1.csv")
    equity_log = Path(f"data/paper_equity_curve_{prefix}_1.csv")
    signal_log = Path(f"data/paper_signals_{prefix}_1.csv")
    return trade_log, equity_log, signal_log


@dataclass
class PaperPortfolio:
    symbols: List[str]
    fee_per_turnover: float = 0.0003
    equity: float = 1.0
    weights: pd.Series = field(init=False)
    trades: List[Dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.weights = pd.Series(0.0, index=self.symbols)

    def apply_returns(self, returns: pd.Series) -> None:
        aligned = returns.reindex(self.weights.index).fillna(0.0)
        pnl = float((self.weights * aligned).sum())
        self.equity *= max(1.0 + pnl, 1e-9)

    def rebalance(self, timestamp: pd.Timestamp, target: pd.Series) -> Dict:
        current = self.weights.reindex(target.index).fillna(0.0)
        delta = target - current
        turnover = float(delta.abs().sum())
        cost_fraction = turnover * self.fee_per_turnover
        self.equity *= max(1.0 - cost_fraction, 1e-9)
        trade = {
            "timestamp": timestamp.isoformat(),
            "from_weights": current.to_dict(),
            "to_weights": target.to_dict(),
            "turnover": turnover,
            "cost_fraction": cost_fraction,
            "equity_after_cost": self.equity,
        }
        self.weights = target
        self.trades.append(trade)
        return trade


@dataclass
class LiveResidualPairsTrader:
    universe: List[str] = field(default_factory=default_universe)
    resample_rule: str = "1h"
    lookback_days: int = 20
    zwin_days: int = 45
    k_per_side: int = 3
    z_threshold: float = 0.8
    fee_per_side: float = 0.0003
    poll_interval: int = 60
    history_minutes: int = 120_000
    price_client: KucoinSpotClient = field(default_factory=KucoinSpotClient)
    trade_log_path: Optional[Path] = None
    equity_log_path: Optional[Path] = None
    signal_log_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.universe = list(dict.fromkeys(self.universe))
        prefix = "residual_pairs"
        if self.trade_log_path is None or self.equity_log_path is None or self.signal_log_path is None:
            trade_log, equity_log, signal_log = get_log_paths(prefix)
            self.trade_log_path = self.trade_log_path or trade_log
            self.equity_log_path = self.equity_log_path or equity_log
            self.signal_log_path = self.signal_log_path or signal_log

        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.equity_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.signal_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.history: pd.DataFrame = pd.DataFrame()
        self.last_timestamp: Optional[pd.Timestamp] = None
        self._last_hourly_ts: Optional[pd.Timestamp] = None
        self.portfolio = PaperPortfolio(symbols=self.universe, fee_per_turnover=self.fee_per_side)

        LOGGER.info(
            "Log paths: trades=%s equity=%s signals=%s",
            self.trade_log_path,
            self.equity_log_path,
            self.signal_log_path,
        )

    def bootstrap_history(self) -> None:
        minutes = self.history_minutes
        frames = []
        for sym in self.universe:
            try:
                df = self.price_client.fetch_recent_minutes(sym, minutes)
            except Exception as exc:
                LOGGER.warning("Skipping %s during bootstrap: %s", sym, exc)
                continue
            if df.empty:
                LOGGER.warning("No history for %s during bootstrap", sym)
                continue
            frames.append(df.rename(columns={"close": sym}))
        if not frames:
            raise RuntimeError("Unable to bootstrap history from KuCoin spot.")
        df_all = pd.concat(frames, axis=1).sort_index()
        self.history = df_all.ffill().dropna(how="all")
        self.last_timestamp = self.history.index.max()
        LOGGER.info("Bootstrapped %d minute bars ending at %s", len(self.history), self.last_timestamp)

    def _log_trade(self, trade: Dict) -> None:
        header_needed = not self.trade_log_path.exists()
        with self.trade_log_path.open("a", encoding="utf-8") as fh:
            if header_needed:
                fh.write("timestamp,from_weights,to_weights,turnover,cost_fraction,equity_after_cost\n")
            fh.write(
                f"{trade['timestamp']},{json.dumps(trade['from_weights'])},{json.dumps(trade['to_weights'])},"
                f"{trade['turnover']:.6f},{trade['cost_fraction']:.6f},{trade['equity_after_cost']:.6f}\n"
            )

    def _log_equity(self, timestamp: pd.Timestamp) -> None:
        header_needed = not self.equity_log_path.exists()
        with self.equity_log_path.open("a", encoding="utf-8") as fh:
            if header_needed:
                fh.write("timestamp,equity," + ",".join(self.portfolio.weights.index) + "\n")
            weights_csv = ",".join(f"{self.portfolio.weights.get(sym,0.0):.4f}" for sym in self.portfolio.weights.index)
            fh.write(f"{timestamp.isoformat()},{self.portfolio.equity:.6f},{weights_csv}\n")

    def _log_signal(self, timestamp: pd.Timestamp, target: pd.Series) -> None:
        header_needed = not self.signal_log_path.exists()
        with self.signal_log_path.open("a", encoding="utf-8") as fh:
            if header_needed:
                fh.write("timestamp," + ",".join(target.index) + "\n")
            weights_csv = ",".join(f"{target.get(sym,0.0):.4f}" for sym in target.index)
            fh.write(f"{timestamp.isoformat()},{weights_csv}\n")

    def _compute_weights(self, prices_1h: pd.DataFrame) -> Optional[pd.Series]:
        lb = int(self.lookback_days * 24)
        zw = int(self.zwin_days * 24)
        if len(prices_1h) < max(lb, zw) + 5:
            return None

        rets_lin = prices_1h.pct_change()
        rets_log = np.log(prices_1h).diff()
        btc_r_lin = rets_lin["BTC-USDT"]
        btc_r_log = rets_log["BTC-USDT"]
        alts = [c for c in prices_1h.columns if c != "BTC-USDT"]

        corr = rets_log[alts].rolling(lb).corr(btc_r_log)
        std_alt = rets_log[alts].rolling(lb).std()
        std_btc = btc_r_log.rolling(lb).std()
        beta = (corr * std_alt).div(std_btc, axis=0).shift(1).clip(-5, 5)

        resid_log = rets_log[alts] - beta.mul(btc_r_log, axis=0)
        spread = resid_log.cumsum()
        m = spread.rolling(zw, min_periods=max(zw // 4, 1)).mean().shift(1)
        s = spread.rolling(zw, min_periods=max(zw // 4, 1)).std().shift(1)
        z = (spread - m) / s

        zt = float(self.z_threshold)
        k = int(self.k_per_side)
        z_latest = z.iloc[-1]
        valid = z_latest[np.isfinite(z_latest)]
        if valid.empty:
            return None

        longs = valid[valid <= -zt].sort_values().head(k)
        shorts = valid[valid >= zt].sort_values(ascending=False).head(k)

        target = pd.Series(0.0, index=prices_1h.columns.tolist())
        if len(longs) > 0:
            target.loc[longs.index] = 0.5 / len(longs)
        if len(shorts) > 0:
            target.loc[shorts.index] = -0.5 / len(shorts)

        beta_last = beta.iloc[-1] if len(beta) else pd.Series(0.0, index=alts)
        beta_last = beta_last.fillna(0.0)
        target["BTC-USDT"] = -float((target.drop(labels=["BTC-USDT"], errors="ignore") * beta_last).sum())
        return target

    def _resample_hourly(self) -> pd.DataFrame:
        return self.history.resample(self.resample_rule.lower()).last().dropna(how="all")

    def step_once(self) -> None:
        if self.last_timestamp is None:
            self.bootstrap_history()
            return

        new_frames = []
        for sym in self.universe:
            try:
                df = self.price_client.fetch_since(sym, self.last_timestamp, granularity="1min")
            except Exception as exc:
                LOGGER.warning("Skipping %s during fetch: %s", sym, exc)
                continue
            if df.empty:
                continue
            new_frames.append(df.rename(columns={"close": sym}))
        if new_frames:
            df_new = pd.concat(new_frames, axis=1)
            df_new = df_new[df_new.index > self.last_timestamp]
            if not df_new.empty:
                self.history = pd.concat([self.history, df_new]).sort_index()
                self.history = self.history.ffill().dropna(how="all").iloc[-self.history_minutes :]
                self.last_timestamp = self.history.index.max()

        prices_1h = self._resample_hourly()
        if prices_1h.empty:
            return
        last_hour = prices_1h.index.max()
        if self._last_hourly_ts is None:
            self._last_hourly_ts = last_hour
            return
        if last_hour <= self._last_hourly_ts:
            return

        prev = prices_1h.iloc[-2]
        curr = prices_1h.iloc[-1]
        hourly_returns = curr / prev - 1.0
        self.portfolio.apply_returns(hourly_returns)

        target = self._compute_weights(prices_1h)
        if target is not None:
            trade = self.portfolio.rebalance(last_hour, target)
            self._log_trade(trade)
            self._log_signal(last_hour, target)
            LOGGER.info(
                "Rebalanced at %s | equity=%.4f | turnover=%.3f",
                last_hour,
                self.portfolio.equity,
                trade["turnover"],
            )

        self._log_equity(last_hour)
        self._last_hourly_ts = last_hour

    def run_forever(self) -> None:
        LOGGER.info("Starting residual pairs paper trader on KuCoin spot.")
        if self.last_timestamp is None:
            self.bootstrap_history()
        while True:
            try:
                self.step_once()
            except Exception as exc:
                LOGGER.exception("Error during trading loop: %s", exc)
            time.sleep(self.poll_interval)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def create_trader_from_env() -> LiveResidualPairsTrader:
    universe_env = os.environ.get("RESIDUAL_UNIVERSE", "")
    universe = [s.strip() for s in universe_env.split(",") if s.strip()] if universe_env else default_universe()
    poll_interval = int(os.environ.get("RESIDUAL_POLL_INTERVAL", "60"))
    history_minutes = int(os.environ.get("RESIDUAL_HISTORY_MINUTES", "120000"))
    fee = float(os.environ.get("RESIDUAL_FEE_PER_SIDE", "0.0003"))
    lookback_days = int(os.environ.get("RESIDUAL_LOOKBACK_DAYS", "20"))
    zwin_days = int(os.environ.get("RESIDUAL_ZWIN_DAYS", "45"))
    k_per_side = int(os.environ.get("RESIDUAL_K_PER_SIDE", "3"))
    z_threshold = float(os.environ.get("RESIDUAL_Z_THRESHOLD", "0.8"))
    return LiveResidualPairsTrader(
        universe=universe,
        poll_interval=poll_interval,
        history_minutes=history_minutes,
        fee_per_side=fee,
        lookback_days=lookback_days,
        zwin_days=zwin_days,
        k_per_side=k_per_side,
        z_threshold=z_threshold,
    )


def main() -> None:
    configure_logging()
    trader = create_trader_from_env()
    trader.run_forever()


if __name__ == "__main__":
    main()
