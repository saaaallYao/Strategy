from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import pandas as pd
import requests

KUCOIN_SPOT_BASE_URL = "https://api.kucoin.com"


def to_timestamp(sec: int) -> pd.Timestamp:
    return pd.Timestamp(sec, unit="s", tz=timezone.utc).tz_convert(None)


def rule_to_seconds(rule: str) -> int:
    td = pd.Timedelta(rule)
    return int(td.total_seconds())


class KucoinSpotClient:
    """Lightweight REST client for KuCoin spot candles."""

    def __init__(self, base_url: str = KUCOIN_SPOT_BASE_URL, request_timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._timeout = request_timeout

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params or {}, timeout=self._timeout)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") != "200000":
            raise RuntimeError(f"KuCoin API error: {payload}")
        return payload["data"]

    def fetch_candles(
        self,
        symbol: str,
        start_sec: int,
        end_sec: int,
        granularity: str = "1min",
        limit: int = 1500,
    ) -> pd.DataFrame:
        cursor = start_sec
        frames: List[pd.DataFrame] = []
        step = limit * rule_to_seconds(granularity)
        while cursor <= end_sec:
            to_sec = min(cursor + step, end_sec)
            params = {
                "symbol": symbol,
                "type": granularity,
                "startAt": cursor,
                "endAt": to_sec,
            }
            data = self._get("/api/v1/market/candles", params=params)
            if not data:
                cursor = to_sec + rule_to_seconds(granularity)
                continue
            df = self._parse_candles(data)
            frames.append(df)
            cursor = int(df.index.max().timestamp()) + rule_to_seconds(granularity)
        if not frames:
            return pd.DataFrame(columns=["close"], dtype=float)
        return pd.concat(frames).sort_index()

    def fetch_recent_minutes(self, symbol: str, minutes: int, granularity: str = "1min") -> pd.DataFrame:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - minutes * 60_000
        return self.fetch_candles(symbol, start_ms // 1000, end_ms // 1000, granularity)

    def fetch_since(
        self,
        symbol: str,
        since_ts: pd.Timestamp,
        granularity: str = "1min",
    ) -> pd.DataFrame:
        start_ms = int(since_ts.timestamp() * 1000) + rule_to_seconds(granularity) * 1000
        end_ms = int(time.time() * 1000)
        return self.fetch_candles(symbol, start_ms // 1000, end_ms // 1000, granularity)

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


def symbol_to_column(symbol: str) -> str:
    sym = symbol.upper()
    if sym.startswith("BTC-"):
        return sym.replace("-", "_")
    return sym.replace("-", "")


def to_kucoin_symbol(symbol: str) -> str:
    """Normalize symbols to KuCoin spot format with hyphen."""
    sym = symbol.upper()
    if "-" in sym:
        return sym
    if sym.endswith("USDT"):
        return f"{sym[:-4]}-USDT"
    if sym.endswith("USD"):
        return f"{sym[:-3]}-USD"
    return sym


def fetch_history(
    symbols: Iterable[str],
    resample_rule: str,
    lookback_days: int,
    client: Optional[KucoinSpotClient] = None,
) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback_days * 24 * 60 * 60 * 1000
    cli = client or KucoinSpotClient()
    frames = []
    for sym in symbols:
        try:
            df = cli.fetch_candles(sym, start_ms // 1000, end_ms // 1000, granularity="1min")
        except Exception as exc:
            print(f"[kucoin] skip {sym}: {exc}", flush=True)
            continue
        if df.empty:
            print(f"[kucoin] empty history for {sym}", flush=True)
            continue
        df = df.rename(columns={"close": symbol_to_column(sym)})
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, axis=1).sort_index()
    history = history.ffill().dropna(how="all")
    return ensure_resample(history, resample_rule, base_column=symbol_to_column(symbols[0]))


def fetch_incremental(
    symbols: Iterable[str],
    resample_rule: str,
    last_timestamp: pd.Timestamp,
    cushion_bars: int = 5,
    client: Optional[KucoinSpotClient] = None,
) -> pd.DataFrame:
    sec_per_bar = rule_to_seconds(resample_rule)
    start_sec = int(last_timestamp.timestamp()) - cushion_bars * sec_per_bar
    start_sec = max(start_sec, 0)
    end_sec = int(time.time())
    cli = client or KucoinSpotClient()
    frames = []
    for sym in symbols:
        try:
            df = cli.fetch_candles(sym, start_sec, end_sec, granularity="1min")
        except Exception:
            continue
        if df.empty:
            continue
        df = df.rename(columns={"close": symbol_to_column(sym)})
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, axis=1).sort_index()
    history = history.ffill().dropna(how="all")
    return ensure_resample(history, resample_rule, base_column=symbol_to_column(symbols[0]))


def ensure_resample(df: pd.DataFrame, rule: str, base_column: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .last()
        .ffill()
        .dropna(subset=[base_column], how="any")
    )


def seconds_until_next_bar(rule: str, grace_seconds: int = 5) -> int:
    period = rule_to_seconds(rule)
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    epoch = int(now.timestamp())
    next_epoch = ((epoch // period) + 1) * period + grace_seconds
    return max(1, next_epoch - epoch)
