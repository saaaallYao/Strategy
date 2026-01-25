import random
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List

import alpaca_trade_api as tradeapi


class AlpacaBarFeed:
    def __init__(self, api_key: str, secret_key: str, base_url: str, symbols: List[str], timeframe: str):
        self.symbols = symbols
        self.timeframe = timeframe
        self.simulation = not (api_key and secret_key)
        self.crypto_symbols = [s for s in symbols if "/" in s]
        self.equity_symbols = [s for s in symbols if "/" not in s]
        self._last_prices = {s: random.uniform(50, 200) for s in symbols}
        if not self.simulation:
            self.api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        else:
            self.api = None

    def _simulate_bar(self, symbol: str) -> Dict:
        last = self._last_prices[symbol]
        step = random.uniform(-0.5, 0.5)
        price = max(1.0, last + step)
        self._last_prices[symbol] = price
        ts = datetime.now(timezone.utc).isoformat()
        return {
            "symbol": symbol,
            "timestamp": ts,
            "open": price - 0.1,
            "high": price + 0.2,
            "low": price - 0.2,
            "close": price,
            "volume": random.randint(100, 1000),
        }

    def get_latest_bars(self) -> Iterable[Dict]:
        if self.simulation:
            for symbol in self.symbols:
                yield self._simulate_bar(symbol)
            return

        for symbol in self.equity_symbols:
            try:
                bar = self.api.get_latest_bar(symbol)
            except Exception as exc:
                print(f"[feed] equity latest bar failed for {symbol}: {exc}")
                continue
            yield {
                "symbol": symbol,
                "timestamp": bar.t.isoformat(),
                "open": float(bar.o),
                "high": float(bar.h),
                "low": float(bar.l),
                "close": float(bar.c),
                "volume": int(bar.v),
            }

        if not self.crypto_symbols:
            return

        if hasattr(self.api, "get_latest_crypto_bars"):
            try:
                bars = self.api.get_latest_crypto_bars(self.crypto_symbols)
            except Exception as exc:
                print(f"[feed] crypto latest bars failed: {exc}")
                return
            if hasattr(bars, "data"):
                bars = bars.data
            for symbol in self.crypto_symbols:
                bar = bars.get(symbol)
                if bar is None:
                    continue
                yield {
                    "symbol": symbol,
                    "timestamp": bar.t.isoformat(),
                    "open": float(bar.o),
                    "high": float(bar.h),
                    "low": float(bar.l),
                    "close": float(bar.c),
                    "volume": float(bar.v),
                }
            return

        if hasattr(self.api, "get_latest_crypto_bar"):
            for symbol in self.crypto_symbols:
                try:
                    bar = self.api.get_latest_crypto_bar(symbol)
                except Exception as exc:
                    print(f"[feed] crypto latest bar failed for {symbol}: {exc}")
                    continue
                yield {
                    "symbol": symbol,
                    "timestamp": bar.t.isoformat(),
                    "open": float(bar.o),
                    "high": float(bar.h),
                    "low": float(bar.l),
                    "close": float(bar.c),
                    "volume": float(bar.v),
                }
            return

        for symbol in self.crypto_symbols:
            try:
                bar = self.api.get_latest_bar(symbol)
            except Exception as exc:
                print(f"[feed] crypto latest bar failed for {symbol}: {exc}")
                continue
            yield {
                "symbol": symbol,
                "timestamp": bar.t.isoformat(),
                "open": float(bar.o),
                "high": float(bar.h),
                "low": float(bar.l),
                "close": float(bar.c),
                "volume": float(bar.v),
            }

    def sleep(self, seconds: int) -> None:
        time.sleep(seconds)
