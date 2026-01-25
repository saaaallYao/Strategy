from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _add_window360_path() -> None:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    window360_root = project_root / "btc_v1_2_2" / "window360"
    if window360_root.exists():
        sys.path.insert(0, str(window360_root))


_add_window360_path()

from fina.strategies.crypto.btc_eth_v1.strategy_engine import StatArbEngine  # noqa: E402


class Strategy:
    """
    Adapter for window360 strategy using the simple Strategy interface.
    Consumes 1-min BTC/ETH bars and emits buy/sell/hold decisions.
    """

    def __init__(self, config: dict):
        symbols = config.get("symbols", [])
        self.symbols = [s for s in symbols if "/" in s]
        if "BTC/USD" not in self.symbols or "ETH/USD" not in self.symbols:
            self.symbols = ["BTC/USD", "ETH/USD"]

        self.engine = StatArbEngine({
            "window": 360,
            "z_enter": 1.2,
            "z_exit": 0.4,
            "signal_persistence": 3,
            "min_hold_bars": 30,
            "cooldown_bars": 30,
            "min_edge_return": 0.0014,
            "dyn_edge_enabled": True,
            "dyn_edge_fee_mult": 5.0,
            "dyn_edge_vol_mult": 0.5,
            "fee_exit_enabled": True,
            "fee_exit_mult": 2.0,
            "stop_loss_pct": 0.01,
            "scale_base": 0.05,
            "inv_cap": 0.15,
            "fee": 5e-4,
            "clip_resid0": 800,
            "clip_beta": 6,
        })

        self.current_pos = 0.0
        self.current_pos_by_symbol: Dict[str, float] = {"BTC/USD": 0.0, "ETH/USD": 0.0}
        self.pending_minute: Optional[pd.Timestamp] = None
        self.pending_prices: Dict[str, float] = {}
        self.last_prices: Dict[str, float] = {}
        self.preload_prices: Optional[pd.DataFrame] = None
        self.desired_pos: Dict[str, float] = {}
        self.last_action_minute: Dict[str, Optional[pd.Timestamp]] = {"BTC/USD": None, "ETH/USD": None}
        self.history = pd.DataFrame(columns=["BTC_USD", "ETH_USD"])

        preload_minutes = int(os.environ.get("WINDOW360_PRELOAD_MINUTES", "2000"))
        if preload_minutes > 0:
            self._preload_history(preload_minutes)

    def _minute_key(self, ts: str) -> pd.Timestamp:
        return pd.to_datetime(ts, errors="coerce").floor("min")

    def _append_history(self, minute_key: pd.Timestamp) -> None:
        row = {
            "BTC_USD": self.pending_prices.get("BTC/USD"),
            "ETH_USD": self.pending_prices.get("ETH/USD"),
        }
        if row["BTC_USD"] is None or row["ETH_USD"] is None:
            return
        self.history.loc[minute_key] = row
        if len(self.history) > 2000:
            self.history = self.history.iloc[-2000:]

    def _preload_history(self, minutes: int) -> None:
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except Exception as exc:
            print(f"[window360] preload skipped (alpaca-py unavailable): {exc}")
            return

        try:
            end_dt = pd.Timestamp.utcnow()
            start_dt = end_dt - pd.Timedelta(minutes=minutes)
            client = CryptoHistoricalDataClient()
            req = CryptoBarsRequest(
                symbol_or_symbols=["BTC/USD", "ETH/USD"],
                timeframe=TimeFrame.Minute,
                start=start_dt,
                end=end_dt,
            )
            bars = client.get_crypto_bars(req).df
            if bars is None or bars.empty:
                print("[window360] preload got empty data")
                return
            df = bars.reset_index()
            if "timestamp" not in df.columns:
                df = df.rename(columns={df.columns[0]: "timestamp"})
            df = df[["timestamp", "symbol", "close"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            pivot = df.pivot_table(
                index="timestamp",
                columns="symbol",
                values="close",
                aggfunc="last",
            ).sort_index()
            if "BTC/USD" in pivot.columns:
                pivot = pivot.rename(columns={"BTC/USD": "BTC_USD"})
            if "ETH/USD" in pivot.columns:
                pivot = pivot.rename(columns={"ETH/USD": "ETH_USD"})
            pivot = pivot[["BTC_USD", "ETH_USD"]].dropna()
            if not pivot.empty:
                self.history = pivot.tail(2000)
                self.preload_prices = df.copy()
                print(f"[window360] preloaded {len(self.history)} bars")
        except Exception as exc:
            print(f"[window360] preload failed: {exc}")

    def _compute_target_pos(self) -> tuple[Optional[float], Optional[float], Optional[str]]:
        px = self.history.copy()
        if len(px) < self.engine.window + 2:
            return None, None, "warmup"
        sub, resid, _ = self.engine.rolling_beta_resid(px)
        if resid.isna().all():
            return None, None, "warmup"
        _, scale_dyn = self.engine.calculate_dynamic_adjustments(resid)
        pos_series, _ = self.engine.generate_signals(resid, scale_dyn, px=sub)
        if pos_series.empty or pd.isna(pos_series.iloc[-1]):
            return None, None, "warmup"
        zscore = None
        try:
            resid_signal = resid
            mu = resid_signal.rolling(self.engine.window).mean()
            sig = resid_signal.rolling(self.engine.window).std()
            z = (resid_signal - mu) / sig
            zscore = float(z.iloc[-1]) if not pd.isna(z.iloc[-1]) else None
        except Exception:
            zscore = None
        blocked_reason = None
        try:
            if zscore is not None:
                z_enter = self.engine.z_enter
                if abs(zscore) < z_enter:
                    blocked_reason = "below_z_enter"
                else:
                    # consecutive counts for persistence
                    z_tail = z.dropna()
                    long_count = 0
                    short_count = 0
                    for val in reversed(z_tail.values.tolist()):
                        if val < -z_enter:
                            long_count += 1
                            short_count = 0
                        elif val > z_enter:
                            short_count += 1
                            long_count = 0
                        else:
                            break
                    # expected return vs edge filter
                    price = float(sub["BTC_USD"].iloc[-1]) if "BTC_USD" in sub.columns else None
                    sig_last = float(sig.iloc[-1]) if not pd.isna(sig.iloc[-1]) else None
                    expected_return = None
                    if price and sig_last:
                        expected_return = abs(zscore) * sig_last / price
                    edge_ok = True
                    if expected_return is not None:
                        if self.engine.dyn_edge_enabled:
                            required_edge = (self.engine.dyn_edge_fee_mult * self.engine.fee) + (
                                self.engine.dyn_edge_vol_mult * (sig_last / price)
                            )
                            edge_ok = expected_return >= required_edge
                        elif self.engine.min_edge_return is not None:
                            edge_ok = expected_return >= self.engine.min_edge_return
                    if not edge_ok:
                        blocked_reason = "edge_filter"
                    elif max(long_count, short_count) < self.engine.signal_persistence:
                        blocked_reason = "persistence"
                    else:
                        blocked_reason = "engine_no_entry"
        except Exception:
            pass
        return float(pos_series.iloc[-1]), zscore, blocked_reason

    def on_bar(self, bar: dict, state: dict) -> dict:
        symbol = bar.get("symbol")
        if symbol not in {"BTC/USD", "ETH/USD"}:
            return {"action": "hold", "qty": 0, "reason": "non_btc_eth", "skip_log": True}

        minute_key = self._minute_key(bar.get("timestamp"))
        if self.pending_minute is None or minute_key != self.pending_minute:
            self.pending_minute = minute_key
            self.pending_prices = {}

        self.pending_prices[symbol] = float(bar.get("close", 0) or 0)
        self.last_prices[symbol] = self.pending_prices[symbol]

        if not all(s in self.pending_prices for s in ("BTC/USD", "ETH/USD")):
            missing = [s for s in ("BTC/USD", "ETH/USD") if s not in self.pending_prices]
            if all(s in self.last_prices for s in missing):
                for s in missing:
                    self.pending_prices[s] = self.last_prices[s]
            else:
                return {"action": "hold", "qty": 0, "reason": "waiting_pair"}

        self._append_history(minute_key)

        target_pos, zscore, blocked_reason = self._compute_target_pos()
        if target_pos is None:
            return {
                "action": "hold",
                "qty": 0,
                "reason": "warmup",
                "zscore": zscore,
                "target_pos": target_pos,
                "blocked_reason": blocked_reason or "warmup",
            }

        # compute hedge ratio (beta) for ETH position sizing
        hedge_beta = 1.0
        try:
            sub, _, beta = self.engine.rolling_beta_resid(self.history)
            if not beta.empty and not pd.isna(beta.iloc[-1]):
                hedge_beta = float(beta.iloc[-1])
        except Exception:
            hedge_beta = 1.0

        self.desired_pos = {
            "BTC/USD": target_pos,
            "ETH/USD": -target_pos * hedge_beta,
        }

        # only act once per symbol per minute
        if self.last_action_minute.get(symbol) == minute_key:
            return {"action": "hold", "qty": 0, "reason": "same_minute", "zscore": zscore, "target_pos": target_pos}

        current_pos = self.current_pos_by_symbol.get(symbol, 0.0)
        desired = self.desired_pos.get(symbol, 0.0)
        delta = desired - current_pos
        if abs(delta) < 1e-6:
            self.last_action_minute[symbol] = minute_key
            return {
                "action": "hold",
                "qty": 0,
                "reason": "no_change",
                "zscore": zscore,
                "target_pos": desired,
                "blocked_reason": blocked_reason or "no_change",
            }

        action = "buy" if delta > 0 else "sell"
        qty = abs(delta)
        self.current_pos_by_symbol[symbol] = desired
        self.last_action_minute[symbol] = minute_key
        return {
            "action": action,
            "qty": qty,
            "reason": "window360_target",
            "zscore": zscore,
            "target_pos": desired,
            "blocked_reason": "",
        }
