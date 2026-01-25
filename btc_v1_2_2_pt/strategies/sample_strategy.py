from datetime import datetime


class Strategy:
    def __init__(self, config: dict):
        symbols = config.get("symbols", [])
        self.crypto_symbols = [s for s in symbols if "/" in s]
        if not self.crypto_symbols:
            self.crypto_symbols = ["BTC/USD"]
        self.stage = {s: "need_buy" for s in self.crypto_symbols}
        self.last_action_minute = {s: None for s in self.crypto_symbols}

    def _minute_key(self, ts: str):
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.replace(second=0, microsecond=0)

    def on_bar(self, bar: dict, state: dict) -> dict:
        symbol = bar["symbol"]
        if symbol not in self.crypto_symbols:
            return {"action": "hold", "qty": 0, "reason": "non_crypto_symbol"}

        minute_key = self._minute_key(bar["timestamp"])
        if self.last_action_minute.get(symbol) == minute_key:
            return {"action": "hold", "qty": 0, "reason": "same_minute"}

        stage = self.stage.get(symbol, "need_buy")
        if stage == "need_buy":
            self.stage[symbol] = "need_sell"
            self.last_action_minute[symbol] = minute_key
            return {"action": "buy", "qty": 0.001, "reason": "first_buy"}

        position = float(state.get("positions", {}).get(symbol, 0) or 0)
        if stage == "need_sell" and position > 0:
            self.stage[symbol] = "done"
            self.last_action_minute[symbol] = minute_key
            return {"action": "sell", "qty": 0.001, "reason": "first_sell"}

        return {"action": "hold", "qty": 0, "reason": "done"}
