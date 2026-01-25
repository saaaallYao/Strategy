import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentConfig:
    symbols: List[str] = field(default_factory=lambda: ["AAPL"])
    timeframe: str = "1Min"
    poll_interval: int = 10
    starting_cash: float = 100000.0
    bars_per_year: int = 252 * 390  # 1-min bars in a trading year
    log_dir: str = os.path.join(os.path.dirname(__file__), "..", "logs")
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
