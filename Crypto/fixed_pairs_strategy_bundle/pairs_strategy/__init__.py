"""
Unified entrypoint for shared signal, backtest, and paper-trading helpers that
wrap the existing fixed pairs mean-reversion strategy.
"""

from .signal import LiveStrategyState, PairsSignalEngine

__all__ = ["LiveStrategyState", "PairsSignalEngine"]
