#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Engine for BTC-ETH Statistical Arbitrage
- Wrapper around the shared strategy core
- Maintains backward compatibility for live trading
"""

from .strategy_core import StatArbStrategyCore


class StatArbEngine(StatArbStrategyCore):
    """
    Statistical Arbitrage Strategy Engine
    
    This is now a simple wrapper around StatArbStrategyCore to maintain
    backward compatibility while ensuring all logic comes from the shared core.
    """
    
    def __init__(self, config):
        # Initialize the shared core
        super().__init__(config)
        print(f"[ENGINE] StatArbEngine wrapper initialized - using shared strategy core") 