"""Main Entry Point - Simple example of using the trading system"""

from typing import Dict
from core.engine import TradingEngine
from core.data import DataProvider
from core.portfolio import Portfolio
from strategies.zscore import ZScoreStrategy
from log_utils.logger import Logger
from config import Config


def main():
    """Main entry point for the trading system"""
    
    # Load configuration
    config = Config()
    
    # Initialize components
    logger = Logger()
    data_provider = DataProvider(config.get_data_config())
    portfolio = Portfolio(initial_capital=100000)
    
    # Create trading engine
    engine = TradingEngine(config.get_api_config())
    
    # Add strategies with different symbols
    # Strategy 1: Monitor SPY, QQQ, IWM
    zscore_strategy_1 = ZScoreStrategy("zscore_large_cap", config.get_strategy_config("zscore"))
    engine.add_strategy(zscore_strategy_1)
    
    # Strategy 2: Monitor different symbols (example)
    # You can create another strategy with different symbols
    # zscore_strategy_2 = ZScoreStrategy("zscore_tech", {
    #     'symbols': ['AAPL', 'GOOGL', 'MSFT'],
    #     'lookback_period': 15,
    #     'entry_threshold': 1.5,
    #     'exit_threshold': 0.3
    # })
    # engine.add_strategy(zscore_strategy_2)
    
    # Start trading
    try:
        engine.start()
        print("Trading system started successfully")
    except KeyboardInterrupt:
        print("Shutting down trading system...")
    finally:
        engine.stop()


def run_backtest():
    """Run a simple backtest"""
    pass


def run_paper_trading():
    """Run paper trading mode"""
    pass


def run_live_trading():
    """Run live trading mode"""
    pass


if __name__ == "__main__":
    main()