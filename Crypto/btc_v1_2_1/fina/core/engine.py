"""Trading Engine - Core orchestrator for trading operations"""

from typing import Dict, List
from datetime import datetime
import random
from fina.log_utils import Logger


class TradingEngine:
    """Main trading engine that coordinates data, strategies, and portfolio"""

    def __init__(self, config: Dict):
        self.config = config
        self.strategies = {}
        self.running = False
        self.portfolio = None
        self.data_provider = None
        self.logger: Logger = Logger()

    def start(self) -> None:
        """Start the trading engine"""
        self.logger.info("Starting trading engine...")
        self.running = True

        # Get all symbols from registered strategies
        symbols = self.get_all_symbols()
        if not symbols:
            self.logger.error(
                "No symbols found in any strategy. Please add strategies with symbols first."
            )
            return

        self.logger.info(f"Processing market data for all symbols...")

        # TODO: Finate State Machine for market data, heartbeat, etc.
        # Simulate market data for each symbol
        for symbol in symbols:
            # Generate simulated market data
            market_data = {
                "symbol": symbol,
                "price": random.uniform(100, 200),
                "volume": random.randint(1000, 10000),
                "timestamp": datetime.now(),
            }

            # Process data through all strategies
            self.process_market_data(symbol, market_data)

        self.logger.info("Trading engine simulation completed!")

    def get_all_symbols(self) -> List[str]:
        """Get all unique symbols from all registered strategies"""
        all_symbols = set()
        for strategy in self.strategies.values():
            all_symbols.update(strategy.get_symbols())
        return list(all_symbols)

    def stop(self) -> None:
        """Stop the trading engine"""
        self.logger.warning("Stopping trading engine...")
        self.running = False

    def add_strategy(self, strategy) -> None:
        """Add a trading strategy to the engine"""
        self.strategies[strategy.name] = strategy
        self.logger.info(
            f"Added strategy: {strategy.name} with symbols: {strategy.get_symbols()}"
        )

    def remove_strategy(self, strategy_id: str) -> None:
        """Remove a trading strategy from the engine"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.logger.warning(f"Removed strategy: {strategy_id}")

    def process_market_data(self, symbol: str, data: Dict) -> None:
        """Process incoming market data"""
        self.logger.info(f"Processing data for {symbol}: price=${data['price']:.2f}")

        for strategy_name, strategy in self.strategies.items():
            # Only process data for symbols that this strategy monitors
            if symbol in strategy.get_symbols():
                signal = strategy.process_data(symbol, data)
                if signal:
                    quantity = 100
                    result = self.execute_trade(
                        symbol, quantity, signal.signal_type.lower()
                    )
                    self.logger.track(
                        f"Strategy {strategy_name} generated signal: {signal.signal_type} for {symbol}",
                        data={
                            "algorithm": strategy_name,
                            "data": {
                                "symbol": symbol,
                                "signal": signal.signal_type,
                                "quantity": quantity,
                                "result": result,
                            },
                        },
                    )

    def execute_trade(self, symbol: str, quantity: float, side: str) -> bool:
        """Execute a trade order"""
        self.logger.info(
            f"Executing trade: {side.upper()} {quantity} shares of {symbol}"
        )

        # In a real system, this would interface with a broker
        # For prototype, just log the trade
        trade_info = {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "timestamp": datetime.now(),
            "status": "FILLED",
        }
        self.logger.info(f"Trade executed: {trade_info}", data={"trade": trade_info})

        return True

    def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            "running": self.running,
            "strategies": list(self.strategies.keys()),
            "symbols": self.get_all_symbols(),
            "config": self.config,
        }
