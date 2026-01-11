"""Z-Score Strategy - Mean reversion trading strategy"""

from typing import Dict, List, Optional
from datetime import datetime
import random
from .strategy import StrategyBase, Signal


class ZScoreStrategy(StrategyBase):
    """Z-Score based mean reversion strategy"""
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.price_history = {}
        self.rolling_mean = {}
        self.rolling_std = {}
        self.lookback_period = config.get('lookback_period', 20)
        self.entry_threshold = config.get('entry_threshold', 2.0)
        self.exit_threshold = config.get('exit_threshold', 0.5)
    
    def initialize(self) -> None:
        """Initialize Z-Score strategy parameters"""
        super().initialize()
        print(f"Z-Score strategy '{self.name}' initialized with lookback={self.lookback_period}, entry_threshold={self.entry_threshold}")
    
    def process_data(self, symbol: str, data: Dict) -> Optional[Signal]:
        """Process market data and generate Z-Score signals"""
        if not self.initialized:
            self.initialize()
        
        # For prototype, simulate price data
        current_price = data.get('price', random.uniform(100, 200))
        
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(current_price)
        
        # Keep only recent prices
        if len(self.price_history[symbol]) > self.lookback_period:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_period:]
        
        # Calculate Z-Score
        zscore = self.calculate_zscore(symbol, current_price)
        
        # Generate signal based on Z-Score
        signal = None
        if zscore > self.entry_threshold:
            signal = Signal(symbol, 'SELL', zscore, datetime.now())
            self.performance_metrics['sell_signals'] += 1
        elif zscore < -self.entry_threshold:
            signal = Signal(symbol, 'BUY', abs(zscore), datetime.now())
            self.performance_metrics['buy_signals'] += 1
        
        if signal:
            self.performance_metrics['total_signals'] += 1
            print(f"Generated signal: {signal}")
        
        return signal
    
    def update_parameters(self, params: Dict) -> None:
        """Update Z-Score strategy parameters"""
        super().update_parameters(params)
        if 'lookback_period' in params:
            self.lookback_period = params['lookback_period']
        if 'entry_threshold' in params:
            self.entry_threshold = params['entry_threshold']
        if 'exit_threshold' in params:
            self.exit_threshold = params['exit_threshold']
    
    def calculate_zscore(self, symbol: str, current_price: float) -> float:
        """Calculate Z-Score for current price"""
        prices = self.price_history.get(symbol, [])
        if len(prices) < 2:
            return 0.0
        
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_price = variance ** 0.5
        
        if std_price == 0:
            return 0.0
        
        return (current_price - mean_price) / std_price
    
    def update_rolling_stats(self, symbol: str, price: float) -> None:
        """Update rolling mean and standard deviation"""
        if symbol not in self.rolling_mean:
            self.rolling_mean[symbol] = price
            self.rolling_std[symbol] = 0.0
        else:
            # Simple exponential moving average for prototype
            alpha = 0.1
            self.rolling_mean[symbol] = alpha * price + (1 - alpha) * self.rolling_mean[symbol]
    
    def get_entry_threshold(self) -> float:
        """Get entry threshold for Z-Score"""
        return self.entry_threshold
    
    def get_exit_threshold(self) -> float:
        """Get exit threshold for Z-Score"""
        return self.exit_threshold
    
    def should_enter_position(self, zscore: float) -> bool:
        """Check if should enter position based on Z-Score"""
        return abs(zscore) > self.entry_threshold
    
    def should_exit_position(self, zscore: float) -> bool:
        """Check if should exit position based on Z-Score"""
        return abs(zscore) < self.exit_threshold