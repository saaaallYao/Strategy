"""Strategy Base Class - Foundation for all trading strategies"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod
import pytz


class Signal:
    """Trading signal container"""
    
    def __init__(self, signal_id: str, symbol: str, action: str, quantity: int, 
                 price: float, timestamp: datetime, reason: str):
        self.signal_id = signal_id
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.reason = reason
        
    def to_dict(self) -> Dict:
        return {
            'id': self.signal_id,
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'reason': self.reason,
            'source': 'AT'
        }
        
    def __str__(self):
        return f"Signal({self.signal_id}, {self.symbol}, {self.action}, {self.quantity}, ${self.price:.2f})"


class StrategyBase(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.initialized = False
        self.symbols = config.get('symbols', [])  # Stock symbols this strategy monitors
        self.performance_metrics = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'total_pnl': 0.0
        }
        
        # 策略状态
        self.signal_count = 0
        self.current_positions = {}
        self.signal_history = []
        self.trade_history = []
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize strategy parameters"""
        self.initialized = True
    
    @abstractmethod
    def process_data(self, symbol: str, data: Dict) -> Optional[Signal]:
        """Process market data and generate signals"""
        pass
    
    @abstractmethod
    def update_parameters(self, params: Dict) -> None:
        """Update strategy parameters"""
        self.config.update(params)
    
    def get_symbols(self) -> List[str]:
        """Get the symbols this strategy monitors"""
        return self.symbols.copy()
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to monitor"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from monitoring"""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': self.name,
            'initialized': self.initialized,
            'symbols': self.symbols,
            'config': self.config
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        return self.performance_metrics.copy()
    
    def get_status(self) -> Dict:
        """Get strategy status"""
        return {
            'name': self.name,
            'initialized': self.initialized,
            'signal_count': self.signal_count,
            'current_positions': self.current_positions,
            'total_trades': len(self.trade_history)
        }
    
    def _get_eastern_time(self) -> datetime:
        """Get Eastern time"""
        eastern = pytz.timezone('US/Eastern')
        return datetime.now(eastern)
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.initialized = False
        self.performance_metrics = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'total_pnl': 0.0
        }
        self.signal_count = 0
        self.current_positions = {}
        self.signal_history = []
        self.trade_history = []