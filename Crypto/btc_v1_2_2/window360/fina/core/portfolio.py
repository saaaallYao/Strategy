"""Portfolio Management - Position and performance tracking"""

from typing import Dict, List, Optional
from datetime import datetime


class Position:
    """Represents a trading position"""
    
    def __init__(self, symbol: str, quantity: float, entry_price: float, side: str = 'LONG'):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.side = side  # 'LONG' or 'SHORT'
        self.entry_time = datetime.now()
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
    
    def update_price(self, current_price: float) -> None:
        """Update position with current market price"""
        self.current_price = current_price
        if self.side == 'LONG':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def close_position(self, exit_price: float) -> float:
        """Close position and return realized P&L"""
        if self.side == 'LONG':
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
        
        return self.realized_pnl
    
    def __str__(self):
        return f"Position({self.symbol}, {self.quantity}, ${self.entry_price:.2f}, {self.side})"


class Portfolio:
    """Portfolio manager for tracking positions and performance"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> Position
        self.cash = initial_capital
        self.total_pnl = 0.0
        self.trade_history = []
    
    def add_position(self, symbol: str, quantity: float, price: float, side: str = 'LONG') -> bool:
        """Add a new position to the portfolio"""
        if symbol in self.positions:
            print(f"Position already exists for {symbol}")
            return False
        
        # Check if we have enough cash
        required_cash = quantity * price
        if required_cash > self.cash:
            print(f"Insufficient cash. Required: ${required_cash:.2f}, Available: ${self.cash:.2f}")
            return False
        
        # Create new position
        position = Position(symbol, quantity, price, side)
        self.positions[symbol] = position
        self.cash -= required_cash
        
        # Record trade
        trade = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'side': side,
            'timestamp': datetime.now(),
            'type': 'OPEN'
        }
        self.trade_history.append(trade)
        
        print(f"Opened {side} position: {position}")
        return True
    
    def close_position(self, symbol: str, price: float) -> bool:
        """Close an existing position"""
        if symbol not in self.positions:
            print(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        realized_pnl = position.close_position(price)
        
        # Update cash
        if position.side == 'LONG':
            self.cash += position.quantity * price
        else:  # SHORT
            self.cash += (position.entry_price * position.quantity) + realized_pnl
        
        # Update total P&L
        self.total_pnl += realized_pnl
        
        # Record trade
        trade = {
            'symbol': symbol,
            'quantity': position.quantity,
            'price': price,
            'side': 'CLOSE',
            'timestamp': datetime.now(),
            'type': 'CLOSE',
            'realized_pnl': realized_pnl
        }
        self.trade_history.append(trade)
        
        print(f"Closed position: {position}, Realized P&L: ${realized_pnl:.2f}")
        
        # Remove position
        del self.positions[symbol]
        return True
    
    def update_positions(self, market_data: Dict[str, float]) -> None:
        """Update all positions with current market prices"""
        total_unrealized_pnl = 0.0
        
        for symbol, price in market_data.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position.update_price(price)
                total_unrealized_pnl += position.unrealized_pnl
        
        # Update current capital
        self.current_capital = self.cash + total_unrealized_pnl
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = self.total_pnl
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'cash': self.cash,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'num_positions': len(self.positions),
            'positions': {symbol: {
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'side': pos.side
            } for symbol, pos in self.positions.items()}
        }
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.trade_history.copy()