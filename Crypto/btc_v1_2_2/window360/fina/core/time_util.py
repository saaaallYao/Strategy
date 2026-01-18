"""Time Utilities - Market hours and timezone handling"""

from datetime import datetime, time, timedelta
from typing import Optional
import pytz


class TimeUtil:
    """Utility class for time and market hours operations"""
    
    # Market types
    CRYPTO = "crypto"
    STOCK = "stock"
    
    @staticmethod
    def get_market_timezone():
        """Get market timezone (Eastern Time)"""
        return pytz.timezone('US/Eastern')
    
    @staticmethod
    def convert_to_market_time(dt: datetime) -> datetime:
        """Convert datetime to market timezone"""
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(TimeUtil.get_market_timezone())
    
    @staticmethod
    def convert_to_utc(dt: datetime) -> datetime:
        """Convert datetime to UTC"""
        if dt.tzinfo is None:
            dt = TimeUtil.get_market_timezone().localize(dt)
        return dt.astimezone(pytz.UTC)
    
    @staticmethod
    def is_market_open(market_type: str = CRYPTO, dt: Optional[datetime] = None) -> bool:
        """Check if market is open at given time
        
        Args:
            market_type: "crypto" or "stock"
            dt: datetime to check, defaults to current time
        """
        if dt is None:
            dt = TimeUtil.now_market_time()
        
        if market_type == TimeUtil.CRYPTO:
            return True  # Crypto markets are always open
        
        elif market_type == TimeUtil.STOCK:
            # Stock market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
            if not TimeUtil.is_trading_day(market_type, dt):
                return False
            
            current_time = dt.time()
            market_open = time(9, 30)
            market_close = time(16, 0)
            
            return market_open <= current_time <= market_close
        
        else:
            raise ValueError(f"Unknown market type: {market_type}")
    
    @staticmethod
    def get_market_open_time(market_type: str = CRYPTO) -> time:
        """Get market opening time
        
        Args:
            market_type: "crypto" or "stock"
        """
        if market_type == TimeUtil.CRYPTO:
            return time(0, 0)  # 24/7
        elif market_type == TimeUtil.STOCK:
            return time(9, 30)  # 9:30 AM ET
        else:
            raise ValueError(f"Unknown market type: {market_type}")
    
    @staticmethod
    def get_market_close_time(market_type: str = CRYPTO) -> time:
        """Get market closing time
        
        Args:
            market_type: "crypto" or "stock"
        """
        if market_type == TimeUtil.CRYPTO:
            return time(23, 59)  # 24/7
        elif market_type == TimeUtil.STOCK:
            return time(16, 0)  # 4:00 PM ET
        else:
            raise ValueError(f"Unknown market type: {market_type}")
    
    @staticmethod
    def is_trading_day(market_type: str = CRYPTO, dt: Optional[datetime] = None) -> bool:
        """Check if given date is a trading day
        
        Args:
            market_type: "crypto" or "stock"
            dt: datetime to check, defaults to current time
        """
        if dt is None:
            dt = TimeUtil.now_market_time()
        
        if market_type == TimeUtil.CRYPTO:
            return True  # Crypto markets trade every day
        
        elif market_type == TimeUtil.STOCK:
            # Stock markets are closed on weekends
            return dt.weekday() < 5  # Monday = 0, Sunday = 6
        
        else:
            raise ValueError(f"Unknown market type: {market_type}")
    
    @staticmethod
    def get_next_trading_day(market_type: str = CRYPTO, dt: Optional[datetime] = None) -> datetime:
        """Get next trading day
        
        Args:
            market_type: "crypto" or "stock"
            dt: reference datetime, defaults to current time
        """
        if dt is None:
            dt = TimeUtil.now_market_time()
        
        if market_type == TimeUtil.CRYPTO:
            return dt + timedelta(days=1)
        
        elif market_type == TimeUtil.STOCK:
            next_day = dt + timedelta(days=1)
            # Skip weekends
            while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
                next_day += timedelta(days=1)
            return next_day
        
        else:
            raise ValueError(f"Unknown market type: {market_type}")
    
    @staticmethod
    def get_previous_trading_day(market_type: str = CRYPTO, dt: Optional[datetime] = None) -> datetime:
        """Get previous trading day
        
        Args:
            market_type: "crypto" or "stock"
            dt: reference datetime, defaults to current time
        """
        if dt is None:
            dt = TimeUtil.now_market_time()
        
        if market_type == TimeUtil.CRYPTO:
            return dt - timedelta(days=1)
        
        elif market_type == TimeUtil.STOCK:
            prev_day = dt - timedelta(days=1)
            # Skip weekends
            while prev_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
                prev_day -= timedelta(days=1)
            return prev_day
        
        else:
            raise ValueError(f"Unknown market type: {market_type}")
    
    @staticmethod
    def now_market_time() -> datetime:
        """Get current time in market timezone"""
        return datetime.now(TimeUtil.get_market_timezone())
    
    @staticmethod
    def today_market_date() -> str:
        """Get today's date in market timezone as YYYY-MM-DD string"""
        return TimeUtil.now_market_time().strftime("%Y-%m-%d")
    
    @staticmethod
    def days_ago_market_date(days: int) -> str:
        """Get date N days ago in market timezone as YYYY-MM-DD string"""
        dt = TimeUtil.now_market_time() - timedelta(days=days)
        return dt.strftime("%Y-%m-%d")
    
    @staticmethod
    def format_market_datetime(dt: datetime) -> str:
        """Format datetime to market timezone string"""
        market_dt = TimeUtil.convert_to_market_time(dt)
        return market_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def test_crypto_market():
    """Test crypto market time functions"""
    print("=== Crypto Market Tests ===")
    
    now = TimeUtil.now_market_time()
    print(f"Current market time: {TimeUtil.format_market_datetime(now)}")
    print(f"Crypto market open: {TimeUtil.is_market_open(TimeUtil.CRYPTO)}")
    print(f"Crypto trading day: {TimeUtil.is_trading_day(TimeUtil.CRYPTO)}")
    print(f"Next crypto trading day: {TimeUtil.format_market_datetime(TimeUtil.get_next_trading_day(TimeUtil.CRYPTO))}")
    print(f"Previous crypto trading day: {TimeUtil.format_market_datetime(TimeUtil.get_previous_trading_day(TimeUtil.CRYPTO))}")


def test_stock_market():
    """Test stock market time functions"""
    print("\n=== Stock Market Tests ===")
    
    now = TimeUtil.now_market_time()
    print(f"Current market time: {TimeUtil.format_market_datetime(now)}")
    print(f"Stock market open: {TimeUtil.is_market_open(TimeUtil.STOCK)}")
    print(f"Stock trading day: {TimeUtil.is_trading_day(TimeUtil.STOCK)}")
    print(f"Next stock trading day: {TimeUtil.format_market_datetime(TimeUtil.get_next_trading_day(TimeUtil.STOCK))}")
    print(f"Previous stock trading day: {TimeUtil.format_market_datetime(TimeUtil.get_previous_trading_day(TimeUtil.STOCK))}")


if __name__ == "__main__":
    test_crypto_market()
    test_stock_market()