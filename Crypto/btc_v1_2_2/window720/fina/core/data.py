"""Data Provider - Simplified market data management"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random
import pytz

class SimpleDataProvider:
    """简化的数据提供器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config.get('symbols', ['BTC-USD', 'ETH-USD'])
        self.interval_minutes = self._parse_interval(config.get('interval', '5min'))
        self.eastern_tz = pytz.timezone('US/Eastern')
        
        # 简单的价格缓存
        self.current_prices = {}
        self.last_update = {}
        
        print(f"[DATA] Simple provider initialized")
        print(f"[DATA] Symbols: {self.symbols}")
        print(f"[DATA] Interval: {self.interval_minutes} minutes")
        
    def _parse_interval(self, interval: str) -> int:
        """解析时间间隔"""
        if interval.endswith('min'):
            return int(interval[:-3])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval == '1d':
            return 24 * 60
        else:
            return 5  # 默认5分钟
    
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        # 生成模拟价格
        if symbol == 'BTC-USD':
            base_price = 45000
        elif symbol == 'ETH-USD':
            base_price = 2500
        else:
            base_price = 100
            
        # 添加随机波动
        price = base_price * (1 + random.uniform(-0.02, 0.02))
        
        # 缓存价格
        self.current_prices[symbol] = price
        self.last_update[symbol] = datetime.now(self.eastern_tz)
        
        return price
    
    def should_wait_for_interval(self) -> bool:
        """检查是否应该等待下一个interval"""
        now = datetime.now(self.eastern_tz)
        
        # 计算当前时间距离下一个interval时间点的分钟数
        minutes_since_midnight = now.hour * 60 + now.minute
        minutes_to_next_interval = self.interval_minutes - (minutes_since_midnight % self.interval_minutes)
        
        # 如果距离下一个interval还有超过1分钟，就等待
        return minutes_to_next_interval > 1
    
    def get_seconds_to_next_interval(self) -> int:
        """获取到下一个interval的秒数"""
        now = datetime.now(self.eastern_tz)
        
        # 计算到下一个interval的时间
        minutes_since_midnight = now.hour * 60 + now.minute
        minutes_to_next_interval = self.interval_minutes - (minutes_since_midnight % self.interval_minutes)
        
        # 转换为秒数，考虑当前的秒数
        total_seconds = minutes_to_next_interval * 60 - now.second
        
        return max(0, total_seconds)
    
    def get_interval_info(self) -> Dict:
        """获取interval信息"""
        now = datetime.now(self.eastern_tz)
        
        return {
            'current_time': now.strftime('%H:%M:%S'),
            'interval_minutes': self.interval_minutes,
            'should_wait': self.should_wait_for_interval(),
            'seconds_to_next': self.get_seconds_to_next_interval()
        }

# 保持兼容性的别名
DataProvider = SimpleDataProvider