# @command: python3 s_btc_eth_v1.py
# @command: python /Users/zhihaoouyang/Desktop/code/when2buy/github/fina/fina/strategies/s_btc_eth_v1.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC-ETH Trading Strategy V1 - 简化版本
- 每10秒随机买入BTC或ETH
- 下一个10秒卖出持仓
- 支持interval同步和模拟模式
"""

import time
import random
import json
from datetime import datetime
from typing import Dict, List

# 使用包导入
from fina.core.data import SimpleDataProvider
from fina.consts import DATA_PATH


class SimpleSignal:
    """简单的交易信号"""
    
    def __init__(self, action: str, symbol: str, quantity: float, price: float):
        self.action = action
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.timestamp = datetime.now()
    
    def __str__(self):
        return f"{self.action} {self.quantity:.4f} {self.symbol} @ ${self.price:.2f}"


class BTCETHStrategy:
    """BTC-ETH 随机交易策略 - 简化版"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 简化配置
        self.symbols = ['BTC-USD', 'ETH-USD']
        self.trade_interval = 10  # 10秒交易一次
        self.position_size = 1000  # 每次交易1000美元
        
        # 数据提供器
        self.data_provider = SimpleDataProvider(config)
        
        # 交易状态
        self.positions = {'BTC-USD': 0, 'ETH-USD': 0}
        self.last_trade_time = 0
        self.trade_history = []
        
        print(f"[STRATEGY] 简化版BTC-ETH策略已初始化")
        print(f"[STRATEGY] 交易间隔: {self.trade_interval}秒")
    
    def should_trade(self) -> bool:
        """检查是否应该交易"""
        current_time = time.time()
        
        # 检查是否到了交易时间
        if current_time - self.last_trade_time >= self.trade_interval:
            return True
        
        return False
    
    def generate_signal(self) -> SimpleSignal:
        """生成交易信号"""
        if not self.should_trade():
            return SimpleSignal('HOLD', '', 0, 0)
        
        # 随机选择交易
        symbol = random.choice(self.symbols)
        action = random.choice(['BUY', 'SELL'])
        
        # 获取当前价格
        current_price = self.data_provider.get_current_price(symbol)
        
        # 计算交易数量
        quantity = self.position_size / current_price
        
        # 更新交易时间
        self.last_trade_time = time.time()
        
        return SimpleSignal(action, symbol, quantity, current_price)
    
    def execute_signal(self, signal: SimpleSignal) -> bool:
        """执行交易信号"""
        if signal.action == 'HOLD':
            return True
        
        # 更新持仓
        if signal.action == 'BUY':
            self.positions[signal.symbol] += signal.quantity
        elif signal.action == 'SELL':
            self.positions[signal.symbol] -= signal.quantity
        
        # 记录交易
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'action': signal.action,
            'symbol': signal.symbol,
            'quantity': signal.quantity,
            'price': signal.price,
            'position': self.positions[signal.symbol]
        }
        
        self.trade_history.append(trade_record)
        
        print(f"[TRADE] {signal.action} {signal.quantity:.4f} {signal.symbol} @ {signal.price:.2f}")
        print(f"[POSITION] {signal.symbol}: {self.positions[signal.symbol]:.4f}")
        
        return True
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        if not self.trade_history:
            return {'total_trades': 0, 'total_pnl': 0}
        
        total_trades = len(self.trade_history)
        
        # 计算总PnL (简化版)
        total_pnl = 0
        for trade in self.trade_history:
            if trade['action'] == 'BUY':
                total_pnl -= trade['quantity'] * trade['price']
            else:
                total_pnl += trade['quantity'] * trade['price']
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'positions': self.positions.copy()
        }
    
    def save_trade_log(self):
        """保存交易记录"""
        if not self.trade_history:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = DATA_PATH / "trade_log" /  f"btc_eth_v1_log_{timestamp}" / "trade_log.json"
        
        log_data = {
            'strategy': 'btc_eth_v1_simple',
            'config': self.config,
            'performance': self.get_performance_metrics(),
            'trades': self.trade_history
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"[LOG] 交易记录已保存到 {filename}")


def run_strategy_test():
    """运行策略测试"""
    print("=" * 50)
    print("开始测试简化版BTC-ETH策略")
    print("=" * 50)
    
    # 简化配置
    config = {
        'symbols': ['BTC-USD', 'ETH-USD'],
        'interval': '5min'
    }
    
    # 创建策略
    strategy = BTCETHStrategy(config)
    
    # 运行30秒测试
    start_time = time.time()
    test_duration = 30
    
    print(f"运行{test_duration}秒测试...")
    
    while time.time() - start_time < test_duration:
        # 生成信号
        signal = strategy.generate_signal()
        
        # 执行信号
        strategy.execute_signal(signal)
        
        # 等待1秒
        time.sleep(1)
    
    # 显示结果
    print("\n" + "=" * 30)
    print("测试结果:")
    print("=" * 30)
    
    metrics = strategy.get_performance_metrics()
    print(f"总交易次数: {metrics['total_trades']}")
    print(f"总PnL: ${metrics['total_pnl']:.2f}")
    print(f"最终持仓: {metrics['positions']}")
    
    # 保存交易记录
    strategy.save_trade_log()
    
    print("\n测试完成!")


if __name__ == "__main__":
    run_strategy_test()
