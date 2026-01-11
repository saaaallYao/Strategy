# BTC/ETH 统计套利（费率适配版）

本目录为可独立运行的策略包（含实时/回测）。已按 fee=5e-4 做过对比与筛选，并保留可配置项用于后续优化。

## 目录结构
- `fina/strategies/crypto/btc_eth_v1/live_monitor.py` 实盘/纸面执行逻辑
- `fina/strategies/crypto/btc_eth_v1/run_strategy_bar_by_bar.py` 逐 bar 回测
- `fina/strategies/crypto/btc_eth_v1/strategy_core.py` 核心逻辑
- `fina/strategies/crypto/btc_eth_v1/strategy_engine.py` 包装层
- `run_paper_trading.py` 纸面运行入口（Alpaca 数据）

## 策略原理（简要）
1) 滚动回归：BTC = alpha + beta * ETH（默认 720 分钟窗口）。  
2) 残差 = BTC − (alpha + beta * ETH)。  
3) 对残差做 Z-score。  
4) 信号：
   - Z < −z_enter → 做多 BTC
   - Z > z_enter → 做空 BTC
   - |Z| < z_exit → 平仓
5) 仓位按残差波动率缩放并做上限限制（inv_cap）。  
6) 价格止损（stop_loss_pct）。  
7) 费率过滤：`min_edge_return` 低于阈值不入场。  

## 推荐参数（fee=5e-4）
- `min_edge_return=0.0014`
- `stop_loss_pct=0.010`
- `z_enter=1.2`, `z_exit=0.4`
- `signal_persistence=3`, `cooldown_bars=30`, `min_hold_bars=30`

## 运行方式
### 纸面交易（Alpaca 数据）
```bash
cd /Users/chenxiyao/Downloads/fina/btc_v1_2_1
python run_paper_trading.py
```

参数覆盖示例：
```bash
python run_paper_trading.py --fee 0.0005 --min-edge 0.0014 --stop-loss 0.010
```

单次执行：
```bash
python run_paper_trading.py --once
```

### 逐 bar 回测
```bash
cd /Users/chenxiyao/Downloads/fina/btc_v1_2_1
python -m fina.strategies.crypto.btc_eth_v1.run_strategy_bar_by_bar
```

## 与原版的区别
- 已在 live 与 bar-by-bar 中加入费率扣除。
- 提供 `min_edge_return` 等入场过滤（默认关闭）。 
- 选用 fee=5e-4 条件下更稳的止损配置（1%）。
- 本精简包已移除 ETF 执行模块（仅数据+信号/回测）。
