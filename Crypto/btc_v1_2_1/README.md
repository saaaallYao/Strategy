# Fina - 量化交易框架

一个简洁的量化交易框架，专注于加密货币统计套利策略。

## 项目结构

```
fina/
├── fina/                          # 核心包
│   ├── core/                      # 核心功能
│   │   ├── data.py               # 数据处理
│   │   ├── engine.py             # 交易引擎
│   │   ├── portfolio.py          # 组合管理
│   │   └── trading_api.py        # 交易API
│   ├── strategies/               # 策略模块
│   │   └── crypto/
│   │       └── btc_eth_v1/       # BTC-ETH统计套利策略
│   │           ├── data_manager.py      # 数据管理
│   │           ├── strategy_engine.py   # 策略引擎
│   │           ├── run_strategy.py      # 策略运行
│   │           ├── run_strategy_bar_by_bar.py  # 逐bar执行
│   │           ├── analyze_bar_by_bar_log.py   # 结果分析
│   │           └── README.md            # 策略文档
│   └── utils/                    # 工具函数
├── data/                         # 数据目录
│   ├── cache/                    # 缓存数据
│   └── logs/                     # 日志文件
├── scripts/                      # 脚本文件
└── tests/                        # 测试文件
```

## 核心组件

### 1. 数据管理 (`fina/core/data.py`)
- 统一的数据接口
- 多数据源支持
- 数据缓存机制

### 2. 交易引擎 (`fina/core/engine.py`)
- 策略执行框架
- 风险管理
- 订单管理

### 3. 策略模块 (`fina/strategies/`)
- 模块化策略设计
- 可扩展架构
- 标准化接口

## 主要策略

### BTC-ETH 统计套利策略
位置：`fina/strategies/crypto/btc_eth_v1/`

**核心特性：**
- 基于BTC/ETH/SOL的统计套利
- 滚动OLS回归分析
- Z-score信号生成
- 动态仓位管理
- 实时数据更新

**快速开始：**
```bash
# 设置环境变量
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"

# 启动实时监控（一行命令）
./start_monitor.sh

# 查看状态
./check_status.sh

# 运行策略
python -m fina.strategies.crypto.btc_eth_v1.run_strategy_bar_by_bar

# 分析结果
python -m fina.strategies.crypto.btc_eth_v1.analyze_bar_by_bar_log
```

详细文档：[策略README](fina/strategies/crypto/btc_eth_v1/README.md)

## 安装

```bash
# 克隆仓库
git clone <repository_url>
cd fina

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
```

## 依赖

- `alpaca-py` - Alpaca交易API
- `pandas` - 数据处理
- `numpy` - 数值计算
- `matplotlib` - 图表绘制
- `pytz` - 时区处理

## 使用示例

```python
# 运行BTC-ETH策略
from fina.strategies.crypto.btc_eth_v1.run_strategy_bar_by_bar import run_bar_by_bar

# 执行策略
run_bar_by_bar('2025-07-01', '2025-07-05')

# 分析结果
from fina.strategies.crypto.btc_eth_v1.analyze_bar_by_bar_log import analyze_log
analyze_log('data/logs/btc_eth_bar_by_bar_*.json')
```

## 开发

```bash
# 运行测试
python -m pytest tests/

# 代码格式化
black fina/
```

## 许可证

MIT License
