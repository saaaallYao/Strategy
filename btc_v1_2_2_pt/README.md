# Trading Agent (paper trading + dashboard)

一个轻量交易 Agent：读取 Alpaca 实时 1 分钟数据，执行策略插件，记录 logs，并用 Dash 展示指标/图表。

## 核心文件

### 1. 引擎
- **`agent/engine.py`** - 交易引擎与日志写入
  - `PaperEngine` - 主循环（拉行情、跑策略、记录 logs）
  - `--mode offline|online` - 离线/在线切换

### 2. 数据源
- **`agent/feed.py`** - Alpaca 实时数据拉取
  - `AlpacaBarFeed` - 轮询 1min bars

### 3. 策略
- **`strategies/`** - 策略插件目录
  - `sample_strategy.py` - 示例策略
  - `window360_strategy.py` - BTC/ETH 统计套利（基于 fina/window360）

### 4. 仪表盘
- **`dashboard/`** - Dash Web Dashboard
  - `dashboard/app.py` - 图表与指标展示

## 快速开始

### 1. 环境准备
```bash
cd /Users/chenxiyao/Downloads/PTagent/trading-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 设置 Alpaca API
```bash
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

### 3. 运行 Agent
```bash
# Offline：只读实时数据，不下单（默认）
python -m agent.engine \
  --strategy /Users/chenxiyao/Downloads/PTagent/trading-agent/strategies/sample_strategy.py \
  --mode offline

# Online：下 Alpaca paper 订单
python -m agent.engine \
  --strategy /Users/chenxiyao/Downloads/PTagent/trading-agent/strategies/sample_strategy.py \
  --mode online
```

### 4. 启动 Dashboard
```bash
python -m dashboard.app
```
打开：`http://127.0.0.1:8050`

## 策略接口

策略文件必须提供 `Strategy` 类，包含：

- `__init__(self, config: dict)`
- `on_bar(self, bar: dict, state: dict) -> dict`

### 示例
```python
class Strategy:
    def __init__(self, config: dict):
        self.symbols = config.get("symbols", [])

    def on_bar(self, bar: dict, state: dict) -> dict:
        return {"action": "hold", "qty": 0, "reason": "no_signal"}
```

### 输入 bar 格式
```python
{
  "symbol": "BTC/USD",
  "timestamp": "2026-01-21T23:04:00+00:00",
  "open": 43210.12,
  "high": 43230.55,
  "low": 43180.01,
  "close": 43205.88,
  "volume": 123.45
}
```

### 输入 state 格式
```python
{
  "cash": 100000.0,
  "positions": {
    "BTC/USD": 0.002,
    "ETH/USD": 0.0
  }
}
```

### 输出 decision 格式
```python
{
  "action": "buy",  # buy | sell | hold
  "qty": 0.001,
  "reason": "my_signal"
}
```

## 常用命令

### 跑 BTC/ETH window360 策略
```bash
python -m agent.engine \
  --strategy /Users/chenxiyao/Downloads/PTagent/trading-agent/strategies/window360_strategy.py \
  --symbols BTC/USD,ETH/USD \
  --mode offline
```

### 只跑一次（建议调试时）
```bash
python -m agent.engine \
  --strategy /Users/chenxiyao/Downloads/PTagent/trading-agent/strategies/window360_strategy.py \
  --symbols BTC/USD,ETH/USD \
  --mode offline \
  --poll-interval 60
```

## 模式说明

- **offline**：读实时数据，不下单；适合策略验证 + dashboard。
- **online**：下 Alpaca paper 订单；适合模拟真实交易。

## 输出文件（logs）

`logs/` 下会生成：
- `prices.csv` - 价格数据
- `signals.csv` - 策略信号
- `trades.csv` - 交易记录
- `equity.csv` - 账户曲线

Dashboard 直接读取这些文件。
