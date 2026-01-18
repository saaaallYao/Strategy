# BTC v1.2.2 Bundle

This folder contains two runnable strategy bundles:

- `window720/` (window=720)
- `window360/` (optimized window=360)

Both subfolders are self-contained and can run backtests and paper trading on their own.

## How to Run

Window 720:
```
cd /Users/chenxiyao/Downloads/fina/btc_v1_2_2/window720
export ALPACA_API_KEY="YOUR_KEY"
export ALPACA_SECRET_KEY="YOUR_SECRET"
python run_paper_trading.py
```

Window 360:
```
cd /Users/chenxiyao/Downloads/fina/btc_v1_2_2/window360
export ALPACA_API_KEY="YOUR_KEY"
export ALPACA_SECRET_KEY="YOUR_SECRET"
python run_paper_trading.py
```

Backtests and optimization scripts are under each subfolder's `scripts/` directory.

## Comparison

Recent weekly comparison artifacts live in:

- `comparison/`
