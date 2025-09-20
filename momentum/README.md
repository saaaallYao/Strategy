# Momentum (T+1) Strategy

## Overview
This repository contains an intraday momentum strategy for the China A-share market and ETFs. The strategy identifies stocks/ETFs that are strong intraday performers and takes overnight positions to capture continuation effects.

---

## Strategy Description

### Core Concept
- **Momentum Continuation**: Stocks that perform strongly from open to the afternoon cutoff (14:45) tend to continue their trend into the next trading day.
- **Cross-Sectional Selection**: Every day, the top N symbols with the strongest signals are selected for trading.
- **Short Holding Horizon**: Positions are entered at the close (T) and exited at the next close (T+1).

### Signal Generation
1. **Day Open**: Use the first minute price (close of 09:31 candle, as proxy for 09:30 open).  
2. **Cutoff Price**: Last price before 14:45.  
3. **Signal Formula**:  
   ```
   signal = cutoff_close / day_open - 1
   ```
   This represents intraday return from open → 14:45.
4. **Ranking**: Sort all symbols in descending order of signal. Stronger = higher rank.

### Filters
- **Liquidity filter**: Yesterday’s total traded volume ≥ 0.8 × median of the past 20 days.  
（- **Extreme move filter**: Yesterday’s daily return (close/prev_close − 1) < ±8%.  
- Both filters use **yesterday’s data (shifted)** to avoid lookahead bias.）

### Trading Rules
1. Each day, select **Top N (default 5)** symbols that pass filters.  
2. **Buy** at today’s close (T).  
3. **Sell** at next day’s close (T+1).  
4. Portfolio allocation: equally weighted across selected symbols.  
5. **Transaction Costs**: 0.03% per side (0.0003 per trade).  
   - Buy: 0.03%  
   - Sell: 0.03%  
   - Total round trip = 0.06%

---

## Benchmark (Buy & Hold)
The benchmark is constructed as an **equal-weighted daily rebalanced portfolio of all available symbols** in the dataset.  
- This avoids cherry-picking only traded symbols.  
- Provides a fair comparison to the “market opportunity set.”

---

## Key Parameters
- **Cutoff Time**: 14:45  
- **Top N**: 5  
- **Fee Rate**: 0.0003 (0.03% per side)  
- **Holding Period**: 1 day (T → T+1)  
- **Liquidity Threshold**: 0.8 × 20-day median volume (yesterday’s)  
- **Extreme Return Filter**: |yesterday return| < 8%

---

## Algorithm Logic
1. Load 1-minute data from all symbols.  
2. For each day, compute `signal = cutoff/ open − 1`.  
3. Apply filters using yesterday’s info.  
4. Rank all symbols by signal.  
5. Pick Top N, buy at T close, sell at T+1 close.  
6. Aggregate daily returns into portfolio PnL.  
7. Compare against Buy&Hold benchmark.  
8. Output metrics (Annual Return, Sharpe, MaxDD, MAR, Win Rate).

---

## Output Files
Generated in the `./momentum_out/` folder:
- `metrics_momentum.csv`: Performance summary (annualized return, Sharpe, MaxDD, MAR, win rate).  
- `trades_momentum_simple.csv`: Trade log (date, symbol, returns).  
- `equity_momentum.png`: Strategy vs Buy&Hold equity curve.  
- `metrics_train_test_full.csv`: Metrics split into Train (first 70%), Test (last 30%), and Full.  
- `equity_train.png`, `equity_test.png`, `equity_full.png`: Equity curves by split.  
- `trade_frequency_full.png`: Number of trades per day (exit events).  
- `split_info.txt`: Actual cutoff date used for 70/30 split.

---

## Performance Metrics
Key metrics calculated:
- **Annual Return**: Compounded annualized gross/net return.  
- **Sharpe Ratio**: Risk-adjusted return (daily frequency × √252).  
- **Maximum Drawdown**: Worst peak-to-trough portfolio decline.  
- **MAR Ratio**: Return ÷ MaxDD.  
- **Win Rate (Monthly)**: Fraction of profitable months.

---

## Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib reportlab
```

### Running the Strategy
```bash
python momentum_strategy_runner.py   --zip "stock_search_cn_1min.zip"   --out "./momentum_out"   --cutoff "14:45"   --top_n 5   --fee 0.0003
```

### Train/Test Split
The script automatically creates a **70/30 split by time** on the daily calendar.  
- **Train**: First 70% of dates.  
- **Test**: Last 30% of dates.  
- **Full**: Entire sample.

---

## Risk Considerations
- **Gap risk**: Overnight announcements/policy shifts can reverse momentum.  
- **Liquidity trap**: Thinly traded stocks may not execute at modeled prices.  
- **Transaction costs**: Slippage and taxes (e.g., China stock stamp duty) are not fully modeled.  
- **Regime shifts**: Momentum continuation is not guaranteed in all markets.  
- **Benchmark gap**: Buy&Hold benchmark may have different volatility profile.

---

## Disclaimer
This strategy is for **educational and research purposes only**.  
Past performance does not guarantee future results. Always test carefully before live trading.
