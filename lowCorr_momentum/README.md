# Low-Correlation Momentum Basket Strategy

## Overview
This repository implements a **cross-sectional momentum strategy** that constructs long-short equity baskets using a **low-correlation universe selection** method.  
The strategy dynamically selects a subset of assets with low pairwise correlation during the training period, applies **volatility-normalized momentum signals**, and rebalances periodically to form diversified long/short portfolios.  

It is designed to explore robust statistical arbitrage across multiple assets while controlling risk, turnover, and transaction costs.  

---

## Strategy Description

### Core Concept
- **Low-Correlation Universe Selection**: Use greedy correlation pruning on the training set to select assets with low mutual correlation.  
- **Cross-Sectional Momentum**: Rank assets by volatility-adjusted momentum scores.  
- **Volatility Normalization**: Scale signals by inverse realized volatility.  
- **Risk Parity Weighting**: Allocate 50% gross exposure to long side and 50% to short side.  

### Key Parameters
- **Fee Rate**: 0.03% per side (0.0003), applied to turnover.  
- **Lookback Fast**: 24 bars (~2 hours with 5-min data).  
- **Lookback Slow**: 96 bars (~8 hours with 5-min data).  
- **Volatility Window**: 780 bars (~10 RTH days with 5-min data).  
- **Rebalance Interval**: Every 12 bars (~60 minutes).  
- **Selection Fraction (Q)**: Top 20% (long) and bottom 20% (short).  
- **Training/Test Split**: 70% train, 30% test.  

### Algorithm Logic
1. **Universe Selection**  
   - Compute correlation matrix on training set returns.  
   - Apply greedy selection to pick `K` assets under correlation threshold.  
2. **Signal Calculation**  
   - Momentum = 0.6 × fast vol-normalized return + 0.4 × slow vol-normalized return.  
   - Rank assets at each rebalance point.  
3. **Portfolio Construction**  
   - Go long top-Q assets, short bottom-Q assets.  
   - Weights ∝ 1/volatility (inverse-vol), normalized within each side.  
   - Gross exposure = 100% (50% long, 50% short).  
4. **Execution & Costs**  
   - Forward-fill weights until next rebalance.  
   - Deduct transaction costs proportional to turnover.  

---

## Files

- **lowcorr_momentum.py**  
  - Core implementation with:  
    - Data loading & preprocessing  
    - Universe selection  
    - Strategy backtest engine  
    - Performance metrics  
    - Equity curve visualization  

- **metrics_all.csv**  
  - All performance metrics (Train/Test/Full for Strategy & Equal-Weight Buy&Hold) across different `K`.  

- **universes_params.csv**  
  - Universe composition for each `K` with key parameters (correlation threshold, lookbacks, rebalance interval, etc).  

- **equity_all.png**  
  - Overlay of equity curves across all `K`.  

- **equity_K{K}.png**  
  - Individual equity curve (Strategy vs Equal-Weight BH) with annotated performance metrics.  

---

## Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib
```

### Running the Strategy
```bash
python lowcorr_momentum.py
```

### Data Requirements
- Input CSV must contain:
  - **timestamp** column (datetime format)  
  - **asset price columns** (numeric, one per ticker)  
- Example file: `universe.csv`  
- 5-minute bar data recommended, covering multiple assets.  

---

## Performance Metrics

Key metrics calculated:
- **Annual Return**: Annualized percentage return  
- **Sharpe Ratio**: Risk-adjusted return measure  
- **Maximum Drawdown (MaxDD)**: Largest peak-to-trough decline  
- **MAR Ratio**: Return-to-drawdown ratio  

Outputs are saved in:
- `metrics_all.csv` → detailed metrics for all `K` values  
- `equity_all.png` → overlay comparison chart  
- `equity_K{K}.png` → strategy vs BH equity curve  

---

## Example Results (Demo)

Top performing configurations (`K=5, K=6, K=10`) typically show:
- Sharpe Ratios above **1.5–2.0** in test sets  
- Annualized returns between **20–40%**  
- Maximum drawdowns in the range **-15% to -25%**  
- MAR Ratios above **1.0**, showing balanced risk-return efficiency  

---

## Configuration

Modify parameters in the script (`lowcorr_momentum.py`):
```python
LOOKBACK_FAST = 24       # Fast momentum lookback
LOOKBACK_SLOW = 96       # Slow momentum lookback
REBALANCE_EVERY = 12     # Bars between rebalances
Q = 0.2                  # Fraction of assets long/short
COST_RATE = 0.0003       # Transaction cost per turnover
K_LIST = [10, 6, 5, 4]   # Different universe sizes to test
```

---

## Risk Considerations

### Strategy Risks
- **Model Risk**: Assumes persistence of momentum & decorrelation.  
- **Market Risk**: Correlation structures may shift dramatically.  
- **Execution Risk**: Slippage and latency not modeled.  
- **Data Risk**: Requires clean, synchronized intraday data.  

### Limitations
- Assumes perfect execution at bar close prices.  
- Does not account for overnight gaps.  
- Restricted to liquid assets with reliable 5-min data.  

---

## Future Enhancements
- Dynamic correlation thresholding  
- Regime-switching momentum models  
- Transaction cost & slippage modeling  
- Portfolio-level risk overlays (e.g., HRP or volatility targeting)  
- Integration with live execution (Alpaca / IBKR APIs)  

---

## Disclaimer
This strategy is for **educational and research purposes only**.  
Past performance does not guarantee future results.  
Always conduct thorough testing and risk assessment before deploying in live trading.  
