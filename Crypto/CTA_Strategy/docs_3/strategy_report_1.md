# CTA Momentum Sleeve Report

- Performance Plot: performance_curve_1.png

## Asset Selection (training-driven)
- Momentum asset (highest 60-day return): XRPUSDT (20.95%)
- Convex asset (highest training skew): SUIUSDT (1.97)
- Core asset (liquid benchmark with strong training Sharpe): BTC_USD (Sharpe 1.95)

## Key Metrics
### Train
- Annualized Net Return: 68.73%
- Annualized Gross Return: 74.35%
- Sharpe Ratio: 1.38
- Max Drawdown (equity): -37.33%
- Max Drawdown (monthly): -25.03%
- MAR Ratio: 1.84
- Monthly Win Rate: 72.73%

### Test
- Annualized Net Return: 180.76%
- Annualized Gross Return: 189.43%
- Sharpe Ratio: 2.02
- Max Drawdown (equity): -40.44%
- Max Drawdown (monthly): -24.29%
- MAR Ratio: 4.47
- Monthly Win Rate: 75.00%

### Full
- Annualized Net Return: 117.65%
- Annualized Gross Return: 124.64%
- Sharpe Ratio: 1.73
- Max Drawdown (equity): -40.44%
- Max Drawdown (monthly): -25.03%
- MAR Ratio: 2.91
- Monthly Win Rate: 75.00%
- Average Daily Turnover (abs weight change): 0.17

## Buy & Hold Benchmark (Full Period)
- Annualized Return: 53.80%
- Sharpe Ratio: 0.97
- Max Drawdown (equity): -61.34%
- MAR Ratio: 0.88
- Monthly Win Rate: 54.17%

## Reproduction
- Activate environment if needed: `conda activate myenv`
- Run: `python code_1/strategy_analysis_1.py`