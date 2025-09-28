Residual-Pairs Signals Package
--------------------------------
Contents:
- *_beta.png: Rolling beta vs BTC (30d window, shifted 1h)
- *_spread.png: Residual cumulative spread with 10d rolling mean (shifted)
- *_zscore.png: Standardized spread Z with ±0.8 thresholds
- *_signals.png: Close price with Long (Z<=-0.8) / Short (Z>=0.8) markers
- *_signals.csv: Table of close, beta, residual, spread, zscore (hourly)

Notes:
- Data window: 2023-08-01 to 2025-08-01 (UTC), resampled to 1H last prices.
- Anti-lookahead: beta/rolling stats are shifted by 1 hour; signal visual markers are trigger times.
- Thresholds and windows: lookback=30d for beta; z-window=10d; z-threshold=0.8.
