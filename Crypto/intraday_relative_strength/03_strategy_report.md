# 03 Intraday Relative-Strength Reversion

## Core Idea
- Resample prices to 30-minute bars (≈48 decisions per day).
- Measure each alt-coin’s performance relative to BTC over the past ~4 hours.
- Buy the three weakest relative movers (expecting mean reversion) while shorting BTC for an equal notional hedge.
- Inverse-volatility weights and a 22% annualised volatility cap keep leverage moderate (avg ≈0.35×).
- Cooldown logic forces positions to stay flat for four bars after exit signals, damping churn.

## Performance Summary
| Period | Ann. Return | Sharpe | Max DD | MAR | Monthly Win Rate | Avg Trades/Day |
| --- | --- | --- | --- | --- | --- | --- |
| Train (≤ 2024-06-30) | 228% | 6.97 | -10.8% | 21.1 | 81.8% | 47.2 |
| Test (≥ 2024-07-01) | 831% | 11.41 | -3.6% | 230.9 | 100% | 47.2 |

BTC buy & hold over the same window: Sharpe 1.91 / 1.43 (train/test) with drawdowns -20.1% / -27.9%.

## Risk & Activity
- Average turnover per 30-minute bar: 0.40 (train) / 0.34 (test); median 0.34 / 0.31.
- Portfolio stays near market-neutral (BTC hedge ensures net exposure ≈0), with leverage safely below the 1.6× cap.
- Monthly drawdowns stay inside -0.8% (train) and are flat over the test window, pointing to a smooth equity curve.

## Artefacts
- Metrics: `docs/03_strategy_metrics.csv`, `docs/03_benchmark_metrics.csv`
- Figures: `docs/figures/03_equity.png`, `docs/figures/03_trades.png`, `docs/figures/03_monthly_returns.png`
- Manifest: `docs/03_outputs.json`

> **Reproduction:** `python code/03_intraday_relative_strength.py`
