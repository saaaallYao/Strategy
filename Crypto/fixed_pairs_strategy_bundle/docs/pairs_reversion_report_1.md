# Pairs Reversion Strategy Report (1)

## Data & Split
- Source: `crypto_data.zip`, 1-minute bars from 2023-08-01 to 2025-08-02 resampled to 15-minute closes.
- Train window: up to 2024-06-30; Test window: 2024-07-01 onward.
- Universe: BTC as hedge asset with ETCUSDT, APTUSDT, and ARBUSDT as spread legs.

## Strategy Outline
- Estimate static hedge ratios (`asset = beta × BTC + intercept`) on the training window, then monitor the residual spread.
- Enter a *long spread* (long alt, short BTC) when z-score ≤ −2.0; enter *short spread* when z-score ≥ +2.0. Exit once |z-score| < 0.8.
- Allocate 18% notional per active spread and apply an exponential volatility target (25% annualised cap, leverage ≤ 1.2×) on aggregate P&L.
- Transaction cost: 5 bps per leg with turnover measured on both asset and hedge.

Simple operation:
1. Recompute hedge betas on the latest training cut.
2. Track each spread’s rolling mean and standard deviation (36-hour window).
3. Trade only when the residual exceeds ±2 σ, close when it mean-reverts inside ±0.8 σ.

## Performance Summary

| Segment | Annual Return | Sharpe | Max Drawdown | MAR | Monthly Win % | Avg Turnover (per 15 min) |
| --- | --- | --- | --- | --- | --- | --- |
| **Strategy – Train** | 684.3% | 9.85 | 5.5% | 124.9 | 100% | 1.79% |
| **Strategy – Test** | 145.8% | 4.49 | 15.6% | 9.33 | 80% | 1.60% |
| **Strategy – Overall** | 283.9% | 6.59 | 15.6% | 18.17 | 91.3% | 1.68% |
| **Buy & Hold (BTC) – Overall** | 74.9% | 1.39 | 35.9% | 2.09 | 65.2% | 0.00% |

Additional notes:
- Average active spreads ≈1.0 (max 3); signals cluster during volatility spikes.
- Equity curve and entry/exit marks: `docs/pnl_pairs_reversion_1.png`.
- Metrics CSV: `docs/metrics_pairs_reversion_1.csv`; JSON summary: `docs/strategy_summary_pairs_reversion_1.json`.

## Reproduction
```bash
scripts/run_strategy_pairs_reversion_1.sh
```

Outputs land in `docs/`.
