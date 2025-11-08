# Pairs Reversion Strategy Report (Bias Fix v1)

## Data & Split
- Source: `crypto_data.zip`, resampled to 15-minute closes.
- Train window: up to 2024-06-30; Test window: 2024-07-01 onward.
- Legs: BTC hedge with ETCUSDT, APTUSDT, ARBUSDT.

## Strategy Outline
- Estimate static hedge betas on the train set (`asset ≈ beta × BTC + intercept`) and monitor residual spreads.
- Trade mean reversion: go long spread when z-score ≤ −2.0, short when ≥ +2.0; flatten when |z| < 0.8.
- Allocate 18% notional per active spread. Apply volatility targeting with *lagged* annualised residual volatility (25% cap, leverage ≤1.2×) to avoid lookahead.
- Transaction cost: 5 bps per leg, counted on both spread and hedge turnover.

## Performance (bias-free)

| Segment | Annual Return | Sharpe | Max Drawdown | MAR | Monthly Win % | Avg Turnover (per 15 min) |
| --- | --- | --- | --- | --- | --- | --- |
| **Strategy – Train** | 711.2% | 9.02 | 6.0% | 119.38 | 100% | 1.82% |
| **Strategy – Test** | 195.1% | 4.58 | 16.6% | 11.72 | 80% | 1.62% |
| **Strategy – Overall** | 335.3% | 6.25 | 16.6% | 20.14 | 91.3% | 1.70% |
| **BTC Buy & Hold – Overall** | 74.9% | 1.39 | 35.9% | 2.09 | 65.2% | 0.00% |

Additional observations:
- Average active spreads ≈1.0 (max 3).
- Equity curve with entry/exit markers: `docs/pairs_reversion_biasfix_1/pnl_pairs_reversion_1.png`.
- Metrics CSV: `docs/pairs_reversion_biasfix_1/metrics_pairs_reversion_1.csv`; JSON summary: `docs/pairs_reversion_biasfix_1/strategy_summary_pairs_reversion_1.json`.

## Reproduction
```bash
scripts/run_strategy_pairs_reversion_1.sh docs/pairs_reversion_biasfix_1
```

Omit the argument to default to `docs/pairs_reversion_biasfix_1`, or pass a custom folder for new runs.
