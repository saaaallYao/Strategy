# Pairs Reversion Bias-Fix README

## 1. Strategy Overview
- **Goal:** Capture mean reversion inside cointegrated crypto/BTC spreads while keeping exposure market-neutral to broad BTC moves.
- **Universe:** BTC hedge asset with ETCUSDT, APTUSDT, ARBUSDT residual spreads.
- **High-level flow:**
  1. Load minute data (`crypto_data.csv`), resample to 15-minute closes.
  2. Estimate static hedge betas on the training window (`<= 2024-06-30`).
  3. Track residual spreads with a 144-bar (36 h) rolling mean/std.
  4. Enter long/short when z-score ≤ −2.0 or ≥ +2.0; flatten inside ±0.8.
  5. Allocate 18 % notional per active spread; aggregate exposure volatility-targeted (25 % annual cap, max leverage 1.2×).
  6. Charge 5 bps per leg on turnover (spread and hedge).

## 2. Implementation Notes
- **Source:** `code/strategy_pairs_reversion_1.py`
- **Lookahead safety:**
  - Positions are shifted one bar (`lagged_positions`) before returns are applied.
  - Residual volatility driving the leverage cap is shifted by one bar (`ann_vol.shift(1)`), so each sizing decision uses only prior information.
  - Rolling statistics (`spread_ma`, `spread_std`) depend solely on historical observations.
- **Outputs:** Metrics, JSON summary, plot, and reports land in `docs/pairs_reversion_biasfix_1/`.

## 3. Running the Backtest
```bash
scripts/run_strategy_pairs_reversion_1.sh docs/pairs_reversion_biasfix_1
```
- Omit the argument to reuse the default folder.
- All artefacts will refresh inside the chosen directory.

## 4. Key Results (Bias-Free)
| Segment | Ann. Return | Sharpe | Max DD | MAR | Monthly Win% |
| --- | --- | --- | --- | --- | --- |
| Train | 711% | 9.02 | 6.0% | 119.4 | 100% |
| Test | 195% | 4.58 | 16.6% | 11.7 | 80% |
| Overall | 335% | 6.25 | 16.6% | 20.1 | 91% |

## 5. Operational Checklist
- Recompute hedge betas after any data extension.
- Confirm rolling window (144 bars) spans full history; warmup rows are dropped naturally.
- Monitor turnover (~1.7 % per 15 min step avg) and adjust cost assumptions if trading venue fees differ.

## 6. Future Enhancements
1. Broaden the spread set (e.g., SOL/BTC, LINK/BTC) to diversify residual alpha.
2. Make z-score entry/exit thresholds volatility-adaptive to filter noisy regimes.
3. Add a timeout or stop mechanism for spreads that fail to mean revert within N bars.
