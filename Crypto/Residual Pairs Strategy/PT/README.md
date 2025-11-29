# Residual Pairs Strategy – Paper Trading Runner

Paper-trading the residual pairs market-neutral strategy using KuCoin spot 1m data (resampled to 1H). No API key required.

## Run
```bash
bash scripts/run_residual_pairs_paper.sh
# or override envs:
RESIDUAL_UNIVERSE="BTC-USDT,ETH-USDT,SOL-USDT,XRP-USDT" \
RESIDUAL_POLL_INTERVAL=60 RESIDUAL_HISTORY_MINUTES=120000 \
RESIDUAL_FEE_PER_SIDE=0.0003 RESIDUAL_LOOKBACK_DAYS=20 \
RESIDUAL_ZWIN_DAYS=45 RESIDUAL_K_PER_SIDE=3 RESIDUAL_Z_THRESHOLD=0.8 \
bash scripts/run_residual_pairs_paper.sh
```

## Outputs (paper trading)
- `data/paper_trades_residual_pairs_1.csv` – Each rebalance (from/to weights, turnover, fee, equity)
- `data/paper_equity_curve_residual_pairs_1.csv` – Equity snapshots with weights per hour
- `data/paper_signals_residual_pairs_1.csv` – Target weights at each rebalance

## Strategy defaults
- Universe: BTC-USDT hedge + ETH/SOL/XRP/DOGE/ADA/MATIC (override with `RESIDUAL_UNIVERSE`)
- Windows: beta lookback 20 days (1H bars), z-score window 45 days
- Selection: top K=3 per side where |Z| >= 0.8; long/short 0.5 notional per side; BTC hedge via beta
- Fees: 0.0003 per side
- Rebalance cadence: on each new 1H bar (polling 60s for fresh 1m data)
