# Fixed Pairs Strategy – Paper Trading (KuCoin spot)

Paper trading for the pairs reversion strategy using Binance US klines (public, no API key needed), with standardized logs (`data/paper_*_fixed_pairs_1.csv`).

## Run
```bash
bash scripts/run_fixed_pairs_paper.sh
# optional overrides:
FP_BASE_SYMBOL=BTCUSD \
FP_PAIRS="ETCUSDT,APTUSDT,ARBUSDT" \
FP_RESAMPLE_RULE=15min \
FP_SEED_DAYS=200 \
FP_INITIAL_CAPITAL=1000000 \
FP_CUSHION_BARS=5 \
FP_BAR_GRACE_SECONDS=5 \
FP_MAX_BARS=0 \
FP_LOG_PREFIX=fixed_pairs \
bash scripts/run_fixed_pairs_paper.sh
```

## Outputs (paper trading)
- `data/paper_trades_{prefix}_1.csv` – executed rebalances (default prefix `fixed_pairs`)
- `data/paper_equity_curve_{prefix}_1.csv` – equity snapshots with weights
- `data/paper_signals_{prefix}_1.csv` – target weights per rebalance

## Notes
- Uses KuCoin spot klines (public, no API key) via `code/kucoin_client.py`.
- Signal engine from `pairs_strategy` (reversion config), broker from `fixed_pairs_pt.broker` (fee default 0.0005).
- Polls on bar close for the chosen resample rule (default 15min); cushions incremental fetches to avoid gaps.***
