# pairs_strategy

Unified wrapper around the fixed pairs mean-reversion strategy. The shared
signal engine lives in `pairs_strategy/signal.py` and is reused by both the
offline backtester and the live paper-trading loop so weights, PnL, and
diagnostics stay aligned across environments.

## Strategy snapshot
- **Universe:** BTC hedge leg (`BTC_USD` column) paired with residual spreads in
  ETCUSDT, APTUSDT, and ARBUSDT (configurable).
- **Signal:** For each spread, compute log-price residuals versus BTC, track a
  144-bar rolling z-score (15-minute bars by default), and enter ±18 % notional
  when |z| ≥ 2.0, exiting inside ±0.8.
- **Risk:** Aggregate portfolio volatility-targeted to 25 % annualized with a
  1.2× gross leverage cap, including 5 bps per-leg transaction costs.

## Layout
```
pairs_strategy/
├─ README.md
├─ __init__.py
├─ signal.py          # shared signal generator (imports original strategy math)
├─ backtest.py        # CLI backtester built on the shared signal module
└─ paper_trading.py   # Live/PT runner that streams BinanceUS data
```

## Shared signal module
Use `PairsSignalEngine` whenever you have a price DataFrame with columns
`[base_asset, *pairs]`. Example:

```python
from fixed_pairs_strategy_bundle.code.strategy_pairs_reversion_1 import StrategyConfig
from pairs_strategy.signal import PairsSignalEngine

cfg = StrategyConfig(resample_rule="15min", train_split="2024-06-30")
engine = PairsSignalEngine(cfg)
weights, state = engine.latest_weights(price_frame)
```

`state` contains `results` (returns/equity), `raw_positions`, `scaled_positions`,
and per-symbol `weights`, so downstream consumers get the same diagnostics as
the original backtest.

## Backtesting with local data
`pairs_strategy/backtest.py` reads whatever dataset you point to. By default it
uses `fixed_pairs_strategy_bundle/crypto_data.zip` (CSV inside the zip named
`crypto_data.csv`), but you can pass local files:

```bash
python pairs_strategy/backtest.py \
  --dataset /path/to/history.zip \
  --dataset-member my_prices.csv \
  --base-asset BTC_USD \
  --pairs ETCUSDT APTUSDT ARBUSDT \
  --output-dir pairs_strategy/artifacts_backtest_local
```

The loader expects:
- A CSV with an `open_time` column (UTC) and price columns matching your
  `--base-asset`/`--pairs`.
- Data packed inside a ZIP; if you keep a plain CSV, zip it or extend the loader
  to read raw CSV before running.

Outputs written to `--output-dir`:
- `backtest_results.csv`, `raw_positions.csv`, `scaled_positions.csv`,
  `weights.csv`
- `metrics_summary.json` (Sharpe, drawdown, turnover, etc.)

## Paper trading
`pairs_strategy/paper_trading.py` reuses the same engine but feeds it live
Binance US data via the existing REST helper and paper broker. Example run:

```bash
python pairs_strategy/paper_trading.py \
  --seed-days 200 \
  --pairs ETCUSDT APTUSDT ARBUSDT \
  --iterations 0 \
  --train-split 2025-11-12 \
  --output-dir pairs_strategy/artifacts_paper_trading
```

Key flags:
- `--seed-days`: bootstrap history window before live polling.
- `--iterations`: rebalance count after warmup (`0` = run indefinitely).
- `--cushion-bars`, `--bar-grace-seconds`: control gap filling and polling lag.
- `--max-bars`: optional rolling window to cap in-memory history.

Artifacts land under `--output-dir`:
- `pnl.csv`, `trades.csv`, `positions.csv`

Because both runners call `PairsSignalEngine`, the live loop matches the offline
math (betas, z-scores, scaling, costs) exactly—swap datasets or configs in one
place and both paths stay consistent.
