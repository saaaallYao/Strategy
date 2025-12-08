# Nebula Pairs (KuCoin mean-reversion)

Fixed basket pairs reversion on KuCoin spot (BTC-USDT vs ETC/ARB/APT by default). Backtest and paper trading share the same signal engine and configs under `code/`.

## Backtest (offline, KuCoin history)
- Fetches public 1min klines from KuCoin, resamples to 15min, and runs the signal engine.
- Command: `bash scripts/run_backtest.sh --lookback-days 400`
- Optional flags/env (prefix `FP_`): `--base-symbol BTC-USDT`, `--pairs ETC-USDT APT-USDT ARB-USDT`, `--resample-rule 15min`, `--train-split 2024-09-01`, `--output-prefix nebula_pairs`.
- Outputs in `data/`: `backtest_results_{prefix}.csv`, `backtest_weights_{prefix}.csv`, `backtest_zscores_{prefix}.csv`, `backtest_equity_{prefix}.csv`, `backtest_metrics_{prefix}.json`, plus raw/scaled positions.

## Paper trading (streaming klines)
- Polls KuCoin spot klines (no API key), generates weights, and simulates fills with a simple paper broker (fee default 0.0005).
- Command: `bash scripts/run_paper.sh`
- Optional env overrides: `FP_BASE_SYMBOL`, `FP_PAIRS` (comma-separated), `FP_RESAMPLE_RULE`, `FP_SEED_DAYS`, `FP_INITIAL_CAPITAL`, `FP_CUSHION_BARS`, `FP_BAR_GRACE_SECONDS`, `FP_MAX_BARS`, `FP_LOG_PREFIX`.
- Outputs in `data/`: `paper_trades_{prefix}_1.csv`, `paper_equity_curve_{prefix}_1.csv`, `paper_signals_{prefix}_1.csv`, `paper_signal_state_{prefix}_1.csv`.

## Quick notes
- Requires Python 3 with `pandas`, `numpy`, and `requests` installed (`python3 -m pip install -r requirements.txt` style).
- All code lives in `code/`; scripts set `PYTHONPATH` for you. Data/logs are written under `data/`.
