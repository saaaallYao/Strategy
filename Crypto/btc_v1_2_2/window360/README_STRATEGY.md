# Fee-Aware BTC-ETH Strategy (Window=360)

This folder is a self-contained copy of the strategy package for running
BTC/ETH stat-arb with fee-aware logic and the baseline_best parameters.

## Key files
- `fina/strategies/crypto/btc_eth_v1/live_monitor.py`
- `fina/strategies/crypto/btc_eth_v1/run_strategy_bar_by_bar.py`
- `fina/strategies/crypto/btc_eth_v1/strategy_core.py`
- `fina/strategies/crypto/btc_eth_v1/strategy_engine.py`

## Signal Logic (High Level)

1. **Rolling beta + residual**: compute BTC~ETH regression over rolling window, take residual.
2. **Z-score**: z = (residual - rolling_mean) / rolling_std.
3. **Entry**: open long/short when |z| exceeds z_enter and expected edge exceeds fee-adjusted threshold.
4. **Exit**: close when |z| falls below z_exit or fee-aware exit triggers.
5. **Risk controls**: stop-loss, min-hold, cooldown, and persistence to reduce churn.
6. **Sizing**: position size scaled by residual volatility and capped by `inv_cap`.

## Recommended config (fee=5e-4)
- `min_edge_return=0.0014`
- `dyn_edge_enabled=true`, `dyn_edge_fee_mult=5.0`, `dyn_edge_vol_mult=0.5`
- `fee_exit_enabled=true`, `fee_exit_mult=2.0`
- `stop_loss_pct=0.010`
- `z_enter=1.2`, `z_exit=0.4`
- `window=360`
- `signal_persistence=3`, `cooldown_bars=30`, `min_hold_bars=30`

## Run examples
- Live monitor:
  - `python -m fina.strategies.crypto.btc_eth_v1.live_monitor`
- Bar-by-bar backtest:
  - `python -m fina.strategies.crypto.btc_eth_v1.run_strategy_bar_by_bar`
- Paper trading runner (Alpaca data):
  - `python run_paper_trading.py`
  - Example overrides: `python run_paper_trading.py --fee 0.0005 --min-edge 0.0014 --dyn-edge-fee-mult 5.0 --fee-exit-mult 2.0`

## Notes
- This is a copy of `orig_pt/fina` plus minimal top-level files to run.
- Adjust config inside `live_monitor.py` or `run_strategy_bar_by_bar.py` as needed.

## Differences vs Original

- Fees increased to 5e-4 (original used 2e-4).
- Added fee-aware entry filter (`min_edge_return`) and dynamic edge scaling.
- Added fee-aware exit (exit if expected edge is too small after fees).
- Added stop-loss at 1%.
- Added signal persistence and holding/cooldown controls to reduce churn.
- Window shortened from 720 to 360 for faster adaptation.

## Optimization Period

- Window=360 was selected using the most recent 12 months:
  - 2025-01-16 -> 2026-01-16
