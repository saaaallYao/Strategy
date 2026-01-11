# Results & Comparisons

This folder contains the backtest and sweep results used to select the current strategy configuration.

## Key result files
- `results/live_monitor_backtest_20260109.csv`
- `results/live_monitor_backtest_20260109.md`
- `results/fee5e4_sweep_random_30d.csv`
- `results/fee5e4_sweep_random_30d.md`
- `results/fee5e4_refine_random_30d.csv`
- `results/fee5e4_refine_random_30d.md`
- `results/fee5e4_stoploss_sweep_random_30d.csv`
- `results/fee5e4_stoploss_sweep_random_30d.md`
- `results/fee5e4_sl10_more_windows.csv`
- `results/fee5e4_sl10_more_windows.md`

## Recommended config (fee=5e-4)
- `min_edge_return=0.0014`
- `stop_loss_pct=0.010`
- `z_enter=1.2`, `z_exit=0.4`
- `signal_persistence=3`, `cooldown_bars=30`, `min_hold_bars=30`

## Notes
- Random window tests use multiple seeds; see each `*.md` for details.
- Full and latest-30d runs are included where noted in the result files.
