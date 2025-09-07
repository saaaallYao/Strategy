# XOM–USO Statistical Arbitrage (Pairs Mean-Reversion)

## Overview
This repository implements a **dollar‑neutral pairs mean‑reversion** strategy on **XOM (Exxon Mobil)** and **USO (United States Oil Fund)**.  
It estimates a **dynamic hedge ratio** β with **Recursive Least Squares (RLS)**, constructs a spread
**spreadₜ = XOMₜ − βₜ·USOₜ**, standardizes it into a Z‑score, and trades reversion with **next‑bar execution** and **volatility targeting**.  
All estimators use `shift(1)` (history‑only) to avoid look‑ahead.

---

## Strategy Description

### Core Concepts
- **Dynamic Hedge Ratio (β)**: Online RLS updates βₜ using only data up to time t.  
- **Mean Reversion on Spread**: Z‑score of the spread drives entries/exits (no drift assumption on standardized spread).  
- **Dollar Neutrality**: Position sizing offsets XOM with β‑scaled USO to keep net beta‑exposure compact.  
- **Volatility Targeting**: Rescales position to a **daily volatility target** (e.g., 1–2%/day) using past‑day realized vol (`shift(1)`).  
- **RTH Session Discipline**: Trade only during **US cash session 09:30–16:00 America/New_York** (optional open/close buffers).

### Key Parameters (typical defaults)
- **Entry / Exit thresholds**: `k_enter = 2.0`, `k_exit = 0.5` (symmetric).  
- **Z‑score window**: `win ≈ BPD` (bars‑per‑day inferred from timestamps).  
- **Volatility target**: `vol_target = 0.02` (≈2%/day).  
- **Fees**: `fee_rate = 0.0003` (= 0.03% per trade).  
- **Train/Test split**: 70/30 by default, or controlled via `--split-date`.  
- **RTH buffers**: `--open-buffer`, `--close-buffer` (minutes).

### Algorithm Logic
1. **Preprocessing**
   - Parse/validate datetime index (monotonic, de‑duplicated, correct timezone).  
   - Convert to **America/New_York** and strip tz for session filtering.  
   - Optionally filter to **RTH 09:30–16:00** with buffers.
2. **Dynamic β via RLS**
   - Update β sequentially (history‑only).  
   - Compute **spread = y − β·x**.
3. **Z‑score & Signals**
   - Rolling mean/std of spread (both `shift(1)`).  
   - `z > +k_enter` → **short spread** (short y, long β·x).  
   - `z < −k_enter` → **long spread** (long y, short β·x).  
   - `|z| < k_exit` → **exit**.
4. **Sizing & Vol Target**
   - Base dollar‑neutral leg weights from β magnitude and sign.  
   - Scale by **vol_target / past‑day vol** (EWMA/rolling, `shift(1)`).  
5. **Execution & PnL**
   - **Next‑bar execution**: positions applied at t+1.  
   - PnL uses **position(t) × return(t+1)**; transaction cost = turnover × fee.
6. **Benchmark**
   - **BH (50/50)** equal‑weight buy‑and‑hold of XOM & USO.

---

## Data Requirements

### Minimum columns
- **Datetime column** (e.g., `datetime`, `timestamp`, `time`, `date`, `dt`).  
- **Two price columns** for y/x (prefer **close/adj_close/last**).

### Example schema
```csv
datetime,XOM,USO
2025-08-01 09:30:00,118.62,76.41
2025-08-01 09:31:00,118.59,76.38
...
```

### Frequency & timezone
- Bar frequency: **1–5 minute** preferred (the script infers BPD/BPY).  
- Timezone: ideally **America/New_York**; UTC or other tz are converted.

---

## Execution

### Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pandas numpy matplotlib
```

### Run (explicit columns + tz + RTH)
```bash
python run_baseline_userdata.py \
  --csv data/xom_uso_1m.csv \
  --dtcol datetime \
  --ycol XOM --xcol USO \
  --tz America/New_York --rth \
  --k-enter 2.0 --k-exit 0.5 \
  --vol 0.02 --fee 0.0003
```

### Notable CLI flags
| Flag | Meaning | Typical |
|---|---|---|
| `--csv` | Input CSV path | `data/xom_uso_1m.csv` |
| `--dtcol` | Datetime column name | `datetime` |
| `--ycol`, `--xcol` | Price column names | `XOM`, `USO` |
| `--tz` | Timezone for session logic | `America/New_York` |
| `--rth` | Filter to 09:30–16:00 | flag |
| `--open-buffer` | Minutes after open | `0` |
| `--close-buffer` | Minutes before close | `0` |
| `--k-enter` | Entry z threshold | `2.0` |
| `--k-exit` | Exit z threshold | `0.5` |
| `--win` | Z‑score window (bars) | `≈ BPD` |
| `--vol` | Target **daily** vol | `0.02` |
| `--fee` | Per‑trade cost | `0.0003` |
| `--split-date` | Explicit train/test boundary | `YYYY-MM-DD` |

---

## Outputs
```
outputs/
├── metrics.csv         # KPIs per segment (train/test/full)
├── trades.csv          # Executions (side, size, z, timestamps)
├── equity_curve.csv    # Net equity time series
└── meta.json           # {bars_per_day, BPY, paths, run params}
```

### Metrics (columns may vary)
- `segment` (train/test/full)  
- `ann_return`, `ann_vol`, `sharpe`  
- `max_drawdown`, `MAR` (= ann_return / |max_drawdown|)  
- `monthly_win_rate` (count‑wins / count‑months)  
- `turnover`, `n_trades`

---

## Robustness & No‑Look‑Ahead Guarantees
- Rolling μ/σ for Z are **`shift(1)`**; vol‑target scaler is **`shift(1)`**.  
- PnL uses **previous position × current return** (next‑bar execution).  
- RLS β uses only past/current bar; positions still applied next bar.  
- Time handling: tz‑convert to **America/New_York**, drop tz for `between_time`.  
- Bars‑per‑day (`BPD`) inferred robustly (dedup timestamps, ignore non‑positive gaps).

> Alternative accounting (also look‑ahead safe): “t‑1 signal, t build, t PnL by new position, t fees”. Ensure fee and position shifts are aligned if you switch conventions.

---

## Troubleshooting

- **`TypeError: Cannot cast TimedeltaIndex to dtype float64`**  
  When converting bar gaps, do **not** `astype(float)`. Use `.asi8/1e9` or `.dt.total_seconds()`.
- **`NameError: name 'json' is not defined`**  
  Add `import json`. If writing `Path` into JSON, cast with `str(path)`.
- **Wrong price columns picked**  
  Use `--ycol/--xcol` explicitly; prefer `close/adj_close/last`. Exclude `open/high/low/volume`.
- **Session cut looks off**  
  Confirm tz handling; convert to `America/New_York` **before** RTH filtering.
- **Window too small**  
  Ensure `--win ≥ BPD`. With sparse data, increase window or move to daily bars.

---

## Evaluation & Reporting
- Compare against **BH(50/50)**.  
- Inspect **Sharpe**, **Max Drawdown**, **MAR**, **monthly win‑rate**, and **turnover**.  
- Visuals: net equity curve, entry/exit markers (green/red), drawdown chart (optional).

---

## Extending
- **Stops/Timeouts** on extreme |z| or max holding bars.  
- **Pair pre‑filter** by cointegration tests (ADF), half‑life, rolling stability.  
- **Basket of pairs** with risk parity / ERC across spreads.  
- **Execution realism**: slippage models, queue priority.  
- **Auto‑report**: export a PDF with tables + charts for train/test/full.

---

## Reproducibility
- Pin Python deps (`pandas`, `numpy`).  
- Persist run args and inferred `BPD/BPY` in `meta.json`.  
- State the execution/fee timing convention in reports.  
- Keep the same tz (`America/New_York`) across runs.

---

## Disclaimer
For research/educational purposes only. No investment advice.
