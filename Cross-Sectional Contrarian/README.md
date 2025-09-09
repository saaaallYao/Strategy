# T+1 Intraday Contrarian Basket (with Optional ETF Hedge)

> **TL;DR (中文摘要)**：本策略在 **T 日收盘后** 按日内“14:00→14:55 回撤 + 14:55→14:58 速度”的横截面信号做 **逆向选股**，于 **T+1 开盘** 等权进场、**T+2 开盘** 统一平仓；可选用 **510300 ETF** 做滚动 20 日 β 对冲。默认手续费 **0.03%/边**，输出 Sharpe、年化、回撤、MAR、月胜率等指标，并生成 PDF 图表与 CSV。

## Overview

This repository contains a **T+1 intraday contrarian** equity basket strategy designed for liquid China A-shares (or any market with similar microstructure and minute bars). The strategy converts **late‑day intraday behavior** into next‑day open-to-open trades and can **optionally hedge** against market beta using a rolling estimate to an index ETF (e.g., **510300**). It is built for **reproducibility**, includes **no look‑ahead bias** in signal formation, and ships with a simple **CLI** + **PDF** report.

## Strategy Description

### Core Concept
- **Cross‑sectional contrarian**: Buy names with **late‑day weakness** relative to peers.
- **Next‑day execution**: Signals formed after the close; enter **T+1 open**, exit **T+2 open**.
- **Basket construction**: Equal‑weight the day’s selected names with **K_min…K_max** controls.
- **Optional hedge**: Rolling 20‑day beta to an index ETF with capping.

### Intraday Features (per day per symbol)
Let `P_t(hh:mm)` denote the minute close at `hh:mm` on trade day *t*; `O_{t}(09:31)` the 09:31 open.
- **55‑min pullback**:  
  \\\( \mathrm{pm\_ret\_55m}_t = \frac{P_t(14{:}55)}{P_t(14{:}00)} - 1 \\\)
- **2‑min speed**:  
  \\\( \mathrm{pm\_speed\_2m}_t = \frac{P_t(14{:}58)}{P_t(14{:}55)} - 1 \\\)
- **Liquidity filter (example)**: volume ratio vs. 20‑day median (implementation detail in code).
- **Cross‑sectional z‑score**: rank/standardize intraday features within day *t* (e.g., `z_pm`).

### Signal & Selection
- **Candidate filter (default)**:  
  `z_pm < z_threshold` **AND** `vol_ratio ≥ vol_ratio_min` **AND** `pm_ret_55m ≥ pm_ret_floor`  
- **Pick K**: sort by `z_pm` (most negative first), take `K_dyn = clip(len(cand), K_min, K_max)`.
- **Cooldown**: optional **per‑symbol cooldown** (in days) to avoid rapid re‑entries.

### Trade Lifecycle
- **Entry**: at **T+1 09:31** open (`next_open`)  
- **Exit**: at **T+2 09:31** open (`exit_open`)  
- **P&L** per name: \\\( r = \frac{\text{exit\_open}}{\text{entry\_open}} - 1 - \text{ROUND\_TRIP} \\\)  
  Basket day return = **equal‑weighted mean** across selected names.

### Optional ETF Hedge
- Compute rolling **20‑day** β of basket daily returns vs. ETF daily returns.
- **Cap β** in `[0, 0.6]` (default), subtract `β * ETF_return` and hedge trading cost when applied.

## Files

- **`t1_contrarian_enhanced.py`**: end‑to‑end implementation (loading → signals → backtest → plots).
- **Outputs** (created under `--outdir`):  
  - `Report_EN.pdf`: Equity curves (Full/Train/Test), Hedge vs. Unhedged, trade frequency, markers.  
  - `daily_returns_{train|test|full}.csv` and `daily_returns_full_hedged.csv`.  
  - PNGs (if any) and intermediate CSVs for diagnostics.

> The script in this repo is derived from the uploaded version: *“T+1 Intraday Contrarian (Enhanced) — Reproducible Script”*.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pandas numpy matplotlib
```

## Data Requirements

- **Minute bars** per symbol in a ZIP: CSVs with at least:  
  `timestamp, open, high, low, close, volume`
- **Conventions** (default China A-share session):  
  - Use **minute close** at **14:00, 14:55, 14:58** for features.  
  - Use **09:31 open** as the “open” for next‑day entry/exit settlement.  
- **ETF series** for hedge: minute data aggregated to **daily open/close** → compute daily returns.

> If your exchange timing differs, adjust the `*_hhmm` stamps accordingly in the code.

## Usage

```bash
python t1_contrarian_enhanced.py --zip /path/to/minute_data.zip --outdir ./outputs
```

Optional flags (add or modify in code / argparse as needed):
- `--start 2023-01-01 --end 2025-06-30`
- Universe filters, liquidity thresholds, etc.

## Backtest Setup

- **Train / Test / Full** splits are produced for side‑by‑side comparison in `Report_EN.pdf`.
- **Costs**: `FEE = 0.0003` per side (→ `ROUND_TRIP = 0.0006`), applied at each entry/exit.
- **Rebalancing**: equal‑weight across same‑day picks; hedge β updated **daily** using rolling window.

## Key Parameters (defaults in code)

| Parameter | Meaning | Default |
|---|---|---|
| `FEE` | Per‑side fee | `0.0003` |
| `z_threshold` | Max z to qualify (more negative = weaker) | `-0.5` |
| `K_min, K_max` | Basket size bounds per day | `3, 8` |
| `vol_ratio_min` | Liquidity/volume screen | `0.8` |
| `pm_ret_floor` | Avoid extreme crashes | `-0.04` |
| `cooldown_days` | Min days before re‑entry | `1` |
| `beta_window` | Rolling days for ETF β | `20` |
| `beta_cap` | Clip β in hedge | `[0, 0.6]` |

## Performance Metrics

Computed from **daily basket returns** (unhedged & hedged):
- **Annualized Return**: \\\( (1+\bar r)^{252} - 1 \\\)
- **Annualized Volatility**: \\\( \sigma \sqrt{252} \\\)
- **Sharpe**: \\\( \bar r/\sigma \sqrt{252} \\\) (rf≈0)
- **Max Drawdown**: min over equity curve
- **MAR**: Annualized Return / \|MaxDD\|
- **Win Rate (Monthly)**: % of months with positive compounded return

All series and equity curves are saved for reproducibility.

## No‑Look‑Ahead & Data Alignment

This project avoids look‑ahead bias by:
1. **Signal timing**: All intraday features are from **T day**; trades start **T+1 open** and exit **T+2 open**.  
2. **Price usage**: `next_open` / `exit_open` are **shifted forward** and used only for settlement, not selection.  
3. **Rolling stats**: Any rolling/median features should be `shift(1)` if you require “as‑of 14:58” strictness.

**Important alignment note**: `pnl_date` should use **trading‑day offsets** to avoid weekend/holiday drift. Prefer:
```python
from pandas.tseries.offsets import BDay
t["entry_date"] = pd.to_datetime(t["signal_date"]) + BDay(1)
t["exit_date"]  = pd.to_datetime(t["signal_date"]) + BDay(2)
t["pnl_date"]   = t["exit_date"]
```
and align ETF returns by `reindex` **without** unconditional `fillna(0.0)` to prevent silent misalignment.

## Signals & Logic (Pseudo‑Code)

```text
For each trade day t:
  Build universe U_t with liquidity & data availability screens
  Compute pm_ret_55m(t), pm_speed_2m(t)
  Standardize within day → z_pm
  Candidates C_t = { i in U_t | z_pm_i < z_threshold
                                and vol_ratio_i ≥ vol_ratio_min
                                and pm_ret_55m_i ≥ pm_ret_floor }
  K = clip(|C_t|, K_min, K_max)
  Picks S_t = bottom-K by z_pm, excluding symbols in cooldown
  Enter S_t at t+1 open (equal-weight)
  Exit S_t at t+2 open
  Basket daily r_t+2 = mean( (exit/entry - 1) - ROUND_TRIP )
  Optional: Hedge r_t+2 by subtracting β_t * r_ETF,t+2 (β from last 20 days, capped)
```

## Validation & Robustness

- **Train/Test** split with identical rules.  
- **Sensitivity**: try `z_threshold`, `K` bounds, `pm_ret_floor`, `vol_ratio_min`.  
- **Liquidity**: tighten `vol_ratio_min` for conservative execution.  
- **Hedge**: compare **hedged vs. unhedged** Sharpe and drawdown profiles.

## Known Limitations / To‑Dos

- **Trading‑day calendar**: ensure proper holiday/weekend handling in P&L dating.
- **Volume feature timing**: if using full‑day volume medians, add `shift(1)` for strict “as‑of close” information set.
- **Survivorship**: if your ZIP excludes delisted names, be aware of survivorship bias.
- **Low‑correlation basket (extension)**: add a diversification penalty (e.g., greedy selection with rolling corr matrix).

## How to Extend

- **Low‑Correlation Selector**: penalize correlated picks via rolling corr or sector caps.
- **Regime Filters**: skip days around major announcements or limit‑down clusters.
- **Exit Variants**: T+1 close, trailing stops, or time‑varying holding horizon.
- **Execution Model**: add slippage by spread/volume; model queue priority.

## Results Snapshot

See `outputs/Report_EN.pdf` for:
- Equity curves (**Full/Train/Test**), ETF benchmark, and **hedged** curve
- Trade frequency bar chart
- Sample **buy/sell markers** (green ▲ at T+1 open, red ▼ at T+2 open)

## Disclaimer

This strategy and code are for **research and educational** purposes only. Markets involve risk; past performance does not guarantee future results. Validate with your **own data, costs, and execution constraints** before live deployment.
