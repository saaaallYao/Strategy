# Crypto Relative-Momentum Top-N Neutral Strategy

## Overview
This repository implements a **relative momentum + rotation with hedge** strategy for crypto. On each 30-minute bar, it **goes long the Top-N altcoins** by relative momentum versus BTC and **shorts BTC** for a market‑neutral exposure. The system uses **entry/exit thresholds**, a **cooldown**, and optional **volatility targeting / leverage cap**.

## Strategy Description

### Core Concept
- **Relative to BTC:** use the price ratio \( R_{i,t} = P_{i,t} / P_{\text{BTC},t} \) to remove broad market beta.
- **8‑bar Relative Momentum:** signal line \( S_{i,t} = R_{i,t}/R_{i,t-8} - 1 \). To avoid look‑ahead, we trade \( \tilde S_{i,t} = S_{i,t-1} \).
- **Top‑N Selection:** rank all assets by \( \tilde S_{i,t} \) each bar and select **Top‑3**.
- **Market‑Neutral:** long the selected set, short BTC with equal notional.
- **Risk Control:** inverse‑volatility weights, optional annualized **target volatility = 0.22** with **max leverage = 1.6**.

### Default Parameters
```json
{
  "freq": "30min",
  "rel_momentum_lookback": 8,
  "spread_vol_lookback": 48,
  "entry_threshold": 0.0015,
  "decay_threshold": 0.0004,
  "top_n": 3,
  "target_vol": 0.22,
  "max_leverage": 1.6,
  "cooldown_bars": 4
}
```

### Trading Rules

**Entry** (green ▲ on charts)  
At time \( t \), asset \( i \) enters **if all** hold:
1) in **Top‑3** by \( \tilde S_{i,t} \);  
2) \( \tilde S_{i,t} \ge +0.0015 \) (entry threshold);  
3) **not** in cooldown (must wait 4 bars after last exit);  
and a position transitions from **flat → long**.

**Exit** (two reasons, distinct markers on charts)  
- **Decay Exit (red ▼):** while in position, if \( \tilde S_{i,t} < -0.0004 \).  
- **Deselected Exit (purple ×, optional):** while in position, if the asset **drops out of Top‑N**.  
  (You can disable this as a hard exit and keep it as a visual cue only.)

**Weights & Hedge**  
- Among selected names, allocate by **inverse volatility** of the **alt‑BTC spread** (48‑bar window).  
- Hedge with a BTC short of (approximately) equal notional to keep the book close to market‑neutral.  
- Optionally scale to **target volatility** (0.22 annualized) with **max leverage** 1.6.

## File Structure
- `strategy.py`: core implementation (signals, weights, backtest, chart export)  
- `run_strategy.py`: example entry point for loading params/data and producing artifacts  
- `config.json` (optional): parameter overrides (CLI args can supersede this)  
- `data/crypto_data.csv`: input data (at minimum `open_time` + price columns like `BTC_USD`, `ETH_USD`, ...)

> If your actual entry script has a different name, update the references accordingly.

## Usage

### Setup
```bash
pip install pandas numpy matplotlib
```

### Run
```bash
python run_strategy.py   --data ./data/crypto_data.csv   --freq 30min   --rel_momentum_lookback 8   --spread_vol_lookback 48   --entry_threshold 0.0015   --decay_threshold 0.0004   --top_n 3   --target_vol 0.22   --max_leverage 1.6   --cooldown_bars 4
```

### Data Requirements
- Time column: `open_time` (parsable timestamp).  
- Price columns: `BTC_USD` plus multiple alts (e.g., `ETH_USD`, `MATICUSDT`, …).  
- Higher‑frequency data (e.g., 1‑minute) is fine; the script resamples to 30‑minute bars internally.

## Performance Metrics
Backtests report (per Train/Test/Full splits):
- **Annualized Gross Return**
- **Sharpe** (annualized at 30‑min sampling)
- **Max Drawdown**
- **MAR Ratio** (Return / MaxDD)
- **Win Rate (Monthly)**

The benchmark can be configured (e.g., **BTC‑only**). Keep frequency and metric conventions consistent.

## Outputs
- `./results/weights.csv`: per‑bar asset weights (including BTC hedge)  
- `./results/trades.csv`: trade ledger (timestamp, asset, side, exit reason, holding period, PnL, etc.)  
- `./results/metrics_*.json`: metrics for train/test/full  
- `./charts/`:  
  - **Signal charts** (48h / 7d per asset): price (blue), signal line (orange), thresholds (green/red dashed)  
  - **Markers:** Entry ▲ green; Exit ▼ red (decay); Exit × purple (deselected, optional)  
  - **Optional:** position shading/step line, weight evolution, signal histograms, turnover bars

## Configuration
- Parameters can be changed via CLI or `config.json` (CLI takes precedence).  
- Common knobs: `top_n`, thresholds, cooldown, target vol / leverage, fees, benchmark type.  
- Add/remove assets by adding/removing columns in `crypto_data.csv`—the script auto‑detects available tickers.

## Risks
- **Regime Risk:** relative momentum can underperform in some regimes.  
- **Basis Risk:** long alts + short BTC is **not** a perfect hedge.  
- **Turnover & Cost:** oscillation near thresholds may increase churn & costs.  
- **Data/Execution:** data quality, slippage, and fills are not fully modeled.

## Roadmap
- Adaptive thresholds (e.g., vol/liquidity‑aware)  
- Regime detection / dynamic risk switch  
- Fixed vs. adaptive `Top‑N` comparison  
- Multi‑horizon fusion, session filters  
- Paper‑trading/live connector with guardrails

## Disclaimer
For research/education only. Not investment advice. Past performance is not indicative of future results.
