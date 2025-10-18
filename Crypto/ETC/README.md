# Residual Z-Score Mean Reversion Strategy (Crypto)

> Trade the *residual* of a coin relative to a reference (default: BTC) via rolling log-price regression. Signals come from the residual’s **Z-score**; execution is strictly **no look-ahead** (both statistics and positions are lagged by one bar).

---

## 1) Overview
- **Type**: Statistical arbitrage / mean reversion on residuals
- **Idea**: Remove the common trend between the target coin and BTC, and trade the **residual** back toward its mean
- **Bar frequency**: 1 hour (adjustable; update annualization factors accordingly)
- **Universe**: Any single `COIN` (e.g., `ETCUSDT`), reference `REF` defaults to `BTC_USD`
- **Input**: `crypto_data.zip` containing `crypto_data.csv`

---

## 2) Data & Preprocessing
- Resample to **1H** using `last`.
- Forward-fill missing bars up to **3 hours** to avoid long stale carries.
- Work in **log prices** to stabilize variance and give coefficients an elasticity interpretation.
- Ensure prices are strictly positive (e.g., `clip(lower=1e-12)`).

---

## 3) Signal Generation (all rolling stats use **t−1** data; shifted to avoid look-ahead)
We regress log prices of the coin on the reference over a rolling window \(W\):
\[
\ln P_{c,t} = \alpha_t + \beta_t \ln P_{r,t} + \varepsilon_t
\]

- Rolling estimates (**all shifted by 1 bar**):
  \[
  \beta_t = \frac{\operatorname{Cov}(\ln P_c,\ln P_r)}{\operatorname{Var}(\ln P_r)},\quad
  \alpha_t = \mathbb{E}[\ln P_c] - \beta_t\,\mathbb{E}[\ln P_r]
  \]
- Residual and Z-score:
  \[
  \text{spread}_t = \ln P_{c,t} - (\alpha_t + \beta_t \ln P_{r,t}),\quad
  z_t = \frac{\text{spread}_t - \mu_t}{\sigma_t}
  \]
  where \(\mu_t, \sigma_t\) are rolling mean/std of the residual, also **shifted by 1**.

### Trading thresholds
- Default **entry threshold**: \( z_{\text{th}} = 1.0 \)
- Exit uses the **zero** crossing of \(z\).

---

## 4) Trading Rules (“crossing” method; orders execute **next bar**)
**Entries (next bar execution):**
- **Long entry** when z crosses **below** \(-z_{\text{th}}\) from inside the band:  
  \( z_{t-1} \ge -z_{\text{th}} \) **and** \( z_t < -z_{\text{th}} \).
- **Short entry** when z crosses **above** \(+z_{\text{th}}\) from inside the band:  
  \( z_{t-1} \le +z_{\text{th}} \) **and** \( z_t > +z_{\text{th}} \).

**Exits (next bar execution):**
- Close any position when z **crosses 0**:  
  \( (z_{t-1} \le 0 \land z_t>0) \) or \( (z_{t-1}\ge 0 \land z_t<0) \).

**No look-ahead enforcement:** signal is formed at bar \(t\), the position becomes active at \(t+1\): in code `pos = pos_raw.shift(1)`.

---

## 5) PnL and Fees
Single-leg version (default):
\[
r_t^{\text{strat}} = \text{pos}_t \cdot \frac{P_t-P_{t-1}}{P_{t-1}} \, - \, \text{fee}\cdot |\Delta \text{pos}_t|
\]
- Default **fee** = **0.03% per side** (0.0003). A flip (+1 ↔ −1) counts as two changes.

Optional **hedged** version: trade residual return \(r_t^{\text{coin}} - \beta_t r_t^{\text{ref}}\) and include two legs of fees.

---

## 6) Visualization Conventions
- **Price** (left axis): **black**
- **Z-score** (right axis): **blue**
- **Markers on top of lines** (higher `zorder`):
  - **Long entry**: green triangle **▲**
  - **Short entry**: red inverted triangle **▼**
  - **Long exit**: green hollow circle **○**
  - **Short exit**: red hollow circle **○**
- Threshold lines: blue dashed at ±\(z_{\text{th}}\); gray dashed at 0.

---

## 7) How to Run

### 7.1 Single-coin backtest + metrics + equity curve
Run the main script (parameters are embedded at the top of `__main__`):
```bash
python residual_zscore_strategy.py
```
Default embedded params:
- `ZIP_PATH="/mnt/data/crypto_data.zip"`  
- `INNER_CSV="crypto_data.csv"`  
- `COIN="ETCUSDT"`, `REF="BTC_USD"`  
- `WINDOW=168` (7 days), `Z_TH=1.0`, `FEE=0.0003`

**Outputs**
- `metrics_summary.csv`: one table with **Strategy/Buy&Hold × Train/Test/Full** metrics  
  (Annualized Return, Sharpe, Max Drawdown, MAR, Monthly Win Rate, Num Periods)  
- `equity_curve.png`: Strategy vs Buy&Hold (with a 70/30 time split marker)

### 7.2 Pick **two days per month** and plot **48h signal charts**
The batch plotting script will:
1) Build **executed** entry/exit events (from previous → current position).  
2) For each calendar month, count **daily** event totals and pick the **top 2 days**.  
3) For each selected day, plot a **48h** window starting **00:00 UTC** with the conventions above.  
**Outputs**
- `monthly_signal_days_*.csv`: month, selected dates, event counts, image paths  
- `monthly_signal_charts_*.zip`: all images zipped

> You can switch to fixed days (e.g., 1st & 15th), change the window (24h/72h), or loop over multiple coins by tweaking the script’s top-level params.

---

## 8) Parameters & Practical Tips
- `WINDOW` (hours): 96 reacts faster (noisier), 168 default, 336 smoother (slower).
- `Z_TH`: try 0.8–1.5 in a grid for robustness.
- `FEE`: stress test 3–15 bps per side to approximate slippage/fees.
- `ffill_limit`: keep ≤3 to avoid long stale carries; drop very long gaps.

**Annualization** (1H bars): ~**8760** bars/year.

---

## 9) Metrics
- **Annualized Return**: from cumulative equity over the sample annualized by length.
- **Sharpe**: \( \frac{\mathbb{E}[r]}{\sigma[r]} \sqrt{8760} \) for hourly data.
- **Max Drawdown**: min of equity / running max − 1.
- **MAR**: Annualized Return / |Max Drawdown|.
- **Monthly Win Rate**: fraction of months with positive compounded return.

---

## 10) Robustness & Caveats
- Two layers of **no look-ahead**: shifted rolling stats + next-bar execution.
- **Data quality**: ensure positive prices, review giant jumps, handle long gaps.
- **Overfitting**: use parameter grids & out-of-sample (train/test) checks; vary `REF`.
- **Portfolio**: pre-select coins by train-period robustness; equal-weight or |z|-weighted; consider net-β constraints if hedged.

---

## 11) Risks
- **Mean reversion risk**: structural changes can invalidate the residual process.
- **Execution risk**: slippage, funding, limits are simplified as fees.
- **Data risk**: exchange/contract differences; stale data.
- **Parameter risk**: overfitting to a specific sample period.

---

## 12) Extensions
- **Hedged residual** trading in production (use \(\beta_t\) on the fly).
- **Regime** detection (volatility/correlation states).
- **Exit rules** (\(\pm z_{\text{exit}}\), time stops, profit caps).
- **Multi-frequency** confirmation (15m/1H/4H).
- **Adaptive thresholds** based on volatility/liquidity.

---

## 13) Minimal Config Snippet
```python
ZIP_PATH  = "/mnt/data/crypto_data.zip"
INNER_CSV = "crypto_data.csv"
COIN, REF = "ETCUSDT", "BTC_USD"
WINDOW, Z_TH, FEE = 168, 1.0, 0.0003
HOURS_WIN = 48  # 2 days per month plotting window
```
