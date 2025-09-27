#!/usr/bin/env python3
# Residual L/S 4H Backtest (No Look-Ahead, Open-Rebalance)
# Usage: python strategy_resid_ls_4h.py
import zipfile, io, pandas as pd, numpy as np, matplotlib.pyplot as plt, os, math
from datetime import datetime

DATA_ZIP = "merged_1m_2023-08-01_2025-08-01.zip"
CSV_IN_ZIP = "merged_1m_2023-08-01_2025-08-01.csv"
OUTDIR = "strategy_outputs"
FEE = 0.0003
ROLL_WIN_BETA = 100
ROLL_WIN_Z = 200
AVAIL_WIN = 60
AVAIL_THRESH = 0.8
GROSS_LEV = 1.0

os.makedirs(OUTDIR, exist_ok=True)

def load_data():
    with zipfile.ZipFile(DATA_ZIP) as z:
        with z.open(CSV_IN_ZIP) as f:
            df = pd.read_csv(f)
    df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
    df = df.set_index('open_time').sort_index()
    price1m = df.select_dtypes(include=[np.number]).copy()
    price4h = price1m.resample("4H").last()
    return price4h

def prepare(price4h):
    r4h = price4h.pct_change()
    btc_aliases = ['BTC_USD','BTCUSDT','BTC-USD']
    btc_col = None
    for c in price4h.columns:
        if c in btc_aliases: btc_col = c; break
    if btc_col is None:
        for c in price4h.columns:
            if 'BTC' in c and ('USD' in c or 'USDT' in c or '-' in c):
                btc_col = c; break
    if btc_col is None:
        raise ValueError("BTC column not found")
    symbols = [c for c in price4h.columns if c != btc_col]
    avail_ratio = price4h.notna().rolling(AVAIL_WIN, min_periods=max(10, AVAIL_WIN//3)).mean()
    tradable_mask = (avail_ratio >= AVAIL_THRESH)
    return r4h, btc_col, symbols, tradable_mask

def rolling_beta(y, x, win):
    cov = y.rolling(win, min_periods=max(10, win//2)).cov(x)
    var = x.rolling(win, min_periods=max(10, win//2)).var()
    return cov/var

def build_signals(r4h, btc_col, symbols):
    r_btc = r4h[btc_col]
    betas = pd.DataFrame(index=r4h.index, columns=symbols, dtype=float)
    for s in symbols:
        betas[s] = rolling_beta(r4h[s], r_btc, ROLL_WIN_BETA)
    resid = pd.DataFrame(index=r4h.index, columns=symbols, dtype=float)
    for s in symbols:
        resid[s] = r4h[s] - betas[s] * r_btc
    resid_mean = resid.rolling(ROLL_WIN_Z, min_periods=max(10, ROLL_WIN_Z//4)).mean()
    resid_std  = resid.rolling(ROLL_WIN_Z, min_periods=max(10, ROLL_WIN_Z//4)).std()
    z = (resid - resid_mean) / resid_std
    return z.shift(1)  # no look-ahead

def make_weights(z_lag, tradable_mask, symbols):
    def one_bar(row_z, row_tradable):
        valid = row_tradable.index[row_tradable.fillna(False)]
        row = row_z.loc[valid].dropna()
        out = pd.Series(0.0, index=symbols)
        if row.empty: return out
        s = -row
        s_dm = s - s.mean()
        if np.allclose(s_dm.values, 0) or s_dm.abs().sum()==0: return out
        w = (s_dm / s_dm.abs().sum()) * GROSS_LEV
        out.loc[w.index] = w
        return out
    W = []
    for t in z_lag.index:
        W.append(one_bar(z_lag.loc[t], tradable_mask.loc[t] if t in tradable_mask.index else pd.Series(False, index=symbols)))
    W = pd.DataFrame(W, index=z_lag.index, columns=symbols).fillna(0.0)
    return W

def ann_factor(): return 6*365
def cum_from_rets(r): return (1.0 + r.fillna(0)).cumprod()
def max_drawdown(cum):
    peak = cum.cummax(); dd = cum/peak - 1.0; return float(dd.min())
def metrics(r):
    af = ann_factor(); mu, sd = r.mean(), r.std()
    ann_ret = (1+mu)**af - 1 if not np.isnan(mu) else np.nan
    ann_vol = sd * math.sqrt(af) if not np.isnan(sd) else np.nan
    sharpe = (mu/sd)*math.sqrt(af) if (sd is not None and sd>0) else np.nan
    cum = cum_from_rets(r); mdd = max_drawdown(cum)
    mar = (ann_ret/abs(mdd)) if (mdd is not None and mdd<0) else np.nan
    month_ret = r.groupby([r.index.year, r.index.month]).apply(lambda x: (1+x).prod()-1)
    win_rate = (month_ret>0).mean() if len(month_ret)>0 else np.nan
    return dict(Annualized_Return=ann_ret, Annualized_Vol=ann_vol, Sharpe=sharpe, Max_Drawdown=mdd, MAR=mar, Monthly_Win_Rate=win_rate)

def run():
    price4h = load_data()
    r4h, btc_col, symbols, tradable_mask = prepare(price4h)
    z_lag = build_signals(r4h, btc_col, symbols)
    W = make_weights(z_lag, tradable_mask[symbols], symbols)
    ret_port = (W * r4h[symbols]).sum(axis=1)
    turnover = (W - W.shift(1).fillna(0.0)).abs().sum(axis=1)
    costs = FEE * turnover
    ret_net = ret_port - costs
    # dynamic EW benchmark
    ew_ret = []
    for t in r4h.index:
        tradables = tradable_mask[symbols].loc[t].index[tradable_mask[symbols].loc[t].fillna(False)] if t in tradable_mask.index else []
        if len(tradables)==0: ew_ret.append(0.0); continue
        r_t = r4h.loc[t, tradables].dropna()
        ew_ret.append(0.0 if r_t.empty else r_t.mean())
    ew_ret = pd.Series(ew_ret, index=r4h.index)
    # save outputs
    out = pd.DataFrame({"strategy_net":ret_net, "strategy_gross":ret_port, "turnover":turnover, "costs":costs, "ew_bh":ew_ret.reindex(ret_net.index).fillna(0.0)})
    out.to_csv(os.path.join(OUTDIR, "returns_timeseries.csv"))
    # metrics
    valid_idx = ret_net.dropna().index
    split_point = valid_idx[int(len(valid_idx)*0.5)]
    full_m = metrics(ret_net); bh_m = metrics(ew_ret.reindex(ret_net.index).fillna(0.0))
    trn_m  = metrics(ret_net.loc[:split_point]); tst_m = metrics(ret_net.loc[split_point:])
    pd.DataFrame({"Strategy_Full":full_m, "Benchmark_Full":bh_m, "Strategy_Train":trn_m, "Strategy_Test":tst_m}).to_csv(os.path.join(OUTDIR,"metrics_summary.csv"))
    # plots
    cum_s = cum_from_rets(ret_net); cum_bh = cum_from_rets(ew_ret.reindex(ret_net.index).fillna(0.0))
    plt.figure(figsize=(10,5)); plt.plot(cum_s.index,cum_s.values,label="Strategy (Net)"); plt.plot(cum_bh.index,cum_bh.values,label="EW Buy&Hold"); plt.legend(); plt.title("Equity Curves"); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"equity_curves.png")); plt.close()
    plt.figure(figsize=(10,4)); plt.plot(turnover.index, turnover.values); plt.title("Turnover (per 4H)"); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"turnover.png")); plt.close()
    print("Done. Outputs saved to:", OUTDIR)

if __name__ == "__main__":
    run()
