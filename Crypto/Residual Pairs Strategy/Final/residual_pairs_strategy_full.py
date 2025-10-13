#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOTICE: MATICUSDT changed to POLUSDT on 2025-01-16 in source data
Residual Long/Short Crypto Selection (Market-Neutral) — Full Runnable Script
- Loads 1-min merged data (expects inner CSV with 'open_time' and columns like BTC_USD, SOL_USD, etc.)
- Resamples to 1H, builds signals with strict anti-lookahead (shift(1)), applies fees, computes Train/Test/Full metrics
- Outputs: performance CSVs, equity curves, trade markers (green buy / red sell), monthly trade events, and weights/returns CSVs.

Usage:
    python residual_pairs_strategy_full.py
"""

import os, zipfile, math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Parameters
# --------------------------
PARAMS = dict(
    data_zip = "crypto_data.zip",
    inner_csv = "crypto_data.csv",
    resample_rule = "1H",
    lookback_days = 20,     # rolling beta window in days
    zwin_days     = 45,     # Z-score window on cumulative residual in days
    K_per_side    = 3,      # number of long and short positions
    z_threshold   = 0.8,    # entry threshold on |Z|
    fee_per_side  = 0.0003, # 0.03% per side
    train_end     = "2024-07-31 23:59:59+00:00",
    test_start    = "2024-08-01 00:00:00+00:00",
    example_sym   = "SOL_USD" # used for trade markers; falls back to first alt if missing
)

# --------------------------
# Helpers
# --------------------------
def true_equal_weight_bh(prices_df, cols):
    """True EW buy-and-hold (no rebalancing): equal-weight at t0, then drift."""
    P = prices_df[cols].dropna()
    if P.empty:
        return pd.Series(dtype=float)
    base = P.iloc[0]
    rel  = P.divide(base)
    ew_equity = (rel * (1.0/len(cols))).sum(axis=1)
    return ew_equity

def perf_table_from_returns(ret_series, bars_per_year):
    """Metrics from linear returns (not log)."""
    rs = ret_series.dropna()
    if rs.empty:
        return dict(Annualized_Return=np.nan, Sharpe=np.nan, Max_Drawdown=np.nan,
                    MAR=np.nan, Monthly_Win_Rate=np.nan, Bars=0)
    ann_ret = (1+rs).prod() ** (bars_per_year/len(rs)) - 1
    ann_vol = rs.std() * math.sqrt(bars_per_year)
    sharpe  = ann_ret/ann_vol if ann_vol>0 else np.nan
    eq = (1+rs).cumprod()
    dd = eq/eq.cummax() - 1
    maxdd = dd.min()
    mar = (ann_ret/abs(maxdd)) if maxdd<0 else np.nan
    mret = eq.resample("M").last().pct_change().dropna()
    win  = (mret>0).mean() if len(mret)>0 else np.nan
    return dict(Annualized_Return=float(ann_ret), Sharpe=float(sharpe), Max_Drawdown=float(maxdd),
                MAR=float(mar), Monthly_Win_Rate=float(win), Bars=int(len(rs)))

# --------------------------
# Main
# --------------------------
def main():
    p = PARAMS

    # Load zip/csv
    zf = zipfile.ZipFile(p["data_zip"])
    with zf.open(p["inner_csv"]) as f:
        df = pd.read_csv(f, parse_dates=["open_time"])
    df = (df.rename(columns={"open_time":"dt"})
            .set_index("dt")
            .sort_index()
            .ffill())

    # Universe
    universe = [c for c in df.columns if c != "BTC_USD"]
    if not universe:
        raise RuntimeError("No alt symbols found in data (columns except BTC_USD).")

    # Resample to 1H (last price)
    prices   = df[["BTC_USD"] + universe].resample(p["resample_rule"]).last().dropna()
    rets_lin = prices.pct_change()          # linear returns for PnL
    rets_log = np.log(prices).diff()        # log returns for beta stability
    btc_r_lin = rets_lin["BTC_USD"]
    btc_r_log = rets_log["BTC_USD"]

    # Rolling beta via corr*std ratio (log domain), then shift(1) to avoid lookahead
    lb = int(p["lookback_days"]*24)  # hours
    zw = int(p["zwin_days"]*24)      # hours
    corr    = rets_log[universe].rolling(lb).corr(btc_r_log)
    std_alt = rets_log[universe].rolling(lb).std()
    std_btc = btc_r_log.rolling(lb).std()
    beta = (corr * std_alt).div(std_btc, axis=0).shift(1).clip(-5,5)

    # Residual log-returns -> cumulative residual (spread)
    resid_log = rets_log[universe] - beta.mul(btc_r_log, axis=0)
    spread = resid_log.cumsum()
    m = spread.rolling(zw, min_periods=max(zw//4,1)).mean().shift(1)
    s = spread.rolling(zw, min_periods=max(zw//4,1)).std().shift(1)
    z = (spread - m) / s

    # Build weights with next-bar execution (anti-lookahead)
    K  = int(p["K_per_side"])
    zt = float(p["z_threshold"])
    Z  = z.values
    idx = z.index
    cols = list(z.columns)
    W = np.zeros((Z.shape[0], Z.shape[1]+1))  # last col BTC hedge

    for i in range(Z.shape[0]):
        row = Z[i,:]
        valid = np.isfinite(row)
        if valid.any():
            rv = row[valid]; ci = np.where(valid)[0]
            long_idx  = ci[rv <= -zt]
            short_idx = ci[rv >=  zt]
            # cap by K most extreme on each side
            if long_idx.size > K:
                pick = np.argpartition(row[long_idx], K-1)[:K]
                long_idx = long_idx[pick]
            if short_idx.size > K:
                pick = np.argpartition(-row[short_idx], K-1)[:K]
                short_idx = short_idx[pick]
            if long_idx.size>0:
                W[i, long_idx] = 0.5/long_idx.size
            if short_idx.size>0:
                W[i, short_idx] = -0.5/short_idx.size
        # BTC hedge uses previous beta (anti-lookahead)
        b = beta.iloc[i-1,:].values if i>0 else np.zeros(Z.shape[1])
        b = np.nan_to_num(b, nan=0.0)
        W[i, Z.shape[1]] = -np.sum(W[i, :Z.shape[1]] * b)

    W = (pd.DataFrame(W, index=idx, columns=cols+["BTC_HEDGE"])
            .shift(1)          # next bar execution
            .fillna(0.0))

    # Portfolio returns + fees
    fee = float(p["fee_per_side"])
    port_ret_gross = (W[universe].mul(rets_lin[universe]).sum(axis=1) +
                      W["BTC_HEDGE"] * btc_r_lin)
    turnover = W.diff().abs().sum(axis=1).fillna(0.0)
    port_ret_net = port_ret_gross - fee * turnover

    # Equity
    equity = (1+port_ret_net).cumprod()
    equity.to_csv("/mnt/data/strategy_equity_1h_fullrun.csv")
    port_ret_net.to_csv("/mnt/data/strategy_returns_1h_fullrun.csv")
    W.to_csv("/mnt/data/strategy_weights_1h_fullrun.csv")

    # Benchmarks
    ew_bh_equity = true_equal_weight_bh(prices, universe)
    ew_bh_rets   = ew_bh_equity.pct_change()
    btc_equity   = (prices["BTC_USD"]/prices["BTC_USD"].iloc[0])
    btc_rets     = btc_equity.pct_change()

    # Metrics (Full / Train / Test)
    bpy = 365*24
    train_end  = pd.Timestamp(p["train_end"])
    test_start = pd.Timestamp(p["test_start"])
    mtr = (port_ret_net.index <= train_end)
    mte = (port_ret_net.index >= test_start)

    perf_train = pd.DataFrame({
        "Strategy(Net)":        perf_table_from_returns(port_ret_net[mtr], bpy),
        "EW B&A (No-Rebal)":    perf_table_from_returns(ew_bh_rets[mtr], bpy),
        "BTC B&H":              perf_table_from_returns(btc_rets[mtr], bpy)
    }).round(4).rename(columns={"EW B&A (No-Rebal)":"EW B&H (No-Rebal)"})

    perf_test  = pd.DataFrame({
        "Strategy(Net)":        perf_table_from_returns(port_ret_net[mte], bpy),
        "EW B&A (No-Rebal)":    perf_table_from_returns(ew_bh_rets[mte], bpy),
        "BTC B&H":              perf_table_from_returns(btc_rets[mte], bpy)
    }).round(4).rename(columns={"EW B&A (No-Rebal)":"EW B&H (No-Rebal)"})

    perf_full  = pd.DataFrame({
        "Strategy(Net)":        perf_table_from_returns(port_ret_net, bpy),
        "EW B&A (No-Rebal)":    perf_table_from_returns(ew_bh_rets, bpy),
        "BTC B&H":              perf_table_from_returns(btc_rets, bpy)
    }).round(4).rename(columns={"EW B&A (No-Rebal)":"EW B&H (No-Rebal)"})

    perf_train.to_csv("perf_train_fullrun.csv")
    perf_test.to_csv("perf_test_fullrun.csv")
    perf_full.to_csv("perf_full_fullrun.csv")

    # Figures
    # 1) Equity vs benchmarks
    plt.figure()
    equity.plot()
    ew_bh_equity.reindex_like(equity).plot()
    btc_equity.reindex_like(equity).plot()
    plt.legend(["Strategy (Net)","EW B&H (No-Rebal)","BTC B&H"])
    plt.title("Equity Curves (1H)")
    plt.xlabel("Time"); plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig("fig_equity_vs_bh_fullrun.png")
    plt.close()

    # 2) Trade markers for example symbol (explicit green/red)
    sym_ex = p["example_sym"] if p["example_sym"] in universe else universe[0]
    z_sym = z[sym_ex].dropna()
    sig = pd.Series(0, index=z_sym.index)
    sig[z_sym <= -zt] = 1
    sig[z_sym >=  zt] = -1
    sig = sig.shift(1).fillna(0)
    px = prices[sym_ex].reindex(sig.index).dropna()
    sig = sig.reindex(px.index).fillna(0)
    chg = sig.diff().fillna(0)
    buys  = px[chg > 0]
    sells = px[chg < 0]
    plt.figure()
    px.plot()
    plt.scatter(buys.index,  buys.values,  marker="^", color="green")
    plt.scatter(sells.index, sells.values, marker="v", color="red")
    plt.title(f"{sym_ex} Price with Trade Markers (1H)")
    plt.xlabel("Time"); plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("fig_trade_markers_fullrun.png")
    plt.close()

    # 3) Monthly trade events (any portfolio change)
    Wchg_events = (W.diff().abs().sum(axis=1) > 0).astype(int)
    monthly_trades = Wchg_events.resample("M").sum()
    monthly_trades.to_frame("events").to_csv("monthly_trade_events_fullrun.csv")
    plt.figure()
    monthly_trades.plot(kind="bar")
    plt.title("Monthly Trade Events (Any Portfolio Change)")
    plt.xlabel("Month"); plt.ylabel("Events")
    plt.tight_layout()
    plt.savefig("fig_monthly_trades_fullrun.png")
    plt.close()

    print("Artifacts saved:")
    print(" - strategy_equity_1h_fullrun.csv")
    print(" - strategy_returns_1h_fullrun.csv")
    print(" - strategy_weights_1h_fullrun.csv")
    print(" - perf_train_fullrun.csv, perf_test_fullrun.csv, perf_full_fullrun.csv")
    print(" - fig_equity_vs_bh_fullrun.png, fig_trade_markers_fullrun.png, fig_monthly_trades_fullrun.png")

if __name__ == "__main__":
    main()
