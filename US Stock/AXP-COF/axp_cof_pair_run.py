#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AXP/COF Statistical Arbitrage — runnable backtest
Usage:
  python axp_cof_pair_run.py --csv your_minute_data.csv --outdir ./axp_cof_outputs
The CSV must contain a datetime-like column and columns: AXP, COF (minute close prices).
"""
import argparse, os, math
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from datetime import time

# ---------------- Params (you may tweak) ----------------
WIN, Z_WIN = 180, 180            # OLS & zscore windows (in bars)
ENTRY, EXIT = 2.2, 0.6           # entry/exit thresholds on |z|
FEE = 0.0003                     # 0.03% per leg, charged on entry/exit
EOD_FLAT = time(15,55)           # flat before 15:55
LEV_MIN, LEV_MAX = 0.5, 2.0      # vol-target leverage cap (min,max)

# ---------------- Helpers ----------------
def detect_time_col(df):
    cand = [c for c in df.columns if any(k in c.lower() for k in ["datetime","timestamp","time","date"])]
    if not cand:
        raise ValueError("No datetime-like column found.")
    return cand[0]

def load_rth(csv_path):
    df = pd.read_csv(csv_path)
    tcol = detect_time_col(df)
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol).sort_index()
    # RTH 09:30–15:55
    df = df.loc[(df.index.time >= time(9,30)) & (df.index.time <= EOD_FLAT)]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def bars_geometry(idx):
    if len(idx)>2:
        md = pd.Series(idx).diff().dropna().median()
        bar_min = int(round(md.total_seconds()/60.0))
    else:
        bar_min = 1
    bpd = int(round(390/max(1,bar_min)))
    bpy = 252*bpd
    return bar_min, bpd, bpy

def rolling_alpha_beta(y, x, win):
    mY, mX = y.rolling(win).mean(), x.rolling(win).mean()
    vX = x.rolling(win).var()
    cXY = x.rolling(win).cov(y)
    beta = cXY / vX
    alpha = mY - beta*mX
    return alpha, beta

def zscore(s, win):
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - m) / sd

def run_pair_with_pos(y, x, win=WIN, z_win=Z_WIN, entry=ENTRY, exit=EXIT, fee=FEE):
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    a, b = rolling_alpha_beta(df["y"], df["x"], win)
    spread = df["y"] - (a + b*df["x"])
    z = zscore(spread, z_win).dropna()
    # align
    df = df.loc[z.index]; a, b = a.loc[z.index], b.loc[z.index]
    # state machine
    pos = pd.Series(0.0, index=z.index); state=0
    for t, zv in z.items():
        if state==0:
            if zv >= entry: state=-1
            elif zv <= -entry: state=+1
        elif state==+1 and abs(zv)<=exit: state=0
        elif state==-1 and abs(zv)<=exit: state=0
        if t.time() >= EOD_FLAT: state=0
        pos.loc[t]=state
    # returns
    ry = df["y"].pct_change().fillna(0.0); rx = df["x"].pct_change().fillna(0.0)
    leg = ry - b.shift(1)*rx                # no look-ahead
    gross = pos.shift(1).fillna(0.0)*leg    # apply t-1 position
    dpos = pos.diff().abs().fillna(0.0)
    net  = gross - dpos*(2*fee)             # two legs cost
    info = {
        "pos": pos,
        "entL": (pos.diff()==+1),
        "exL":  (pos.diff()==-1) & (pos.shift(1)==+1),
        "entS": (pos.diff()==-1),
        "exS":  (pos.diff()==+1) & (pos.shift(1)==-1)
    }
    return net.dropna(), info

def vol_target(r, bpd, lev_cap=(LEV_MIN, LEV_MAX)):
    win = 21*bpd
    rolling = r.rolling(win).std()
    target = rolling.median()
    scale = (target/rolling).clip(lower=lev_cap[0], upper=lev_cap[1])
    return (scale.shift(1).fillna(1.0)*r).dropna()

def metrics(r, bpy):
    if len(r)<10 or r.std()==0:
        return {"Sharpe":np.nan,"AnnRet":np.nan,"MaxDD":np.nan,"MAR":np.nan,"MonthlyWin":np.nan}
    eq = (1+r).cumprod()
    ann = (1+r).prod()**(bpy/len(r)) - 1
    shp = r.mean()/r.std()*math.sqrt(bpy)
    dd = (eq.cummax()-eq)/eq.cummax(); mdd=float(dd.max())
    mar = ann/mdd if mdd>0 else np.nan
    win = float(((1+r).resample("M").prod()-1 > 0).mean())
    return {"Sharpe":float(shp),"AnnRet":float(ann),"MaxDD":mdd,"MAR":float(mar),"MonthlyWin":win}

def main(csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    px = load_rth(csv_path)
    assert {"AXP","COF"}.issubset(px.columns), "CSV must contain AXP and COF columns"
    bar_min, bpd, bpy = bars_geometry(px.index)
    print(f"[INFO] Bars: {bar_min}-minute | {bpd} bars/day | ~{bpy} bars/year")

    # baseline -> VolTarget (primary)
    r_base, info = run_pair_with_pos(px["AXP"].ffill(), px["COF"].ffill())
    r_vt = vol_target(r_base, bpd)
    # Buy & Hold (AXP)
    bh = px["AXP"].reindex(r_vt.index).pct_change().fillna(0.0)

    # Split: last 1Y test
    end = r_vt.index[-1]; start_test = end - pd.Timedelta(days=365)
    r_train = r_vt.loc[:start_test - pd.Timedelta(microseconds=1)]
    r_test  = r_vt.loc[start_test:]
    bh_train = bh.loc[r_train.index[0]:r_train.index[-1]]
    bh_test  = bh.loc[r_test.index[0]:r_test.index[-1]]

    # Metrics
    M_full = metrics(r_vt, bpy); M_train = metrics(r_train, bpy); M_test = metrics(r_test, bpy)
    MB_full = metrics(bh, bpy); MB_train = metrics(bh_train, bpy); MB_test = metrics(bh_test, bpy)

    # Save metrics
    pd.DataFrame([
        {"Segment":"Full", **M_full, "Start":r_vt.index[0], "End":r_vt.index[-1]},
        {"Segment":"Train", **M_train, "Start":r_train.index[0], "End":r_train.index[-1]},
        {"Segment":"Test", **M_test, "Start":r_test.index[0], "End":r_test.index[-1]},
    ]).to_csv(os.path.join(outdir, "metrics_strategy.csv"), index=False)
    pd.DataFrame([
        {"Segment":"Full", **MB_full},
        {"Segment":"Train", **MB_train},
        {"Segment":"Test", **MB_test},
    ]).to_csv(os.path.join(outdir, "metrics_buyhold_axp.csv"), index=False)
    print("[INFO] Metrics saved to CSV.")

    # Equity CSVs
    pd.DataFrame({"equity_strat": (1+r_vt).cumprod()}).to_csv(os.path.join(outdir, "equity_full_strat.csv"))
    pd.DataFrame({"equity_bh": (1+bh).cumprod()}).to_csv(os.path.join(outdir, "equity_full_bh.csv"))

    # Plots
    def plot_equity(eq1, eq2, title, fname):
        plt.figure(figsize=(11,4))
        plt.plot(eq1.index, eq1.values, label="Strategy + VolTarget (net)")
        plt.plot(eq2.index, eq2.values, label="Buy & Hold (AXP)")
        plt.title(title); plt.xlabel("Time"); plt.ylabel("Equity"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(outdir, fname), dpi=160); plt.close()

    plot_equity((1+r_vt).cumprod(), (1+bh).cumprod(), "Equity — Full", "equity_full.png")
    plot_equity((1+r_train).cumprod(), (1+bh_train).cumprod(), "Equity — Train", "equity_train.png")
    plot_equity((1+r_test).cumprod(), (1+bh_test).cumprod(), "Equity — Test", "equity_test.png")

    # Trade markers on TEST (AXP price as backdrop)
    ypx = px["AXP"].reindex(info["pos"].index).loc[r_test.index[0]:r_test.index[-1]].ffill()
    plt.figure(figsize=(11,4))
    plt.plot(ypx.index, ypx.values, label="AXP Price")
    entL = info["entL"].loc[r_test.index[0]:r_test.index[-1]]
    exL  = info["exL"].loc[r_test.index[0]:r_test.index[-1]]
    entS = info["entS"].loc[r_test.index[0]:r_test.index[-1]]
    exS  = info["exS"].loc[r_test.index[0]:r_test.index[-1]]
    plt.scatter(ypx.index[entL], ypx.values[entL], marker="^", s=28, color="g", label="Enter Long")
    plt.scatter(ypx.index[exL],  ypx.values[exL],  marker="v", s=28, color="g", label="Exit Long")
    plt.scatter(ypx.index[entS], ypx.values[entS], marker="v", s=28, color="r", label="Enter Short")
    plt.scatter(ypx.index[exS],  ypx.values[exS],  marker="^", s=28, color="r", label="Exit Short")
    plt.title("Trade Markers — TEST window"); plt.xlabel("Time"); plt.ylabel("AXP Price"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "trade_markers_test.png"), dpi=160); plt.close()

    # Monthly trade frequency (FULL)
    dpos = info["pos"].diff().fillna(0.0)
    entries = (dpos!=0) & (info["pos"]!=0)
    tpm_full = entries.reindex(r_vt.index, fill_value=False).resample("M").sum()
    plt.figure(figsize=(11,4))
    tpm_full.plot(kind="bar")
    plt.title("Monthly Round-Trips — FULL sample"); plt.xlabel("Month"); plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "monthly_trades_full.png"), dpi=160); plt.close()

    # Console summary
    def pct(x): return f"{x*100:.2f}%"
    print("\n=== SUMMARY (Strategy+VolTarget) ===")
    print(f"Full   Sharpe {M_full['Sharpe']:.2f} | AnnRet {pct(M_full['AnnRet'])} | MaxDD {-M_full['MaxDD']*100:.2f}%")
    print(f"Train  Sharpe {M_train['Sharpe']:.2f} | AnnRet {pct(M_train['AnnRet'])} | MaxDD {-M_train['MaxDD']*100:.2f}%")
    print(f"Test   Sharpe {M_test['Sharpe']:.2f} | AnnRet {pct(M_test['AnnRet'])} | MaxDD {-M_test['MaxDD']*100:.2f}%")
    print("\nOutputs saved to:", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with datetime, AXP, COF columns (minute close)")
    ap.add_argument("--outdir", default="axp_cof_outputs", help="Output directory")
    args = ap.parse_args()
    main(args.csv, args.outdir)
