#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 (T 14:46 → T+1 14:56) — Embedded ZIP path, CSV outputs
"""

import os, zipfile, glob, shutil
import numpy as np, pandas as pd
from datetime import time, datetime
import matplotlib.pyplot as plt

# -------- Parameters (embedded) --------
ZIP_PATH = r"/Users/chenxiyao/Downloads/CN/stock_search_cn_1min.zip"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_V2")
TOPK = 5
COST_PER_SIDE = 0.0003
SIG_START = time(14,16)  # signal 14:16→14:46
SIG_END   = time(14,46)
ENTRY_T   = time(14,46)  # enter at T 14:46
EXIT_T    = time(14,56)  # exit at T+1 14:56

os.makedirs(OUT_DIR, exist_ok=True)

def load_from_zip(zip_path):
    tmp_dir = os.path.join(OUT_DIR, "__tmp_zip_extract__")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp_dir)
    csvs = [f for f in glob.glob(os.path.join(tmp_dir, "**", "*.csv"), recursive=True) if "/__MACOSX/" not in f]
    if not csvs:
        raise FileNotFoundError(f"No CSVs found inside {zip_path}")
    sym_dfs = {}
    for f in csvs:
        try:
            df = pd.read_csv(f, encoding="utf-8")
            need = {'symbol','timestamp','close'}
            if not need.issubset(df.columns): 
                continue
            if 'open' not in df.columns: 
                df['open'] = df['close']
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['t'] = df['timestamp'].dt.time
            df = df.sort_values("timestamp")
            sym = str(df['symbol'].iloc[0])
            sym_dfs[sym] = df
        except Exception as e:
            print("[WARN] skip", f, e)
    symbols = sorted(sym_dfs.keys())
    if not symbols:
        raise RuntimeError("No valid symbols loaded from ZIP.")
    all_dates = sorted(set(pd.concat([df['date'] for df in sym_dfs.values()]).unique()))
    def build_px_at(tar_time, col="close"):
        tbl = pd.DataFrame(index=all_dates, columns=symbols, dtype=float)
        for sym, df in sym_dfs.items():
            g = df[df['t'] == tar_time][['date', col]].drop_duplicates('date')
            s = g.set_index('date')[col]
            tbl.loc[s.index, sym] = s.values
        return tbl
    return tmp_dir, sym_dfs, symbols, all_dates, build_px_at

def metrics_from_daily(r: pd.Series):
    if len(r)==0 or r.std(ddof=0)==0:
        return dict(AnnRet=np.nan, Sharpe=np.nan, MaxDD=np.nan, MAR=np.nan, WinRateM=np.nan)
    ann = (1+r).prod()**(252/len(r)) - 1
    sharpe = np.sqrt(252)*r.mean()/(r.std(ddof=0)+1e-12)
    curve=(1+r).cumprod(); peak=curve.cummax(); mdd=(curve/peak-1).min()
    mar=(ann/abs(mdd)) if mdd<0 else np.nan
    win=(r.resample('M').sum()>0).mean()
    return dict(AnnRet=float(ann), Sharpe=float(sharpe), MaxDD=float(mdd), MAR=float(mar), WinRateM=float(win))

def split_train_test(r: pd.Series, ratio: float=0.7):
    n=len(r); sp=int(n*ratio); return r.iloc[:sp], r.iloc[sp:], r

def run():
    tmp_dir, sym_dfs, symbols, all_dates, build_px_at = load_from_zip(ZIP_PATH)
    try:
        px_sigA = build_px_at(SIG_START, "close")
        px_sigB = build_px_at(SIG_END,   "close")
        px_entry= build_px_at(ENTRY_T,   "close")
        px_exit = build_px_at(EXIT_T,    "close")

        min_names = TOPK
        valid=(px_sigA.notna().sum(axis=1)>=min_names) & (px_sigB.notna().sum(axis=1)>=min_names) & (px_entry.notna().sum(axis=1)>=min_names)
        px_sigA=px_sigA.loc[valid]; px_sigB=px_sigB.loc[valid]; px_entry=px_entry.loc[valid]
        px_exit=px_exit.loc[px_exit.index.intersection(px_entry.index)]

        signal=(px_sigB/px_sigA)-1.0
        dates=list(signal.index); dates_next={dates[i]:dates[i+1] for i in range(len(dates)-1)}

        ROUNDTRIP=COST_PER_SIDE*2.0
        pnl=[]; pnl_dates=[]; trades=[]
        for i,d in enumerate(dates[:-1]):
            row=signal.loc[d].dropna()
            if row.shape[0] < TOPK: continue
            longs=row.sort_values(ascending=True).index[:TOPK]
            d_next=dates_next[d]
            p_in=px_entry.loc[d, longs]; p_out=px_exit.loc[d_next, longs]
            valid=(~p_in.isna()) & (~p_out.isna()) & (p_in>0)
            if valid.sum()==0: continue
            rets=(p_out[valid].values/p_in[valid].values)-1.0
            pnl.append(float(np.mean(rets)-ROUNDTRIP)); pnl_dates.append(d_next)
            for sym,ein,eout,ok in zip(longs, p_in.values, p_out.values, valid.values):
                if not ok: continue
                trades.append(dict(trade_date=d_next, symbol=sym,
                                   entry_day=str(d), entry_time=str(ENTRY_T), entry_px=float(ein),
                                   exit_day=str(d_next), exit_time=str(EXIT_T), exit_px=float(eout),
                                   ret=float((eout/ein)-1.0-ROUNDTRIP)))
        sr=pd.Series(pnl, index=pd.to_datetime(pnl_dates)).sort_index()

        # Baseline
        bh=[]; bh_dates=[]
        for i,d in enumerate(dates[:-1]):
            ein=px_entry.loc[d].dropna(); eout=px_exit.loc[dates_next[d]].dropna()
            common=ein.index.intersection(eout.index)
            if len(common)==0: continue
            bh.append(float(((eout.loc[common]/ein.loc[common]) - 1.0).mean())); bh_dates.append(dates_next[d])
        bh_sr=pd.Series(bh, index=pd.to_datetime(bh_dates)).sort_index().reindex(sr.index).fillna(0.0)

        # Metrics
        def make_metrics(sr_obj, bh_obj):
            str_tr,str_te,str_fu=split_train_test(sr_obj,0.7); bh_tr,bh_te,bh_fu=split_train_test(bh_obj,0.7)
            rows=[]
            for label,s,tag in [("V2 (T 14:46 → T+1 14:56)",str_tr,"Train"),("V2 (T 14:46 → T+1 14:56)",str_te,"Test"),("V2 (T 14:46 → T+1 14:56)",str_fu,"Full"),
                                ("Buy&Hold (EW, same window)",bh_tr,"Train"),("Buy&Hold (EW, same window)",bh_te,"Test"),("Buy&Hold (EW, same window)",bh_fu,"Full")]:
                m=metrics_from_daily(s)
                rows.append(dict(Label=label, Set=tag, Span=(f"{s.index.min().date()} → {s.index.max().date()}" if len(s)>0 else "N/A"),
                                 Days=len(s), AnnRet=m["AnnRet"], Sharpe=m["Sharpe"], MaxDD=m["MaxDD"], MAR=m["MAR"], WinRateM=m["WinRateM"]))
            return pd.DataFrame(rows, columns=["Label","Set","Span","Days","AnnRet","Sharpe","MaxDD","MAR","WinRateM"])

        metrics_df = make_metrics(sr, bh_sr)

        # Save CSVs
        metrics_csv = os.path.join(OUT_DIR, "metrics.csv"); metrics_df.to_csv(metrics_csv, index=False)
        trades_csv  = os.path.join(OUT_DIR, "trade_log.csv");  pd.DataFrame(trades).to_csv(trades_csv, index=False)

        # Plots (sanity)
        eq_str=(1+sr).cumprod(); eq_bh=(1+bh_sr).cumprod()
        plt.figure(figsize=(9,4.2)); plt.plot(eq_str.index, eq_str.values, label="V2 (T 14:46 → T+1 14:56)")
        plt.plot(eq_bh.index, eq_bh.values, label="Buy&Hold (EW, same window)")
        plt.title("Equity Curve — V2 (T 14:46 → T+1 14:56)"); plt.xlabel("Date"); plt.ylabel("Equity (Start=1)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "equity.png"), dpi=150); plt.close()

        trades_df=pd.DataFrame(trades); trades_per_day=trades_df.groupby("trade_date").size() if len(trades_df)>0 else pd.Series(dtype=int)
        plt.figure(figsize=(8,4)); bins=range(0, trades_per_day.max()+2 if len(trades_per_day)>0 else 1)
        plt.hist(trades_per_day.values, bins=bins); plt.title("Trading Frequency — V2")
        plt.xlabel("Trades per Day"); plt.ylabel("Count of Days"); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "trading_frequency.png"), dpi=150); plt.close()

        print("Done. CSVs at:", metrics_csv, trades_csv)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    run()
