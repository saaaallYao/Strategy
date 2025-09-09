# T+1 Intraday Contrarian (Enhanced) — Reproducible Script
# Requirements: pandas, numpy, matplotlib
# Usage:
#   python t1_contrarian_enhanced.py --zip /Users/chenxiyao/Downloads/CN/stock_search_cn_1min.zip --outdir ./outputs
# The script will generate metrics tables, equity plots, and CSVs.

import os, argparse, zipfile
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FEE = 0.0003
ROUND_TRIP = 2*FEE

def load_symbol_df(zf, path):
    df = pd.read_csv(zf.open(path), parse_dates=["timestamp"])
    df.columns = [c.strip().lower() for c in df.columns]
    sym = str(df.iloc[0,0])
    df["symbol"] = sym
    df = df[["symbol","timestamp","open","high","low","close","volume"]].copy()
    df["date"] = df["timestamp"].dt.date
    return df

def price_at_time(group, hm):
    sub = group[group["timestamp"].dt.strftime("%H:%M")==hm]
    if len(sub)==0: return np.nan
    return float(sub["close"].iloc[0])

def make_daily_features(df):
    rows = []
    for day, g in df.groupby("date"):
        rows.append({
            "symbol": g["symbol"].iloc[0],
            "date": pd.Timestamp(day),
            "open_0931": price_at_time(g, "09:31"),
            "pm_1400":   price_at_time(g, "14:00"),
            "pm_1455":   price_at_time(g, "14:55"),
            "pm_1458":   price_at_time(g, "14:58"),
            "d_volume": float(g["volume"].sum()),
        })
    daily = pd.DataFrame(rows).sort_values("date")
    daily["pm_ret_55m"] = daily["pm_1455"]/daily["pm_1400"] - 1.0
    daily["pm_speed_2m"] = daily["pm_1458"]/daily["pm_1455"] - 1.0
    daily["next_open"] = daily["open_0931"].shift(-1)
    daily["exit_open"] = daily["open_0931"].shift(-2)
    return daily

def metrics(rr):
    if len(rr)<5 or rr.std()==0:
        return {"days": int(len(rr))}
    ann=(1+rr.mean())**252-1; vol=rr.std()*np.sqrt(252); sharpe=rr.mean()/rr.std()*np.sqrt(252)
    eq=(1+rr).cumprod(); dd=(eq/eq.cummax()-1).min(); mar=ann/abs(dd) if dd<0 else np.nan
    win=(rr.groupby(pd.Grouper(freq="M")).apply(lambda x:(1+x).prod()-1)>0).mean()
    return {"days": int(len(rr)), "AnnReturn": ann, "AnnVol": vol, "Sharpe": sharpe, "MaxDD": float(dd), "MAR": mar, "WinRateMonthly": win}

def backtest_contrarian_enhanced(daily_all, etf_ret, start, end,
                                 z_threshold=-0.5, K_min=3, K_max=8,
                                 vol_ratio_min=0.8, pm_ret_floor=-0.04,
                                 cooldown_days=1):
    data = daily_all[(daily_all["date"]>=pd.to_datetime(start)) & (daily_all["date"]<=pd.to_datetime(end))].dropna(subset=["next_open","exit_open"])
    last_entry = {}
    trades = []
    for d, g in data.groupby("date"):
        cand = g[(g["z_pm"]<z_threshold) & (g["vol_ratio"]>=vol_ratio_min) & (g["pm_ret_55m"]>=pm_ret_floor)]
        if cooldown_days>0 and len(last_entry)>0:
            cand = cand[~cand["symbol"].isin([s for s,dt0 in last_entry.items() if (pd.Timestamp(d) - dt0).days <= cooldown_days])]
        if cand.empty: continue
        K_dyn = int(np.clip(len(cand), K_min, K_max))
        picks = cand.nsmallest(K_dyn, "score")
        for _, row in picks.iterrows():
            ret = row["exit_open"]/row["next_open"] - 1.0 - ROUND_TRIP
            trades.append({"signal_date": row["date"], "symbol": row["symbol"], "ret": ret,
                           "entry_open": row["next_open"], "exit_open": row["exit_open"]})
            last_entry[row["symbol"]] = pd.Timestamp(d)
    t = pd.DataFrame(trades)
    if t.empty:
        return {"daily": pd.Series(dtype=float), "equity": pd.Series(dtype=float), "trades": t,
                "metrics": {}, "daily_hedged": pd.Series(dtype=float), "equity_hedged": pd.Series(dtype=float),
                "metrics_hedged": {}}
    t["pnl_date"] = t["signal_date"] + pd.Timedelta(days=2)
    dr = t.groupby("pnl_date")["ret"].mean().sort_index()
    eq = (1+dr).cumprod()
    m_un = metrics(dr)
    # rolling 20-day beta hedge
    er = etf_ret.reindex(dr.index).fillna(0.0)
    hedged = []
    window = 20
    for i in range(len(dr)):
        past = dr.iloc[:i].tail(window)
        past_er = er.iloc[:i].tail(window)
        if len(past)>=5 and past_er.var()>0:
            beta = np.cov(past, past_er, ddof=0)[0,1]/past_er.var()
            beta = float(np.clip(beta, 0.0, 0.6))
        else:
            beta = 0.0
        hedged.append(dr.iloc[i] - beta*er.iloc[i] - beta*ROUND_TRIP)
    hr = pd.Series(hedged, index=dr.index)
    eqh = (1+hr).cumprod()
    m_hd = metrics(hr)
    return {"daily": dr, "equity": eq, "trades": t, "metrics": m_un,
            "daily_hedged": hr, "equity_hedged": eqh, "metrics_hedged": m_hd}

def main(zip_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    zf = zipfile.ZipFile(zip_path)
    paths = [p for p in zf.namelist() if p.endswith(".csv") and not p.startswith("__MACOSX/")]
    stock_paths = [p for p in paths if "ETF" not in p]
    etf_path = [p for p in paths if "510300" in p][0]

    # Load up to 16 stocks
    symbols = {}
    for p in stock_paths:
        try:
            df = load_symbol_df(zf, p)
            symbols[df["symbol"].iloc[0]] = df
            if len(symbols) >= 16: break
        except Exception:
            pass
    assert len(symbols)>0, "No stock data loaded."

    etf_df = load_symbol_df(zf, etf_path)
    etf_open = etf_df.groupby(etf_df["timestamp"].dt.date)["open"].first()
    etf_open.index = pd.to_datetime(etf_open.index)
    etf_ret = etf_open.pct_change().dropna()

    # Daily features
    daily = pd.concat([make_daily_features(df) for df in symbols.values()], ignore_index=True).sort_values(["symbol","date"])
    daily["vol_med20"] = daily.groupby("symbol")["d_volume"].transform(lambda s: s.rolling(20, min_periods=5).median())
    daily["vol_ratio"] = daily["d_volume"]/daily["vol_med20"]
    for col in ["z_pm","z_speed","score"]:
        daily[col]=np.nan
    for d, g in daily.groupby("date"):
        def z(x):
            s=x.std(ddof=0)
            if s==0 or np.isnan(s): return pd.Series(0.0, index=x.index)
            return (x - x.mean())/s
        zpm=z(g["pm_ret_55m"]); zsp=z(g["pm_speed_2m"])
        idx=g.index
        daily.loc[idx,"z_pm"]=zpm.values
        daily.loc[idx,"z_speed"]=zsp.values
        daily.loc[idx,"score"]=(zpm+0.5*zsp).values

    TRAIN_START, TRAIN_END = "2024-09-01","2025-02-28"
    TEST_START,  TEST_END  = "2025-03-01","2025-05-23"
    FULL_START,  FULL_END  = "2024-09-01","2025-05-23"

    params = dict(z_threshold=-0.5, K_min=3, K_max=8, vol_ratio_min=0.8, pm_ret_floor=-0.04, cooldown_days=1)
    tr = backtest_contrarian_enhanced(daily, etf_ret, TRAIN_START, TRAIN_END, **params)
    te = backtest_contrarian_enhanced(daily, etf_ret, TEST_START, TEST_END, **params)
    fu = backtest_contrarian_enhanced(daily, etf_ret, FULL_START, FULL_END, **params)

    # Export CSVs
    tr["trades"].to_csv(os.path.join(outdir, "trades_train.csv"), index=False)
    te["trades"].to_csv(os.path.join(outdir, "trades_test.csv"), index=False)
    fu["trades"].to_csv(os.path.join(outdir, "trades_full.csv"), index=False)
#     tr["daily"].to_csv(os.path.join(outdir, "daily_returns_train.csv"))
#     te["daily"].to_csv(os.path.join(outdir, "daily_returns_test.csv"))
#     fu["daily"].to_csv(os.path.join(outdir, "daily_returns_full.csv"))
#     fu["daily_hedged"].to_csv(os.path.join(outdir, "daily_returns_full_hedged.csv"))

    
    
    def _save_fig(fig, outdir, name, dpi=180):
        for ext in ("png","jpg"):
            fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # Equity plots (Full/Train/Test), each saved as PNG/JPG
    def equity_plot(res, label, name_suffix):
        eq  = res["equity"]
        eqh = res["equity_hedged"]
        beq = (1 + etf_ret.reindex(eq.index).fillna(0.0)).cumprod()
        fig, ax = plt.subplots(figsize=(10,6))
        eq.plot(ax=ax, label=f"{label} Strategy")
        eqh.plot(ax=ax, label=f"{label} Hedged")
        beq.plot(ax=ax, label="510300 ETF")
        ax.legend(); ax.set_title(f"Equity — {label}")
        _save_fig(fig, outdir, f"report_equity_{name_suffix}")

    equity_plot(fu, "Full",  "full")
    equity_plot(tr, "Train", "train")
    equity_plot(te, "Test",  "test")

    # Trade frequency (Full)
    fig, ax = plt.subplots(figsize=(8,5))
    freq = pd.DataFrame({
        "Trades":[len(fu["trades"])],
        "Per Day":[len(fu["trades"])/max(len(fu["daily"]),1)],
        "Per Week":[len(fu["trades"])/max(len(fu["daily"])/5,1)]
    }, index=["Enhanced (Full)"])
    freq.plot(kind="bar", ax=ax); ax.set_title("Trade Frequency (Full)")
    _save_fig(fig, outdir, "report_trade_frequency_full")

    # Buy/Sell markers (top symbol) — explicit red/green
    counts = fu["trades"]["symbol"].value_counts()
    if len(counts) > 0:
        sym = counts.index[0]
        s_df = symbols[sym]
        sym_open = s_df.groupby(s_df["timestamp"].dt.date)["open"].first()
        sym_open.index = pd.to_datetime(sym_open.index)
        st = fu["trades"][fu["trades"]["symbol"] == sym].copy()
        st["entry_date"] = st["signal_date"] + pd.Timedelta(days=1)
        st["exit_date"]  = st["signal_date"] + pd.Timedelta(days=2)
        fig, ax = plt.subplots(figsize=(11,6))
        sym_open.plot(ax=ax, label="Open")
        ax.scatter(st["entry_date"], sym_open.reindex(st["entry_date"].values).values, marker="^", color="green", label="Buy (D+1 open)")
        ax.scatter(st["exit_date"],  sym_open.reindex(st["exit_date"].values).values,  marker="v", color="red",   label="Sell (D+2 open)")
        ax.legend(); ax.set_title(f"{sym} — Buy/Sell Markers (Full)")
        _save_fig(fig, outdir, "report_buy_sell_markers_full")

    # === Metrics CSVs for two strategies (unhedged / hedged) across train/test/full ===
    def _metrics_row(m: dict, split: str) -> dict:
        return {
            "Split": split,
            "Annualized Return": m.get("AnnReturn", np.nan),
            "Annualized Vol": m.get("AnnVol", np.nan),
            "Sharpe": m.get("Sharpe", np.nan),
            "Max Drawdown": m.get("MaxDD", np.nan),
            "MAR": m.get("MAR", np.nan),
            "Win Rate (Monthly)": m.get("WinRateMonthly", np.nan),
            "N Days": m.get("days", np.nan),
        }

    unhedged_rows = [
        _metrics_row(tr.get("metrics", {}), "train"),
        _metrics_row(te.get("metrics", {}), "test"),
        _metrics_row(fu.get("metrics", {}), "full"),
    ]
    hedged_rows = [
        _metrics_row(tr.get("metrics_hedged", {}), "train"),
        _metrics_row(te.get("metrics_hedged", {}), "test"),
        _metrics_row(fu.get("metrics_hedged", {}), "full"),
    ]

    df_un = pd.DataFrame(unhedged_rows).set_index("Split")
    df_hd = pd.DataFrame(hedged_rows).set_index("Split")

    df_un.to_csv(os.path.join(outdir, "metrics_unhedged.csv"), float_format="%.6f")
    df_hd.to_csv(os.path.join(outdir, "metrics_hedged.csv"),   float_format="%.6f")

    # === Separate trade logs for two strategies ===
    # (If hedged trades are identical to unhedged at position level, we still export separate files for clarity)
    tr["trades"].to_csv(os.path.join(outdir, "trades_unhedged_train.csv"), index=False)
    te["trades"].to_csv(os.path.join(outdir, "trades_unhedged_test.csv"),  index=False)
    fu["trades"].to_csv(os.path.join(outdir, "trades_unhedged_full.csv"),  index=False)

    tr["trades"].to_csv(os.path.join(outdir, "trades_hedged_train.csv"), index=False)
    te["trades"].to_csv(os.path.join(outdir, "trades_hedged_test.csv"),  index=False)
    fu["trades"].to_csv(os.path.join(outdir, "trades_hedged_full.csv"),  index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to ZIP with CSVs")
    parser.add_argument("--outdir", default="./outputs", help="Output directory")
    args = parser.parse_args()
    main(args.zip, args.outdir)