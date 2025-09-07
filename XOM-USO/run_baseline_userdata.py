#!/usr/bin/env python3
# Baseline Pairs StatArb (XOM–USO style) — autodetect columns & intraday bar size
# Features:
#  - Detect datetime column & price columns (XOM/USO by name; else first 2 numeric)
#  - Restrict to US RTH with open/close buffers
#  - Infer bars-per-day from data spacing (1m/2m/5m) and scale annualization accordingly
#  - RLS hedge ratio (no look-ahead), rolling Z (shifted), vol targeting, fees=0.03%/side
#  - Outputs: equity.png/csv, signals.png, trade_frequency.png, trades.csv, metrics_train_test_full.csv
#
# Usage:
#   python run_baseline_userdata.py --csv symbol.csv --start 2023-01-02 --outdir ./outputs --entry 2.6 --exit_ 0.6 --z_days 12 --lam 0.999 --vol 0.07 --fee_bp 3.0

import os, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
plt.switch_backend("Agg")

# ---------- helpers ----------
def find_datetime_col(df: pd.DataFrame):
    candidates = [c for c in df.columns if str(c).lower() in ["datetime","timestamp","time","date","dt"]]
    for c in candidates:
        try:
            ts = pd.to_datetime(df[c])
            return c, ts
        except Exception:
            pass
    for c in df.columns:
        try:
            ts = pd.to_datetime(df[c])
            return c, ts
        except Exception:
            continue
    raise ValueError("No datetime-like column found.")

def pick_price_cols(df: pd.DataFrame):
    cols_lower = {str(c).lower(): c for c in df.columns}
    y = cols_lower.get("xom") or next((cols_lower[k] for k in cols_lower if "xom" in k), None)
    x = cols_lower.get("uso") or next((cols_lower[k] for k in cols_lower if "uso" in k), None)
    if y and x:
        return y, x
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        return num_cols[0], num_cols[1]
    raise ValueError("Couldn't find two price columns (e.g., XOM and USO).")

def infer_bars_per_day(idx):
    idx = pd.DatetimeIndex(idx)

    # 如果有时区，统一到美东再去掉tz，避免后续 .time/.between_time 出问题
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)

    # 去掉重复时间戳，避免 0 间隔
    idx = idx[~idx.duplicated(keep="first")]

    # A) 用 .asi8 把 TimedeltaIndex 转成纳秒，再除以 1e9 得到“秒”
    diffs_sec = (idx[1:] - idx[:-1]).asi8 / 1e9
    # 忽略非正间隔，取中位数步长
    step = float(np.median(diffs_sec[diffs_sec > 0])) if len(diffs_sec) else np.nan

    # 容错：若不可用，默认 60s 一根
    if not np.isfinite(step) or step <= 0:
        step = 60.0

    # 6.5 小时RTH = 23400秒
    if step >= 3600:               # 日线等低频
        return 1
    bpd = int(round(23400.0 / step))
    return int(np.clip(bpd, 1, 2000))

def rls_beta(logx: pd.Series, logy: pd.Series, lam: float=0.999, delta: float=1000.0):
    idx = logx.index; beta = 0.0; P = delta; out = np.zeros(len(idx))
    for i,t in enumerate(idx):
        x = float(logx.iloc[i]); y = float(logy.iloc[i])
        K = (P * x) / (lam + x * P * x)
        e = y - beta * x
        beta = beta + K * e
        P = (1/lam) * (P - K * x * P)
        out[i] = beta
    return pd.Series(out, index=idx)

def run_baseline(df_prices, y, x,
                 entry=2.6, exit_=0.6, z_days=12, lam=0.999,
                 vol_target=0.07, fee_bp=3.0,
                 open_buffer=15, close_buffer=5):
    # RTH window with buffers
    st = pd.Timestamp(f"09:{30+open_buffer:02d}").time()
    et = pd.Timestamp(f"15:{60-close_buffer:02d}").time()
    data = df_prices[[y,x]].dropna().copy()
    data = data[(data.index.time >= st) & (data.index.time <= et)]
    if len(data) < 1000:
        raise ValueError("Too little RTH data after filtering; check timestamps/timezone.")

    # Infer bars-per-day/year
    BPD = infer_bars_per_day(data.index)
    BPY = 252 * BPD

    # Returns & RLS beta
    ry = data[y].pct_change().fillna(0)
    rx = data[x].pct_change().fillna(0)
    lx, ly = np.log(data[x]), np.log(data[y])
    beta = rls_beta(lx, ly, lam=lam, delta=1000.0).clip(0.01, 5.0)
    spread = ly - beta*lx

    win = int(BPD * z_days)
    mean = spread.rolling(win, min_periods=max(40, win//2)).mean().shift(1)
    sd   = spread.rolling(win, min_periods=max(40, win//2)).std().shift(1)
    z = (spread - mean) / (sd + 1e-12)

    pos = pd.Series(0.0, index=data.index); prev = 0.0
    for t in data.index:
        zt = z.loc[t]
        if pd.isna(zt): curr = 0.0 if prev==0 else prev
        elif zt <= -entry: curr = 1.0
        elif zt >=  entry: curr = -1.0
        elif abs(zt) <= exit_: curr = 0.0 if prev!=0 else prev
        else: curr = prev
        pos.loc[t] = curr; prev = curr

    beta_f = beta.ffill().fillna(1.0)
    w_y_raw =  pos * (1.0/(1.0+beta_f.abs()))
    w_x_raw = -pos * (beta_f.abs()/(1.0+beta_f.abs())) * np.sign(beta_f)

    ret_raw = (w_y_raw.shift().fillna(0)*ry) + (w_x_raw.shift().fillna(0)*rx)

    roll = ret_raw.rolling(BPD, min_periods=BPD//2).std()
    scale = (vol_target/(roll+1e-12)).clip(upper=6.0).ewm(span=BPD*2, min_periods=20).mean().shift(1).fillna(1.0)
    w_y = w_y_raw * scale; w_x = w_x_raw * scale
    ret = (w_y.shift().fillna(0)*ry) + (w_x.shift().fillna(0)*rx)

    fee_rate = fee_bp/10000.0
    turnover = (w_y.diff().abs().fillna(0) + w_x.diff().abs().fillna(0))
    ret_net = ret - turnover*fee_rate

    eq = (1 + ret_net).cumprod()
    bh = (0.5*ry + 0.5*rx); bh_eq = (1 + bh).cumprod()

    chg = pos.diff().fillna(pos).ne(0); tpts = chg[chg].index
    trades = pd.DataFrame({
        "timestamp": tpts,
        "action": ["Open Long" if pos.loc[t]==1 else ("Open Short" if pos.loc[t]==-1 else "Close") for t in tpts],
        y: [data.loc[t, y] for t in tpts],
        x: [data.loc[t, x] for t in tpts],
        "z": [z.loc[t] if t in z.index else np.nan for t in tpts],
    })

    return {"returns": ret_net, "equity": eq, "bh_equity": bh_eq, "pos": pos, "z": z, "trades": trades, "BPD": BPD, "BPY": BPY}

def ann_return(returns: pd.Series, BPY: int) -> float:
    if len(returns)==0: return np.nan
    gross = (1+returns).prod(); years = len(returns)/BPY
    return gross**(1/years) - 1 if years>0 else np.nan

def sharpe_ratio(returns: pd.Series, BPY: int) -> float:
    if len(returns)==0 or returns.std()==0: return np.nan
    return (returns.mean()/(returns.std()+1e-12))*np.sqrt(BPY)

def max_drawdown_from_returns(returns: pd.Series) -> float:
    eq = (1+returns).cumprod(); peak = eq.cummax(); dd = (peak-eq)/peak
    return float(dd.max()) if len(dd) else np.nan

def monthly_win_rate(returns: pd.Series) -> float:
    if len(returns)==0: return np.nan
    m = (1+returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    return float((m>0).mean()) if len(m) else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with a datetime/timestamp column and two price columns (XOM/USO names preferred)")
    ap.add_argument("--start", default="2023-01-02")
    ap.add_argument("--end", default=None)
    ap.add_argument("--entry", type=float, default=2.6)
    ap.add_argument("--exit_", type=float, default=0.6)
    ap.add_argument("--z_days", type=int, default=12)
    ap.add_argument("--lam", type=float, default=0.999)
    ap.add_argument("--vol", type=float, default=0.07)
    ap.add_argument("--fee_bp", type=float, default=3.0)
    ap.add_argument("--outdir", default="./user_run_outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw = pd.read_csv(args.csv)
    dt_col, ts = find_datetime_col(raw)
    raw = raw.copy(); raw[dt_col] = ts
    raw = raw.set_index(dt_col).sort_index()

    y_col, x_col = pick_price_cols(raw)
    df = raw[[y_col, x_col]].astype(float).rename(columns={y_col:"Y", x_col:"X"})

    # window
    if args.end:
        used = df.loc[args.start:args.end].copy()
    else:
        used = df.loc[args.start:].copy()

    res = run_baseline(used.rename(columns={"Y":"XOM","X":"USO"}), "XOM","USO",
                       entry=args.entry, exit_=args.exit_, z_days=args.z_days,
                       lam=args.lam, vol_target=args.vol, fee_bp=args.fee_bp)

    BPD, BPY = res["BPD"], res["BPY"]
    r = res["returns"]; n = len(r); ntr = int(n*0.7)
    r_tr, r_te = r.iloc[:ntr], r.iloc[ntr:]

    def pack(ret, name):
        a = ann_return(ret, BPY); s = sharpe_ratio(ret, BPY); dd = max_drawdown_from_returns(ret)
        mar = (a/dd) if dd and not np.isnan(dd) and dd>0 else np.nan
        wr = monthly_win_rate(ret)
        return {"Which": name, "Bars": len(ret), "Annualized Gross Return": a, "Sharpe": s,
                "Max Drawdown": dd, "MAR": mar, "Win Rate (Monthly)": wr}

    tt = pd.DataFrame([pack(r_tr, "Train"), pack(r_te, "Test"), pack(r, "Full")])
    tt_path = os.path.join(args.outdir, "metrics_train_test_full.csv")
    tt.to_csv(tt_path, index=False)

    # Equity
    fig = plt.figure(figsize=(10,5))
    res["equity"].plot(label="Strategy"); res["bh_equity"].plot(label="Equal-Weight Buy & Hold")
    plt.title(f"Equity — Pair({y_col},{x_col}) mapped to (XOM,USO); bars/day≈{BPD}")
    plt.xlabel("Time"); plt.ylabel("Equity"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "equity.png")); plt.close()

    # Signals
    z = res["z"].dropna(); pos = res["pos"].reindex(z.index).fillna(0)
    chg = pos.diff().fillna(pos).ne(0)
    opens_long  = z[chg & pos.eq(1)]
    opens_short = z[chg & pos.eq(-1)]
    closes      = z[chg & pos.eq(0)]
    fig = plt.figure(figsize=(10,5))
    z.plot(label="Z")
    if not opens_long.empty:
        plt.scatter(opens_long.index,  opens_long.values,  s=24, marker="^", color="green", label="Open Long")
    if not opens_short.empty:
        plt.scatter(opens_short.index, opens_short.values, s=24, marker="v", color="red",   label="Open Short")
    if not closes.empty:
        plt.scatter(closes.index,      closes.values,      s=16, marker="o", color="gray",  label="Close")
    plt.axhline(args.entry, linestyle="--"); plt.axhline(-args.entry, linestyle="--"); plt.axhline(0, linestyle=":")
    plt.title("Signals — Buy (Green), Sell (Red), Flat (Gray)")
    plt.xlabel("Time"); plt.ylabel("Z"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "signals.png")); plt.close()

    # Monthly trade frequency
    trades = res["trades"].copy()
    if not trades.empty:
        trades["month"] = trades["timestamp"].dt.to_period("M").astype(str)
        opens = trades[trades["action"].isin(["Open Long","Open Short"])].copy()
        opens["month"] = opens["timestamp"].dt.to_period("M").astype(str)
        freq_open = opens.groupby(["month","action"]).size().unstack(fill_value=0)
        fig = plt.figure(figsize=(10,5))
        freq_open.sort_index().plot(kind="bar", ax=plt.gca())
        plt.title("Monthly Trade Frequency")
        plt.xlabel("Month"); plt.ylabel("Trades"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "trade_frequency.png")); plt.close()

    # CSVs
    pd.DataFrame({"equity": res["equity"], "bh_equity": res["bh_equity"]}).to_csv(os.path.join(args.outdir, "equity.csv"))
    res["trades"].to_csv(os.path.join(args.outdir, "trades.csv"), index=False)

    # Dump a small JSON summary for logs
    with open(os.path.join(args.outdir, "run_summary.json"), "w") as f:
        json.dump({"bars_per_day_estimate": int(BPD), "BPY": int(BPY), "metrics_csv": tt_path}, f, indent=2)

if __name__ == "__main__":
    main()
