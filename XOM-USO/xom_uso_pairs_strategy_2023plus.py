#!/usr/bin/env python3
# XOM–USO Intraday Pairs Strategy (2023+ sample) — baseline version
# Requirements: pandas, numpy, matplotlib
# Usage: python xom_uso_pairs_strategy_2023plus.py --csv your_minute_csv.csv --start 2023-01-02 --end 2025-07-31

import os, argparse, math, numpy as np, pandas as pd, matplotlib.pyplot as plt

BARS_PER_DAY = int(round(390/5))
BARS_PER_YEAR = 252 * BARS_PER_DAY

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

def statarb_baseline(df_prices, y="XOM", x="USO",
                     entry=2.6, exit_=0.6, z_days=12, lam=0.999,
                     vol_target=0.07, fee_bp=3.0,
                     open_buffer=15, close_buffer=5):
    data = df_prices[[y,x]].dropna().copy()
    st = pd.Timestamp(f"09:{30+open_buffer:02d}").time()
    et = pd.Timestamp(f"15:{60-close_buffer:02d}").time()
    data = data[(data.index.time >= st) & (data.index.time <= et)]
    ry = data[y].pct_change().fillna(0); rx = data[x].pct_change().fillna(0)
    lx, ly = np.log(data[x]), np.log(data[y])
    beta = rls_beta(lx, ly, lam=lam, delta=1000.0).clip(0.01,5.0)
    spread = ly - beta*lx
    win = int(BARS_PER_DAY * z_days)
    mean = spread.rolling(win, min_periods=max(40, win//2)).mean().shift(1)
    sd   = spread.rolling(win, min_periods=max(40, win//2)).std().shift(1)
    z = (spread - mean) / (sd + 1e-12)

    pos = pd.Series(0.0, index=data.index); prev = 0.0
    for t in data.index:
        zt = z.loc[t]
        if pd.isna(zt):
            curr = 0.0 if prev==0 else prev
        elif zt <= -entry: curr = 1.0
        elif zt >= entry:  curr = -1.0
        elif abs(zt) <= exit_: curr = 0.0 if prev!=0 else prev
        else: curr = prev
        pos.loc[t] = curr; prev = curr

    beta_f = beta.ffill().fillna(1.0)
    w_y_raw =  pos * (1.0/(1.0+beta_f.abs()))
    w_x_raw = -pos * (beta_f.abs()/(1.0+beta_f.abs())) * np.sign(beta_f)
    ret_raw = (w_y_raw.shift().fillna(0)*ry) + (w_x_raw.shift().fillna(0)*rx)

    roll = ret_raw.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY//2).std()
    scale = (vol_target/(roll+1e-12)).clip(upper=6.0).ewm(span=BARS_PER_DAY*2, min_periods=20).mean().shift(1).fillna(1.0)
    w_y = w_y_raw * scale; w_x = w_x_raw * scale
    ret = (w_y.shift().fillna(0)*ry) + (w_x.shift().fillna(0)*rx)

    fee_rate = fee_bp/10000.0
    turnover = (w_y.diff().abs().fillna(0) + w_x.diff().abs().fillna(0))
    ret_net = ret - turnover*fee_rate

    eq = (1+ret_net).cumprod()
    bh = (0.5*ry + 0.5*rx); bh_eq = (1+bh).cumprod()

    chg = pos.diff().fillna(pos).ne(0); tpts = chg[chg].index
    trades = pd.DataFrame({
        "timestamp": tpts,
        "action": ["Open Long" if pos.loc[t]==1 else ("Open Short" if pos.loc[t]==-1 else "Close") for t in tpts],
        y: [data.loc[t, y] for t in tpts],
        x: [data.loc[t, x] for t in tpts],
        "z": [z.loc[t] if t in z.index else np.nan for t in tpts],
    })
    return {"returns": ret_net, "equity": eq, "bh_equity": bh_eq, "pos": pos, "z": z, "trades": trades}

def ann_return(returns: pd.Series) -> float:
    if len(returns)==0: return np.nan
    gross = (1+returns).prod(); years = len(returns)/BARS_PER_YEAR
    return gross**(1/years) - 1 if years>0 else np.nan

def sharpe_ratio(returns: pd.Series) -> float:
    if len(returns)==0 or returns.std()==0: return np.nan
    return (returns.mean()/(returns.std()+1e-12))*np.sqrt(BARS_PER_YEAR)

def max_drawdown_from_returns(returns: pd.Series) -> float:
    eq = (1+returns).cumprod(); peak = eq.cummax(); dd = (peak-eq)/peak
    return float(dd.max()) if len(dd) else np.nan

def monthly_win_rate(returns: pd.Series) -> float:
    if len(returns)==0: return np.nan
    m = (1+returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    return float((m>0).mean()) if len(m) else np.nan

def pack_metrics(returns: pd.Series):
    a = ann_return(returns); s = sharpe_ratio(returns); dd = max_drawdown_from_returns(returns)
    mar = (a/dd) if dd and not np.isnan(dd) and dd>0 else np.nan
    wr = monthly_win_rate(returns)
    return a, s, dd, mar, wr

def main():
    import pandas as pd, numpy as np, matplotlib.pyplot as plt
    import argparse, os

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with 'datetime','XOM','USO' minute close columns")
    ap.add_argument("--start", default="2023-01-02")
    ap.add_argument("--end", default=None)
    ap.add_argument("--entry", type=float, default=2.6)
    ap.add_argument("--exit_", type=float, default=0.6)
    ap.add_argument("--z_days", type=int, default=12)
    ap.add_argument("--lam", type=float, default=0.999)
    ap.add_argument("--vol", type=float, default=0.07)
    ap.add_argument("--fee_bp", type=float, default=3.0)
    ap.add_argument("--outdir", default="./xom_uso_outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv, parse_dates=["datetime"]).set_index("datetime").sort_index()
    if args.end:
        df = df.loc[args.start:args.end].copy()
    else:
        df = df.loc[args.start:].copy()

    res = statarb_baseline(df, "XOM","USO", entry=args.entry, exit_=args.exit_,
                           z_days=args.z_days, lam=args.lam, vol_target=args.vol, fee_bp=args.fee_bp)

    r = res["returns"]; n = len(r); ntr = int(n*0.7)
    r_tr, r_te = r.iloc[:ntr], r.iloc[ntr:]
    full = pack_metrics(r); tr = pack_metrics(r_tr); te = pack_metrics(r_te)

    # Save equity & signals
    fig = plt.figure(figsize=(10,5))
    res["equity"].plot(label="Strategy"); res["bh_equity"].plot(label="Equal-Weight Buy & Hold")
    plt.title("Equity — XOM–USO"); plt.xlabel("Time"); plt.ylabel("Equity"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "equity.png")); plt.close()

    z = res["z"].dropna(); pos = res["pos"].reindex(z.index).fillna(0)
    chg = pos.diff().fillna(pos).ne(0)
    opens_long = z[chg & pos.eq(1)]; opens_short = z[chg & pos.eq(-1)]; closes = z[chg & pos.eq(0)]
    fig = plt.figure(figsize=(10,5))
    z.plot(label="Z")
    if not opens_long.empty:  plt.scatter(opens_long.index,  opens_long.values,  s=24, marker="^", color="green", label="Open Long")
    if not opens_short.empty: plt.scatter(opens_short.index, opens_short.values, s=24, marker="v", color="red",   label="Open Short")
    if not closes.empty:      plt.scatter(closes.index,      closes.values,      s=16, marker="o", color="gray",  label="Close")
    plt.axhline(args.entry, linestyle="--"); plt.axhline(-args.entry, linestyle="--"); plt.axhline(0, linestyle=":")
    plt.title("Signals"); plt.xlabel("Time"); plt.ylabel("Z"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "signals.png")); plt.close()

    # Save CSVs
    pd.DataFrame({"equity": res["equity"], "bh_equity": res["bh_equity"]}).to_csv(os.path.join(args.outdir, "equity.csv"))
    res["trades"].to_csv(os.path.join(args.outdir, "trades.csv"), index=False)

    # Save a small metrics txt
    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        def fmt(m): return ",".join([f"{x:.6f}" if x==x else "nan" for x in m])
        f.write("Full Ann,Sharpe,MaxDD,MAR,WinRate(M)\n"+fmt(full)+"\n")
        f.write("Train Ann,Sharpe,MaxDD,MAR,WinRate(M)\n"+fmt(tr)+"\n")
        f.write("Test Ann,Sharpe,MaxDD,MAR,WinRate(M)\n"+fmt(te)+"\n")

if __name__ == "__main__":
    main()
