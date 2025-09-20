# momentum_strategy_runner.py
# Momentum (T+1) Anti-Lookahead Strategy Runner (with Train/Test outputs)
# Usage:
#   python momentum_strategy_runner.py --zip "/path/to/stock_search_cn_1min.zip" --out "./out" --cutoff "14:45" --top_n 5 --fee 0.0003

import os, io, zipfile, argparse, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRADING_DAYS_PER_YEAR = 245

def load_zip_minutes_defensive(zip_path):
    frames = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            try:
                content = zf.read(name)
                df = None
                for enc in [None, "utf-8", "gbk", "cp936", "latin1"]:
                    try:
                        df = pd.read_csv(io.BytesIO(content), encoding=enc) if enc else pd.read_csv(io.BytesIO(content))
                        if df is not None and len(df.columns)>0:
                            break
                    except Exception:
                        df = None
                if df is None or df.empty:
                    continue
                cols = {c.lower(): c for c in df.columns}
                rename_map = {}
                if "symbol" not in cols:
                    for k in ["code","ticker","secid","ts_code"]:
                        if k in cols: rename_map[cols[k]] = "symbol"; break
                if "datetime" not in cols:
                    for k in ["time","timestamp","datatime","date_time","trade_time","datetime"]:
                        if k in cols: rename_map[cols[k]] = "datetime"; break
                for k in ["open","high","low","close","volume"]:
                    if k not in cols and k.upper() in cols:
                        rename_map[cols[k.upper()]] = k
                if rename_map: df = df.rename(columns=rename_map)
                if not all(r in df.columns for r in ["symbol","datetime","open","high","low","close"]):
                    continue
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.dropna(subset=["datetime"]).sort_values(["symbol","datetime"])
                frames.append(df[["symbol","datetime","open","high","low","close","volume"]].copy() if "volume" in df.columns else df[["symbol","datetime","open","high","low","close"]].copy())
            except Exception:
                continue
    if not frames:
        raise RuntimeError("No valid CSVs discovered in the ZIP.")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol","datetime"]).sort_values(["symbol","datetime"])

def annualize_return(total_return, num_days):
    if num_days <= 0: return np.nan
    years = num_days / TRADING_DAYS_PER_YEAR
    return (1.0 + total_return) ** (1.0 / years) - 1.0

def sharpe_ratio(daily_ret, rf=0.0):
    s = daily_ret.replace([np.inf, -np.inf], np.nan).dropna()
    if s.std(ddof=0) == 0: return np.nan
    return math.sqrt(TRADING_DAYS_PER_YEAR) * (s.mean() - rf) / s.std(ddof=0)

def drawdown(equity):
    roll_max = equity.cummax()
    dd = equity/roll_max - 1.0
    return dd, dd.min()

def monthly_win_rate(daily_ret):
    if len(daily_ret)==0: return np.nan
    df = daily_ret.to_frame("ret").copy()
    df["ym"] = df.index.to_period("M")
    m = df.groupby("ym")["ret"].apply(lambda x: (1+x).prod() - 1)
    return float((m > 0).mean()) if len(m)>0 else np.nan

def summarize_series(daily_ret):
    eq = (1.0 + daily_ret).cumprod()
    total_ret = eq.iloc[-1] - 1.0 if len(eq)>0 else np.nan
    ann = annualize_return(total_ret, len(daily_ret))
    shp = sharpe_ratio(daily_ret)
    dd_curve, max_dd = drawdown(eq) if len(eq)>0 else (pd.Series(dtype=float), np.nan)
    mar = (ann/abs(max_dd)) if (max_dd not in [None,0,np.nan] and not np.isnan(ann)) else np.nan
    mwr = monthly_win_rate(daily_ret)
    return total_ret, ann, shp, max_dd, mar, mwr

def run(zip_path, out_dir, cutoff="14:45", top_n=5, fee=0.0003):
    os.makedirs(out_dir, exist_ok=True)
    data = load_zip_minutes_defensive(zip_path)
    data["date"] = data["datetime"].dt.date
    data["clock"] = data["datetime"].dt.strftime("%H:%M")

    first = data.groupby(["symbol","date"]).first().rename(columns={"open":"day_open"})[["day_open"]].reset_index()
    last  = data.groupby(["symbol","date"]).last().rename(columns={"close":"day_close"})[["day_close"]].reset_index()
    cut   = data[data["clock"]<=cutoff].groupby(["symbol","date"]).last().rename(columns={"close":"cut_close"})[["cut_close"]].reset_index()
    daily = first.merge(last, on=["symbol","date"], how="inner").merge(cut, on=["symbol","date"], how="inner")

    if "volume" in data.columns:
        vol_day    = data.groupby(["symbol","date"])["volume"].sum().rename("day_volume").reset_index()
        last_close = data.groupby(["symbol","date"])["close"].last().rename("last_close").reset_index()
        value_df = vol_day.merge(last_close, on=["symbol","date"], how="left")
        value_df["day_value"] = value_df["day_volume"] * value_df["last_close"]
        daily = daily.merge(value_df[["symbol","date","day_value"]], on=["symbol","date"], how="left")
    else:
        daily["day_value"] = np.nan

    daily = daily.sort_values(["symbol","date"])
    daily["intraday_ret_sofar"] = daily["cut_close"] / daily["day_open"] - 1.0

    # Anti-lookahead liquidity
    daily["value_med20_raw"] = daily.groupby("symbol")["day_value"].transform(lambda x: x.rolling(20, min_periods=5).median())
    daily["value_med20_y"]   = daily.groupby("symbol")["value_med20_raw"].shift(1)
    daily["yday_value"]      = daily.groupby("symbol")["day_value"].shift(1)
    daily["liq_ok"]          = (daily["yday_value"] >= 0.8 * daily["value_med20_y"]).fillna(False)

    # T+1
    daily["next_close"] = daily.groupby("symbol")["day_close"].shift(-1)
    daily["ret_close_to_next"] = daily["next_close"] / daily["day_close"] - 1.0

    # Build trades from Top-N momentum at cutoff
    trades = []
    for d, df_day in daily.groupby("date"):
        dd = df_day[df_day["liq_ok"]].dropna(subset=["intraday_ret_sofar"]).copy()
        if len(dd)==0: 
            continue
        pick = dd.sort_values("intraday_ret_sofar", ascending=False).head(top_n)
        for _, row in pick.iterrows():
            if pd.isna(row["next_close"]): 
                continue
            ret_net = (1.0 + row["ret_close_to_next"]) * (1 - fee) * (1 - fee) - 1.0
            trades.append({
                "date_buy": pd.to_datetime(d),
                "symbol": row["symbol"],
                "px_buy": row["day_close"],
                "date_sell": pd.to_datetime(d) + pd.Timedelta(days=1),
                "px_sell": row["next_close"],
                "ret_gross": row["ret_close_to_next"],
                "ret_net": ret_net
            })

    tr = pd.DataFrame(trades).sort_values(["date_buy","symbol"]).reset_index(drop=True)
    tr.to_csv(os.path.join(out_dir, "trades_momentum_Tplus1_anti.csv"), index=False)

    # Portfolio daily ret & equity
    mom_daily = tr.groupby("date_buy")["ret_net"].mean().sort_index()
    mom_eq = (1.0 + mom_daily).cumprod()

    # Buy & Hold (equal-weight on traded universe)
    pivot_close = daily.pivot_table(index="date", columns="symbol", values="day_close", aggfunc="last").sort_index()
    univ = sorted(tr["symbol"].unique().tolist())
    bh_sym = pivot_close[univ].copy()
    bh_ret = bh_sym.pct_change().mean(axis=1)
    bh_daily = bh_ret.reindex(mom_daily.index).fillna(0.0)
    bh_eq = (1.0 + bh_daily).cumprod()

    # Train/Test split (70/30 by date count)
    dates = list(mom_daily.index)
    split_idx = int(len(dates) * 0.7)
    train_ret = mom_daily.reindex(dates[:split_idx])
    test_ret  = mom_daily.reindex(dates[split_idx:])

    # === Metrics CSVs ===
    full_total, full_ann, full_shp, full_dd, full_mar, full_mwr = summarize_series(mom_daily)
    bh_total,   bh_ann,   bh_shp,   bh_dd,   bh_mar,   bh_mwr   = summarize_series(bh_daily)
    train_total, train_ann, train_shp, train_dd, train_mar, train_mwr = summarize_series(train_ret)
    test_total,   test_ann,  test_shp,  test_dd,  test_mar,  test_mwr  = summarize_series(test_ret)

    pd.DataFrame([
        {"Set":"Train (70%)","TotalReturn":train_total,"Annualized":train_ann,"Sharpe":train_shp,"MaxDD":train_dd,"MAR":train_mar,"MonthlyWin":train_mwr},
        {"Set":"Test (30%)","TotalReturn":test_total,"Annualized":test_ann,"Sharpe":test_shp,"MaxDD":test_dd,"MAR":test_mar,"MonthlyWin":test_mwr},
        {"Set":"Full","TotalReturn":full_total,"Annualized":full_ann,"Sharpe":full_shp,"MaxDD":full_dd,"MAR":full_mar,"MonthlyWin":full_mwr},
    ]).to_csv(os.path.join(out_dir, "train_test_metrics.csv"), index=False)

    pd.DataFrame([
        {"Strategy":"Momentum (T+1) - Full","TotalReturn":full_total,"Annualized":full_ann,"Sharpe":full_shp,"MaxDD":full_dd,"MAR":full_mar,"MonthlyWin":full_mwr},
        {"Strategy":"Buy&Hold (EW) - Full","TotalReturn":bh_total,"Annualized":bh_ann,"Sharpe":bh_shp,"MaxDD":bh_dd,"MAR":bh_mar,"MonthlyWin":bh_mwr},
    ]).to_csv(os.path.join(out_dir, "metrics_full_compare.csv"), index=False)

    # === Plots ===
    plt.figure(figsize=(10,6))
    plt.plot(mom_eq.index, mom_eq.values, label="Momentum (T+1) - Net")
    plt.plot(bh_eq.index,  bh_eq.values,  label="Buy & Hold (Equal-Weight)")
    plt.legend(); plt.title("Equity Curves (Full) — Momentum vs Buy&Hold"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "equity_momentum_vs_bh.png")); plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(((1+train_ret).cumprod()).index, ((1+train_ret).cumprod()).values)
    plt.title("Train Equity — Momentum (T+1)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "momentum_eq_train.png")); plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(((1+test_ret).cumprod()).index, ((1+test_ret).cumprod()).values)
    plt.title("Test Equity — Momentum (T+1)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "momentum_eq_test.png")); plt.close()

    # Buy/Sell markers（portfolio-level）
    plt.figure(figsize=(10,6))
    plt.plot(mom_eq.index, mom_eq.values, label="Momentum (T+1) Equity")
    buy_dates = sorted(tr["date_buy"].unique().tolist())
    buy_x = pd.to_datetime(buy_dates)
    buy_y = mom_eq.reindex(buy_x).values
    plt.scatter(buy_x, buy_y, marker="^", color="green", s=20, label="Buy @ Close (T)")
    sell_x = pd.to_datetime(buy_x) + pd.to_timedelta(1, unit="D")
    sell_y = mom_eq.reindex(sell_x).values
    plt.scatter(sell_x, sell_y, marker="v", color="red", s=20, label="Sell @ Close (T+1)")
    plt.legend(); plt.title("Momentum (T+1) — Equity with Buy/Sell Markers"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "momentum_buy_sell_markers.png")); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to the 1-min A-shares ZIP")
    ap.add_argument("--out", default="./out", help="Output directory")
    ap.add_argument("--cutoff", default="14:45")
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--fee", type=float, default=0.0003)
    args = ap.parse_args()
    run(args.zip, args.out, cutoff=args.cutoff, top_n=args.top_n, fee=args.fee)
