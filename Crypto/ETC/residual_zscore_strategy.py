import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import zipfile, io

FREQ_PER_YEAR = 24*365

def compute_metrics(returns, freq_per_year=FREQ_PER_YEAR):
    r = returns.dropna()
    sharpe = np.nan if (len(r)==0 or r.std()==0) else (r.mean()/r.std())*np.sqrt(freq_per_year)
    eq = (1+r).cumprod()
    running_max = eq.cummax()
    dd = eq/running_max - 1.0
    max_dd = dd.min() if len(dd)>0 else np.nan
    if len(r)==0:
        ann = np.nan
    else:
        years = len(r)/freq_per_year
        cum_factor = (1+r).prod()
        ann = cum_factor**(1/years) - 1 if years>0 else np.nan
    monthly = (1+r).groupby(pd.Grouper(freq='M')).prod()-1
    win_rate = (monthly>0).mean() if len(monthly)>0 else np.nan
    mar = (ann/abs(max_dd)) if (max_dd is not None and max_dd<0) else np.nan
    return {"Annualized Gross Return": ann, "Sharpe": sharpe, "Max Drawdown": max_dd,
            "MAR Ratio": mar, "Win Rate (Monthly)": win_rate, "Num Periods": len(r)}, eq

def residual_strategy(prices_df, coin, ref='BTC_USD', window=168, z_th=1.0, fee_per_change=0.0003):
    df = prices_df[[coin, ref]].dropna().copy()
    df['log_coin'] = np.log(df[coin]); df['log_ref']  = np.log(df[ref])
    roll = window
    cov = df['log_coin'].rolling(roll).cov(df['log_ref'])
    var = df['log_ref'].rolling(roll).var()
    beta = cov/var
    mu_coin = df['log_coin'].rolling(roll).mean()
    mu_ref  = df['log_ref'].rolling(roll).mean()
    intercept = mu_coin - beta*mu_ref
    spread = df['log_coin'] - (intercept + beta*df['log_ref'])
    m = spread.rolling(roll).mean()
    s = spread.rolling(roll).std()
    z = (spread - m)/s
    df['z'] = z

    z_prev = df['z'].shift(1)
    entry_short = (z_prev <= z_th) & (df['z'] > z_th)
    entry_long  = (z_prev >= -z_th) & (df['z'] < -z_th)
    exit_any = ((z_prev <= 0) & (df['z'] > 0)) | ((z_prev >= 0) & (df['z'] < 0))

    pos_raw = np.zeros(len(df))
    state = 0
    for i in range(len(df)):
        if state == 0:
            if entry_long.iloc[i]: state = 1
            elif entry_short.iloc[i]: state = -1
        else:
            if exit_any.iloc[i]: state = 0
        pos_raw[i] = state
    df['pos_raw'] = pos_raw
    df['pos'] = df['pos_raw'].shift(1).fillna(0.0)

    ret = df[coin].pct_change().fillna(0.0)
    strat_r = df['pos']*ret

    turnover = df['pos'].diff().abs().fillna(0.0)
    cost = turnover*fee_per_change
    strat_r_after_fee = strat_r - cost

    out = df[[coin, ref, 'z', 'pos', 'pos_raw']].copy()
    out['ret'] = ret
    out['strat_r'] = strat_r_after_fee
    out['fee'] = cost
    out['entry_marker'] = df['pos_raw'].diff().replace(0, np.nan)
    return out

def run_from_zip(zip_path="crypto_data.zip",
                 inner_csv="crypto_data.csv",
                 coin="ETCUSDT", ref="BTC_USD",
                 window=168, z_th=1.0, fee=0.0003):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(inner_csv) as f:
            raw = pd.read_csv(f)
    raw['open_time'] = pd.to_datetime(raw['open_time'], utc=True)
    raw = raw.set_index('open_time').sort_index()
    prices = raw.select_dtypes(include=[np.number]).copy()
    h = prices.resample('1H').last().dropna(how='all').fillna(method='ffill')
    res = residual_strategy(h, coin, ref, window, z_th, fee)
    ser = res['strat_r'].dropna()
    split_t = ser.index.min() + (ser.index.max()-ser.index.min())*0.7
    train_idx = ser.index <= split_t; test_idx = ser.index > split_t
    met_train,_ = compute_metrics(ser[train_idx]); met_test,_ = compute_metrics(ser[test_idx]); met_full,_ = compute_metrics(ser)

    px = h[coin].reindex(ser.index).ffill()
    bh = px.pct_change().fillna(0.0)
    bh_train,_ = compute_metrics(bh[train_idx]); bh_test,_ = compute_metrics(bh[test_idx]); bh_full,_ = compute_metrics(bh)

    return {'strategy': {'train': met_train, 'test': met_test, 'full': met_full},
            'buyhold': {'train': bh_train, 'test': bh_test, 'full': bh_full},
            'series': {'strategy': ser, 'buyhold': bh}, 'res': res}

if __name__ == "__main__":
    # === 内置路径与参数（按你要求写死） ===
    ZIP_PATH   = "crypto_data.zip"   # 放在脚本同目录；如放别处，改成绝对路径
    INNER_CSV  = "crypto_data.csv"
    COIN       = "ETCUSDT"
    REF        = "BTC_USD"
    WINDOW     = 168       # 7天
    Z_TH       = 1.0
    FEE        = 0.0003    # 0.03% per side

    out = run_from_zip(ZIP_PATH, INNER_CSV, COIN, REF, WINDOW, Z_TH, FEE)

    # ========= 1) 导出汇总指标到一个 CSV =========
    # out['strategy'/'buyhold']['train'/'test'/'full'] 里已经是 compute_metrics 的字典
    rows = []
    for model in ["strategy", "buyhold"]:
        for seg in ["train", "test", "full"]:
            row = {"Model": model, "Segment": seg}
            row.update(out[model][seg])  # Annualized Gross Return / Sharpe / Max Drawdown / MAR Ratio / Win Rate (Monthly) / Num Periods
            rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv("metrics_summary.csv", index=False)
    print("Saved metrics to metrics_summary.csv")

    # ========= 2) 画 Equity Curve（策略 vs 买入并持有） =========
    strat_r = out["series"]["strategy"].copy()                # 策略小时收益
    bh_r    = out["series"]["buyhold"].reindex(strat_r.index) # 对齐后的买入并持有小时收益
    bh_r    = bh_r.fillna(0.0)

    eq_strat = (1.0 + strat_r).cumprod()
    eq_bh    = (1.0 + bh_r).cumprod()

    # 计算 70/30 分割线位置（与 run_from_zip 的 split 口径一致）
    split_t = strat_r.index.min() + (strat_r.index.max() - strat_r.index.min()) * 0.7

    plt.figure(figsize=(10, 5))
    plt.plot(eq_strat.index, eq_strat, label=f"Strategy ({COIN})")
    plt.plot(eq_bh.index, eq_bh, label=f"Buy & Hold ({COIN})")
    plt.axvline(split_t, linestyle="--", alpha=0.8)
    plt.title(f"{COIN}: Equity Curve (Strategy vs Buy & Hold)")
    plt.xlabel("Time"); plt.ylabel("Equity (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=160)
    print("Saved equity curve to equity_curve.png")
