import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from datetime import time

# ============= User params =============
CSV_PATH = Path("universe.csv")   # <-- 改成你的CSV路径；需包含 timestamp 列和若干资产价格列
LOOKBACK_FAST = 24                 # 快线窗口（5min bar）
LOOKBACK_SLOW = 96                 # 慢线窗口（5min bar）
REBALANCE_EVERY = 12               # 每 12 根5分钟bar再平衡 = 60分钟
Q = 0.2                            # 每侧（多/空）选取比例（按N个资产中的Top/Bottom）
COST_RATE = 0.0003                 # 0.03%/次（按换手率计）
SIDE_PRUNE_CORR = 0.5              # 训练期低相关择股阈值的起始值（会自适应放宽）
TRAIN_RATIO = 0.7                  # 70% 训练，30% 测试
VOL_WIN = 10 * 78                  # 波动估计窗口（10个RTH日 * 78根5min bar）
K_LIST = [10, 6, 5, 4, 3, 2]      # 依次跑不同的K（资产池大小）

BARS_PER_DAY_5M = 78
BARS_PER_YEAR = 252 * BARS_PER_DAY_5M

# ============= Helpers =============
def greedy_lowcorr_selection(corr_mat, k=10, start_threshold=0.4, step=0.05, max_thr=0.9):
    """基于训练期相关矩阵的贪心低相关选股；若达不到k，会逐步放宽阈值。"""
    avg_corr = corr_mat.replace(1.0, np.nan).mean().sort_values()
    selected = []
    thr = start_threshold
    while len(selected) < k and thr <= max_thr:
        for name in avg_corr.index:
            if name in selected:
                continue
            if len(selected) == 0:
                selected.append(name)
            else:
                if all(abs(corr_mat.loc[name, s]) <= thr for s in selected):
                    selected.append(name)
            if len(selected) >= k:
                break
        if len(selected) < k:
            thr = round(thr + step, 3)
    return selected[:k], thr

def vol_norm_mom(rets, lb):
    """波动归一化动量：rolling mean / rolling std。"""
    m = rets.rolling(lb).mean()
    v = rets.rolling(lb).std(ddof=0).replace(0, np.nan)
    return (m / v).fillna(0.0)

def compute_metrics(equity, returns):
    """基于权益与每bar收益，输出 AnnReturn / Sharpe / MaxDD / MAR。"""
    n = len(returns)
    if n == 0:
        return dict(AnnReturn=np.nan, Sharpe=np.nan, MaxDD=np.nan, MAR=np.nan)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    ann = (1 + total) ** (BARS_PER_YEAR / max(n, 1)) - 1
    mu = returns.mean()
    sd = returns.std(ddof=0)
    sharpe = (mu / sd) * np.sqrt(BARS_PER_YEAR) if sd > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    mdd = dd.min() if len(dd) else np.nan
    mar = ann / abs(mdd) if (mdd is not None and mdd < 0) else np.nan
    return dict(AnnReturn=ann, Sharpe=sharpe, MaxDD=mdd, MAR=mar)

def annual_breakdown(equity, returns, label):
    """按年份汇总年化收益/Sharpe/MDD/MAR（供打印或需要时使用）。"""
    df = pd.DataFrame({"equity": equity, "ret": returns}).dropna()
    out = []
    for y, grp in df.groupby(df.index.year):
        eqy = grp["equity"]; r = grp["ret"]; n = len(r)
        total = eqy.iloc[-1] / eqy.iloc[0] - 1
        ann = (1 + total) ** (BARS_PER_YEAR / max(n, 1)) - 1
        mu = r.mean(); sd = r.std(ddof=0)
        shp = (mu / sd) * np.sqrt(BARS_PER_YEAR) if sd > 0 else np.nan
        dd = eqy / eqy.cummax() - 1.0; mdd = dd.min() if len(dd) else np.nan
        mar = ann / abs(mdd) if (mdd is not None and mdd < 0) else np.nan
        out.append(dict(Model=label, Year=int(y), AnnReturn=ann, Sharpe=shp, MaxDD=mdd, MAR=mar))
    return pd.DataFrame(out)

def save_equity_overlay_png(equity_dict, out_name="equity_all.png"):
    """把所有 K 的策略权益曲线叠加在同一张图（起点归一化为1）。"""
    plt.figure(figsize=(12, 6))
    for k, eq in sorted(equity_dict.items(), reverse=True):
        plt.plot(eq.index, (eq / eq.iloc[0]).values, label=f"K={k}")
    plt.title("Equity Curves (All K)")
    plt.xlabel("Time"); plt.ylabel("Equity (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close()

def save_single_equity_png(k, equity, bh_equity, m_strat_full, m_bh_full, fname=None):
    """单张图：策略 vs 等权BH 的权益曲线 + 右下角指标文本框。"""
    if fname is None:
        fname = f"equity_K{k}.png"
    plt.figure(figsize=(12, 5))
    equity.plot(label="Strategy Equity")
    bh_equity.plot(label="Equal-Weight BH", alpha=0.7)
    plt.title(f"Equity Curve (5-min) — K={k}")
    plt.xlabel("Time"); plt.ylabel("Equity (Start=1.0)")
    plt.legend(loc="upper left")
    txt = (
        "[Strategy]\n"
        f"AnnRet: {m_strat_full['AnnReturn']:.2%}\n"
        f"Sharpe: {m_strat_full['Sharpe']:.2f}\n"
        f"MaxDD: {m_strat_full['MaxDD']:.2%}\n"
        f"MAR: {m_strat_full['MAR']:.2f}\n\n"
        "[Equal-Weight BH]\n"
        f"AnnRet: {m_bh_full['AnnReturn']:.2%}\n"
        f"Sharpe: {m_bh_full['Sharpe']:.2f}\n"
        f"MaxDD: {m_bh_full['MaxDD']:.2%}\n"
        f"MAR: {m_bh_full['MAR']:.2f}"
    )
    plt.gca().text(0.99, 0.01, txt, transform=plt.gca().transAxes,
                   fontsize=9, va="bottom", ha="right",
                   bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"))
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def run_universe_k(prices_5, rets_5, rets_tr, k):
    """对给定K进行：训练期低相关选股 + 波动归一化动量 + 60min再平衡 + 成本扣减。"""
    # 训练期相关矩阵 & 低相关选股
    corr_tr = rets_tr.corr().fillna(0.0)
    universe, thr = greedy_lowcorr_selection(corr_tr, k, SIDE_PRUNE_CORR, 0.05, 0.9)
    prices_u = prices_5[universe]
    rets_u = rets_5[universe]

    # 信号（vol-norm momentum）：0.6*fast + 0.4*slow
    score = 0.6 * vol_norm_mom(rets_u, LOOKBACK_FAST) + 0.4 * vol_norm_mom(rets_u, LOOKBACK_SLOW)

    # 再平衡时间（每 REBALANCE_EVERY 根bar）
    bar_in_day = prices_u.groupby(prices_u.index.date).cumcount()
    rb_times = prices_u.index[(bar_in_day % REBALANCE_EVERY) == 0]
    S = score.reindex(rb_times).dropna(how="all")

    # 侧内逆波动归一化
    asset_vol = rets_u.rolling(VOL_WIN).std(ddof=0).replace(0, np.nan)
    IV = (1.0 / asset_vol).reindex(rb_times).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    N = len(universe)
    k_side = max(1, int(np.floor(Q * N)))  # 每侧选取数量

    rank_desc = S.rank(axis=1, method="first", ascending=False)
    rank_asc  = S.rank(axis=1, method="first", ascending=True)
    mask_long  = (rank_desc <= k_side).astype(float)
    mask_short = (rank_asc  <= k_side).astype(float)

    wL_raw = IV * mask_long
    wS_raw = IV * mask_short
    sumL = wL_raw.sum(axis=1).replace(0, np.nan)
    sumS = wS_raw.sum(axis=1).replace(0, np.nan)
    wL = (wL_raw.T / sumL).T.fillna(0.0) * 0.5
    wS = (wS_raw.T / sumS).T.fillna(0.0) * 0.5
    weights_rb = wL - wS                            # 再平衡时刻权重（中性：多0.5/空0.5）

    # 换手与成本（仅在再平衡时刻）
    dW = weights_rb.diff().abs().fillna(weights_rb.abs())
    turnover_rb = dW.sum(axis=1)
    cost_rb = turnover_rb * COST_RATE

    # 扩展到每bar并计算PnL
    weights_full = weights_rb.reindex(rets_u.index, method="ffill").fillna(0.0)
    cost_series = cost_rb.reindex(rets_u.index).fillna(0.0)

    port_ret = (weights_full * rets_u).sum(axis=1) - cost_series
    equity = (1.0 + port_ret).cumprod()

    # 基准：等权买入持有（同一Universe）
    bh_w = pd.Series(1.0 / N, index=universe)
    bh_ret = (rets_u * bh_w).sum(axis=1)
    bh_equity = (1.0 + bh_ret).cumprod()

    # 指标（Train/Test/Full）
    split_idx_k = int(len(equity) * TRAIN_RATIO)
    idx_tr = equity.index[:split_idx_k]; idx_te = equity.index[split_idx_k:]
    m_strat = {
        "Train": compute_metrics(equity.loc[idx_tr], port_ret.loc[idx_tr]),
        "Test":  compute_metrics(equity.loc[idx_te], port_ret.loc[idx_te]),
        "Full":  compute_metrics(equity, port_ret)
    }
    m_bh = {
        "Train": compute_metrics(bh_equity.loc[idx_tr], bh_ret.loc[idx_tr]),
        "Test":  compute_metrics(bh_equity.loc[idx_te], bh_ret.loc[idx_te]),
        "Full":  compute_metrics(bh_equity, bh_ret)
    }

    # 单K单图（策略 vs BH）
    save_single_equity_png(k, equity, bh_equity, m_strat["Full"], m_bh["Full"])

    # 汇总表（供汇总与CSV）
    df_strat = pd.DataFrame([
        dict(Slice="Train", **m_strat["Train"]),
        dict(Slice="Test",  **m_strat["Test"]),
        dict(Slice="Full",  **m_strat["Full"]),
    ])
    df_strat["Model"] = "Strategy"; df_strat["K"] = k
    df_bh = pd.DataFrame([
        dict(Slice="Train", **m_bh["Train"]),
        dict(Slice="Test",  **m_bh["Test"]),
        dict(Slice="Full",  **m_bh["Full"]),
    ])
    df_bh["Model"] = "Equal-Weight BH"; df_bh["K"] = k
    metrics_tbl = pd.concat([df_strat, df_bh], ignore_index=True)

    # 年度分解（可选，用于打印或调试）
    annual = pd.concat([
        annual_breakdown(equity, port_ret, "Strategy"),
        annual_breakdown(bh_equity, bh_ret, "Equal-Weight BH")
    ], ignore_index=True).sort_values(["Year", "Model"])
    annual["K"] = k

    # 返回结果与关键元信息（给“universes_params.csv”）
    params_row = {
        "K": k,
        "Universe": ", ".join(universe),
        "TrainCorrThreshold": thr,
        "LOOKBACK_FAST": LOOKBACK_FAST,
        "LOOKBACK_SLOW": LOOKBACK_SLOW,
        "REBALANCE_EVERY": REBALANCE_EVERY,
        "Q": Q,
        "COST_RATE": COST_RATE,
        "SIDE_PRUNE_CORR": SIDE_PRUNE_CORR,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VOL_WIN": VOL_WIN
    }

    return {
        "universe": universe,
        "threshold": thr,
        "equity": equity,
        "port_ret": port_ret,
        "bh_equity": bh_equity,
        "bh_ret": bh_ret,
        "metrics": metrics_tbl,
        "annual": annual,
        "params_row": params_row,
    }

def main():
    # -------- 数据预处理 --------
    raw = pd.read_csv(CSV_PATH, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    raw = raw.select_dtypes(include=[np.number])
    raw = raw.between_time(time(9,35), time(15,55))
    # 日内 forward-fill（避免跨日泄漏）
    raw = raw.groupby(raw.index.date, group_keys=False).apply(lambda x: x.ffill())

    prices_5 = raw.resample("5T").last().dropna(how="all")
    valid_frac = prices_5.notna().mean()
    keep_cols = valid_frac[valid_frac > 0.85].index.tolist()
    prices_5 = prices_5[keep_cols].dropna(how="any")
    rets_5 = prices_5.pct_change().fillna(0.0)

    split_idx = int(len(rets_5) * TRAIN_RATIO)
    rets_tr = rets_5.iloc[:split_idx]

    # -------- 跑不同 K --------
    metrics_frames = []
    annual_frames = []
    params_rows = []
    equity_dict = {}  # 叠加图所需

    for k in K_LIST:
        res = run_universe_k(prices_5, rets_5, rets_tr, k)
        metrics_frames.append(res["metrics"])
        annual_frames.append(res["annual"])
        params_rows.append(res["params_row"])
        equity_dict[k] = res["equity"]   # 单K图中已各自归一化，这里保存原曲线用于总图时再归一

    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    universes_params = pd.DataFrame(params_rows).sort_values("K", ascending=False)

    # -------- 保存CSV & 大图 --------
    metrics_all.to_csv("metrics_all.csv", index=False)
    universes_params.to_csv("universes_params.csv", index=False)
    save_equity_overlay_png(equity_dict, out_name="equity_all.png")

    # -------- 控制台摘要（可选） --------
    print("\nSaved files:")
    print(" - metrics_all.csv")
    print(" - universes_params.csv")
    print(" - equity_all.png")
    for k in sorted(equity_dict.keys(), reverse=True):
        print(f" - equity_K{k}.png")

if __name__ == "__main__":
    main()
