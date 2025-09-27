#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run 15m Breakout + ATR (single asset, unchanged core logic).
Generates an English PDF report (Train/Test/Full), metrics CSV, and trades CSV.

Usage:
  python run_breakout15m_single_asset.py --csv 600519_SH_1min.csv --look 10 --atr_n 20 --atr_k 1.4
"""
import argparse, os, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FEE = 0.0003  # 0.03% per side
CN_MARKET_MINUTES_PER_DAY = 240
TRADING_DAYS_PER_YEAR = 252

def load_csv(path: str, need_ohlc: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')]
    if need_ohlc:
        return df[['symbol','open','high','low','close','volume']].copy()
    else:
        return df[['symbol','close']].copy()

def annualize(mean_r: float, std_r: float, periods_per_year: int) -> float:
    return np.nan if std_r == 0 else (mean_r / std_r) * np.sqrt(periods_per_year)

def equity_metrics(returns: pd.Series, periods_per_year: int):
    r = returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    eq = (1 + r).cumprod()
    n = len(r)
    if n == 0:
        return {"Ann. Return": np.nan,"Sharpe": np.nan,"Max DD": np.nan,"MAR": np.nan,"Months Win %": np.nan,"Bars": 0}, eq
    ann_return = eq.iloc[-1] ** (periods_per_year / n) - 1
    sharpe = annualize(r.mean(), r.std(ddof=0), periods_per_year)
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    mar = np.nan if (dd == 0 or np.isnan(dd)) else ann_return / abs(dd)
    monthly = (1 + r).resample('M').prod() - 1
    months_win = (monthly > 0).mean() if len(monthly) > 0 else np.nan
    return {"Ann. Return": ann_return,"Sharpe": sharpe,"Max DD": dd,"MAR": mar,"Months Win %": months_win,"Bars": n}, eq

def resample_ohlc(df, rule='15T'):
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    v = df['volume'].resample(rule).sum()
    out = pd.concat([o.rename('open'), h.rename('high'), l.rename('low'), c.rename('close'), v.rename('volume')], axis=1).dropna()
    return out

def atr(df_ohlc: pd.DataFrame, n=20) -> pd.Series:
    prev_close = df_ohlc['close'].shift(1)
    tr1 = df_ohlc['high'] - df_ohlc['low']
    tr2 = (df_ohlc['high'] - prev_close).abs()
    tr3 = (df_ohlc['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def strat_breakout_15m(df_raw: pd.DataFrame, look=10, atr_n=20, atr_k=1.4):
    # Unchanged core logic from your original code
    ohlc = resample_ohlc(df_raw, '15T')
    hi = ohlc['close'].rolling(look).max().shift(1)
    lo = ohlc['close'].rolling(look).min().shift(1)
    A = atr(ohlc, atr_n)
    r = ohlc['close'].pct_change().fillna(0.0)

    pos = pd.Series(0.0, index=ohlc.index)
    entry_day = None
    entry_price = None

    for t in ohlc.index:
        prev_t = t - pd.Timedelta(minutes=15)
        prev_pos = pos.loc[prev_t] if prev_t in pos.index else 0.0
        new_pos = prev_pos

        if prev_pos == 0.0:
            if (not np.isnan(hi.loc[t])) and (ohlc.loc[t,'close'] > hi.loc[t]):
                new_pos = 1.0
                entry_day = t.date()
                entry_price = ohlc.loc[t,'close']
            elif (not np.isnan(lo.loc[t])) and (ohlc.loc[t,'close'] < lo.loc[t]):
                new_pos = -1.0
                entry_day = t.date()
                entry_price = ohlc.loc[t,'close']
        else:
            day_advanced = (t.date() > entry_day) if entry_day else False
            if day_advanced:
                if prev_pos == 1.0:
                    if ohlc.loc[t,'close'] < entry_price - atr_k * A.loc[t]:
                        new_pos = 0.0; entry_day=None; entry_price=None
                else:
                    if ohlc.loc[t,'close'] > entry_price + atr_k * A.loc[t]:
                        new_pos = 0.0; entry_day=None; entry_price=None
                if new_pos != 0.0 and t.time() >= pd.Timestamp('14:45').time():
                    new_pos = 0.0; entry_day=None; entry_price=None

        pos.loc[t] = new_pos

    strat_ret = pos.shift(1).fillna(0.0) * r
    delta = pos.diff().abs().fillna(abs(pos.fillna(0.0)))
    fee_series = FEE * 1.0 * delta
    strat_ret = (strat_ret - fee_series).fillna(0.0)
    return {"ohlc": ohlc, "pos": pos, "ret": strat_ret}

def buy_hold_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)

def split_by_ratio(index: pd.DatetimeIndex, train_ratio=0.7):
    start, end = index[0], index[-1]
    split_ts = start + (end - start) * train_ratio
    return split_ts

def extract_trades(pos: pd.Series):
    pos_shift = pos.shift(1).fillna(0.0)
    entry_idx = pos[(pos_shift==0) & (pos!=0)].index
    exit_idx  = pos[(pos_shift!=0) & (pos==0)].index
    entries = [(t, pos.loc[t]) for t in entry_idx]
    exits   = [(t, pos_shift.loc[t]) for t in exit_idx]
    return entries, exits

def plot_equity(eq, eq_bh, title, pdf):
    plt.figure(figsize=(10,4))
    eq.plot(label="Strategy")
    eq_bh.plot(label="Buy&Hold")
    plt.title(title); plt.xlabel("Time"); plt.ylabel("Equity (normalized)"); plt.legend()
    plt.tight_layout(); pdf.savefig(); plt.close()

def plot_price_with_trades(ohlc, entries, exits, title, pdf):
    plt.figure(figsize=(10,4))
    ohlc['close'].plot()
    for t, side in entries:
        y = ohlc.loc[t,'close']
        plt.scatter([t],[y], marker='^' if side>0 else 'v', color='green' if side>0 else 'red', s=40, label='Buy (Long Entry)' if side>0 else 'Sell (Short Entry)')
    for t, side in exits:
        y = ohlc.loc[t,'close']
        plt.scatter([t],[y], marker='v' if side>0 else '^', color='red' if side>0 else 'green', s=40, label='Sell (Long Exit)' if side>0 else 'Buy (Cover Short)')
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles)); plt.legend(uniq.values(), uniq.keys(), ncol=2)
    plt.title(title); plt.xlabel("Time"); plt.ylabel("Price")
    plt.tight_layout(); pdf.savefig(); plt.close()

def plot_trade_frequency(entries, title, pdf):
    if len(entries)==0:
        plt.figure(figsize=(10,3)); plt.title(title+" (no trades)"); pdf.savefig(); plt.close(); return
    times = pd.to_datetime([t for t,_ in entries])
    by_month = pd.Series(1, index=times).resample('M').sum()
    plt.figure(figsize=(10,3)); by_month.plot(kind='bar')
    plt.title(title); plt.xlabel("Month"); plt.ylabel("# Trades")
    plt.tight_layout(); pdf.savefig(); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to *_1min.csv for a single stock (e.g., 600519_SH_1min.csv)")
    ap.add_argument("--look", type=int, default=10)
    ap.add_argument("--atr_n", type=int, default=20)
    ap.add_argument("--atr_k", type=float, default=1.4)
    args = ap.parse_args()

    df_raw = load_csv(args.csv, need_ohlc=True)
    out = strat_breakout_15m(df_raw, look=args.look, atr_n=args.atr_n, atr_k=args.atr_k)
    ohlc, pos, rets = out['ohlc'], out['pos'], out['ret']
    periods_per_year = (CN_MARKET_MINUTES_PER_DAY//15)*TRADING_DAYS_PER_YEAR

    m_full, eq_full = equity_metrics(rets, periods_per_year)
    bh_ret = buy_hold_returns(ohlc['close'])
    m_bh_full, eq_bh_full = equity_metrics(bh_ret, periods_per_year)

    split_ts = split_by_ratio(ohlc.index, 0.7)
    mask_train = ohlc.index <= split_ts
    rets_train = rets.loc[mask_train]
    rets_test  = rets.loc[~mask_train]
    bh_train = bh_ret.loc[rets_train.index]
    bh_test  = bh_ret.loc[rets_test.index]

    m_train, eq_train = equity_metrics(rets_train, periods_per_year)
    m_test,  eq_test  = equity_metrics(rets_test,  periods_per_year)
    m_bh_train, eq_bh_train = equity_metrics(bh_train, periods_per_year)
    m_bh_test,  eq_bh_test  = equity_metrics(bh_test,  periods_per_year)

    entries, exits = extract_trades(pos)

    # Save CSVs
    base_dir = os.path.dirname(os.path.abspath(args.csv))
    metrics_csv = os.path.join(base_dir, "Breakout15m_metrics.csv")
    trades_csv  = os.path.join(base_dir, "Breakout15m_trades.csv")
    summary_rows = [
        {"Bucket":"Strategy Full"} | m_full,
        {"Bucket":"BH Full"} | m_bh_full,
        {"Bucket":"Strategy Train"} | m_train,
        {"Bucket":"BH Train"} | m_bh_train,
        {"Bucket":"Strategy Test"} | m_test,
        {"Bucket":"BH Test"} | m_bh_test,
    ]
    pd.DataFrame(summary_rows).to_csv(metrics_csv, index=False)
    pd.DataFrame({
        "timestamp": [t for t,_ in entries] + [t for t,_ in exits],
        "type":      ["entry"]*len(entries) + ["exit"]*len(exits),
        "side":      [int(s) for _,s in entries] + [int(s) for _,s in exits],
    }).sort_values("timestamp").to_csv(trades_csv, index=False)

    # PDF report
    pdf_path = os.path.join(base_dir, "Breakout15m_Report_EN.pdf")
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(10,6)); plt.axis('off')
        txt = f"""
        15-Min Breakout + ATR — Single Asset
        CSV: {os.path.basename(args.csv)}
        Params: look={args.look}, ATR_n={args.atr_n}, ATR_k={args.atr_k}
        Rules:
          • Entry at next bar after close breakout (Donchian close, shift(1)).
          • Exit from T+1 via ATR stop; hard flat at 14:45 next day.
          • Position change takes effect next bar; fee {FEE*100:.3f}%/side at position change.
        Split:
          • Train ≤ {split_ts}
          • Test  > {split_ts}
        Full Metrics:
          • Strategy — AnnRet={m_full['Ann. Return']:.2%}, Sharpe={m_full['Sharpe']:.2f}, MaxDD={m_full['Max DD']:.2%}, MAR={m_full['MAR']:.2f}
          • Buy&Hold — AnnRet={m_bh_full['Ann. Return']:.2%}, Sharpe={m_bh_full['Sharpe']:.2f}
        """
        plt.text(0.02,0.98,textwrap.dedent(txt),va='top',ha='left',fontsize=10,family='monospace'); pdf.savefig(); plt.close()

        plot_equity(eq_full, eq_bh_full, "Equity — Full (Strategy vs Buy&Hold)", pdf)
        plot_equity(eq_train, eq_bh_train, "Equity — Train", pdf)
        plot_equity(eq_test,  eq_bh_test,  "Equity — Test", pdf)
        plot_price_with_trades(ohlc, entries, exits, "Price with Trade Markers (Green=Buy/Cover, Red=Sell/Short)", pdf)
        plot_trade_frequency(entries, "Trade Frequency per Month", pdf)

    print("Saved:", pdf_path)
    print("Metrics:", metrics_csv)
    print("Trades:", trades_csv)

if __name__ == "__main__":
    main()
