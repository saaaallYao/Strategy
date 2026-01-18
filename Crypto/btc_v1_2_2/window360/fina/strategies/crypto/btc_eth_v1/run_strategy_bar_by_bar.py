#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar-by-bar (minute-by-minute) execution for BTC-ETH-SOL stat-arb
- Mimics online/production execution
- Logs trades, equity, and key metrics
- Output format similar to reference crypto_stat_arb_online.py
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from .data_manager import CryptoDataManager
from .strategy_engine import StatArbEngine
from fina.core.time_util import TimeUtil


def run_bar_by_bar(start_date, end_date, log_dir="data/logs"):
    # 1. 数据准备
    config = {'symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD']}
    dm = CryptoDataManager(config)
    px, _ = dm.load_and_align_data(start_date, end_date)
    if px.empty:
        print("[ERROR] No data available!")
        return
    print(f"[INFO] Data loaded: {px.shape}")

    # 2. 策略参数
    strategy_config = {
        'window': 360,
        'z_enter': 1.2,
        'z_exit': 0.4,
        'signal_persistence': 3,
        'min_hold_bars': 30,
        'cooldown_bars': 30,
        'scale_base': 0.05,
        'inv_cap': 0.15,
        'fee': 5e-4,
        'stop_loss_pct': 0.01,
        'clip_resid0': 800,
        'clip_beta': 6,
        'min_edge_return': 0.0014,
        'dyn_edge_enabled': True,
        'dyn_edge_fee_mult': 5.0,
        'dyn_edge_vol_mult': 0.5,
        'fee_exit_enabled': True,
        'fee_exit_mult': 2.0,
        'beta_update_step': 1,
        'dyn_exit_enabled': False,
        'dyn_exit_min_mult': 0.5,
        'dyn_exit_decay_per_bar': 0.001,
        'vol_regime_enabled': False,
        'vol_regime_low_pct': 0.35,
        'vol_regime_high_pct': 0.65,
        'z_enter_low_mult': 0.9,
        'z_enter_high_mult': 1.1,
        'z_exit_low_mult': 0.9,
        'z_exit_high_mult': 1.1,
        'tiered_entry_enabled': False,
        'tier1_enter': 1.2,
        'tier2_enter': 1.8,
        'tier1_size': 0.6,
        'tier2_size': 1.0,
        'tier2_requires_confirmation': True,
        'resid_ema_enabled': False,
        'resid_ema_span': 30,
        'nonlinear_pos_enabled': False,
        'nonlinear_pos_alpha': 1.5,
        'nonlinear_pos_min': 0.4,
        'nonlinear_pos_max': 1.0
    }
    engine = StatArbEngine(strategy_config)

    # 3. 逐bar推进
    results = []
    trades = []
    equity = [1.0]
    position = 0
    entry_price = 0
    entry_time = None
    last_pnl = 0
    last_signal = 0
    long_count = 0
    short_count = 0
    last_exit_time = None
    # 预先计算rolling beta和resid
    sub, resid, beta = engine.rolling_beta_resid(px)
    clip_dyn, scale_dyn = engine.calculate_dynamic_adjustments(resid)
    mu = resid.rolling(engine.window).mean()
    sig = resid.rolling(engine.window).std()
    z = (resid - mu) / sig
    vol_day = resid.rolling(1440).std()
    vol_low = None
    vol_high = None
    if engine.config.get('vol_filter_pct_low') is not None:
        vol_low = vol_day.quantile(engine.config['vol_filter_pct_low'])
    if engine.config.get('vol_filter_pct_high') is not None:
        vol_high = vol_day.quantile(engine.config['vol_filter_pct_high'])
    vol_regime_low = None
    vol_regime_high = None
    if engine.config.get('vol_regime_enabled'):
        low_pct = engine.config.get('vol_regime_low_pct', 0.35)
        high_pct = engine.config.get('vol_regime_high_pct', 0.65)
        vol_regime_low = vol_day.quantile(low_pct)
        vol_regime_high = vol_day.quantile(high_pct)
    btc_price = sub["BTC_USD"].reindex(resid.index).ffill()
    trend_slope = None
    lookback = engine.config.get('entry_trend_lookback_min')
    if lookback:
        trend_slope = btc_price.pct_change(int(lookback)).fillna(0)
    # 动态持仓
    for i in range(engine.window, len(sub)):
        ts = sub.index[i]
        price = sub["BTC_USD"].iloc[i]
        z_score = z.iloc[i]
        scale = scale_dyn.iloc[i]
        if pd.isna(scale):
            scale = 0.0
        expected_return = None
        if sig.iloc[i] == sig.iloc[i] and price:
            expected_move = abs(z_score) * sig.iloc[i]
            expected_return = expected_move / price
        # 信号（加入持仓持久性、最小持仓和冷却时间以降低换手）
        trend_strong = False
        trend_dir = 0
        slope_th = engine.config.get('entry_trend_slope_threshold')
        if trend_slope is not None and slope_th is not None:
            slope_val = trend_slope.iloc[i]
            if slope_val >= slope_th:
                trend_strong = True
                trend_dir = 1
            elif slope_val <= -slope_th:
                trend_strong = True
                trend_dir = -1

        z_enter_eff = engine.z_enter * (engine.config.get('trend_entry_scale', 1.0) if trend_strong else 1.0)
        if engine.config.get('vol_regime_enabled') and vol_regime_low is not None and vol_regime_high is not None:
            vol_val = vol_day.iloc[i]
            if vol_val <= vol_regime_low:
                z_enter_eff *= engine.config.get('z_enter_low_mult', 0.9)
            elif vol_val >= vol_regime_high:
                z_enter_eff *= engine.config.get('z_enter_high_mult', 1.1)

        if z_score < -z_enter_eff:
            long_count += 1
        else:
            long_count = 0
        if z_score > z_enter_eff:
            short_count += 1
        else:
            short_count = 0

        signal = 0
        z_exit_eff = engine.z_exit
        if engine.config.get('dyn_exit_enabled') and position != 0 and entry_time is not None:
            held_minutes = (ts - entry_time).total_seconds() / 60.0
            decay = 1.0 - (engine.config.get('dyn_exit_decay_per_bar', 0.001) * held_minutes)
            decay = max(engine.config.get('dyn_exit_min_mult', 0.5), decay)
            z_exit_eff = z_exit_eff * decay
        if engine.config.get('vol_regime_enabled') and vol_regime_low is not None and vol_regime_high is not None:
            vol_val = vol_day.iloc[i]
            if vol_val <= vol_regime_low:
                z_exit_eff *= engine.config.get('z_exit_low_mult', 0.9)
            elif vol_val >= vol_regime_high:
                z_exit_eff *= engine.config.get('z_exit_high_mult', 1.1)

        if position == 0:
            if long_count >= engine.signal_persistence:
                signal = 1
            elif short_count >= engine.signal_persistence:
                signal = -1
        else:
            if abs(z_score) < z_exit_eff:
                signal = 0
            else:
                signal = 1 if position > 0 else -1

        if engine.config.get('fee_exit_enabled') and expected_return is not None and position != 0:
            if expected_return > (engine.config.get('fee_exit_mult', 2.0) * engine.fee):
                signal = 1 if position > 0 else -1

        if last_exit_time is not None and position == 0:
            cooldown_end = last_exit_time + pd.Timedelta(minutes=engine.cooldown_bars)
            if ts < cooldown_end:
                signal = 0

        if position != 0 and entry_time is not None:
            held_minutes = (ts - entry_time).total_seconds() / 60.0
            if held_minutes < engine.min_hold_bars and signal == 0:
                signal = 1 if position > 0 else -1
        # Optional entry filters (only block new entries)
        if position == 0 and signal != 0:
            if engine.config.get('trend_only') and not trend_strong:
                signal = 0
            if signal != 0 and vol_low is not None and vol_day.iloc[i] < vol_low:
                signal = 0
            if signal != 0 and vol_high is not None and vol_day.iloc[i] > vol_high:
                signal = 0
            if signal != 0 and engine.config.get('min_edge_return') is not None:
                expected_move = abs(z_score) * sig.iloc[i]
                expected_return = expected_move / price if price else 0
                if engine.config.get('dyn_edge_enabled'):
                    required_edge = (engine.config.get('dyn_edge_fee_mult', 2.0) * engine.fee)
                    required_edge += engine.config.get('dyn_edge_vol_mult', 0.0) * (sig.iloc[i] / price if price else 0)
                    if expected_return < required_edge:
                        signal = 0
                elif expected_return < engine.config['min_edge_return']:
                    signal = 0

        # 动态持仓
        pos = signal * scale
        if signal != 0 and engine.config.get('tiered_entry_enabled'):
            tier1 = engine.config.get('tier1_enter', 1.2)
            tier2 = engine.config.get('tier2_enter', 1.8)
            tier1_size = engine.config.get('tier1_size', 0.6)
            tier2_size = engine.config.get('tier2_size', 1.0)
            tier_size = tier2_size if abs(z_score) >= tier2 else tier1_size
            pos = signal * scale * tier_size
        elif signal != 0 and engine.config.get('nonlinear_pos_enabled'):
            z_enter = engine.z_enter
            z_abs = abs(z_score)
            if z_abs <= z_enter:
                size_mult = engine.config.get('nonlinear_pos_min', 0.4)
            else:
                z_ratio = z_abs / z_enter
                alpha = engine.config.get('nonlinear_pos_alpha', 1.5)
                size_mult = engine.config.get('nonlinear_pos_min', 0.4) + (
                    (z_ratio ** alpha) - 1.0
                ) * (
                    engine.config.get('nonlinear_pos_max', 1.0)
                    - engine.config.get('nonlinear_pos_min', 0.4)
                )
            size_mult = max(engine.config.get('nonlinear_pos_min', 0.4),
                            min(engine.config.get('nonlinear_pos_max', 1.0), size_mult))
            pos = signal * scale * size_mult
        if pd.isna(pos) or np.isinf(pos):
            pos = 0.0
        pos = max(min(pos, engine.inv_cap), -engine.inv_cap)
    # 交易逻辑
    trade = None
    if position == 0 and pos != 0:
        # 开仓
        fee_cost = engine.fee * abs(pos - position)
        equity[-1] *= (1 - fee_cost)
        position = pos
        entry_price = price
        entry_time = ts
        trade = {
            'timestamp': str(ts),
            'action': 'open',
            'price': float(price),
            'position': float(pos),
            'pnl': float(-fee_cost),
            'gross_pnl': 0.0,
            'fee': float(fee_cost),
            'zscore': float(z_score),
        }
    elif position != 0 and pos == 0:
        # 平仓
        pnl = (price - entry_price) / entry_price if position > 0 else (entry_price - price) / entry_price
        fee_cost = engine.fee * abs(pos - position)
        net_pnl = pnl - fee_cost
        last_pnl = net_pnl
        trade = {
            'timestamp': str(ts),
            'action': 'close',
            'price': float(price),
            'position': float(position),
            'pnl': float(net_pnl),
            'gross_pnl': float(pnl),
            'fee': float(fee_cost),
            'zscore': float(z_score),
        }
        position = 0
        entry_price = 0
        entry_time = None
        last_exit_time = ts
    elif position != 0 and (pos * position < 0):
        # 反向平仓再开仓
        pnl = (price - entry_price) / entry_price if position > 0 else (entry_price - price) / entry_price
        fee_cost = engine.fee * abs(pos - position)
        net_pnl = pnl - fee_cost
        last_pnl = net_pnl
        trade = {
            'timestamp': str(ts),
            'action': 'reverse',
            'price': float(price),
            'position': float(position),
            'pnl': float(net_pnl),
            'gross_pnl': float(pnl),
            'fee': float(fee_cost),
            'zscore': float(z_score),
        }
        position = pos
        entry_price = price
        entry_time = ts
        # 记录结果
        equity.append(equity[-1] * (1 + last_pnl))
        results.append({
            'timestamp': str(ts),
            'price': float(price),
            'zscore': float(z_score),
            'signal': int(signal),
            'position': float(position),
            'equity': float(equity[-1])
        })
        if trade:
            trades.append(trade)
        last_signal = signal
        last_pnl = 0  # 只在平仓/反向时计入
    # 4. 输出日志
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"btc_eth_bar_by_bar_{timestamp}.json"
    with open(log_file, 'w') as f:
        json.dump({'results': results, 'trades': trades}, f, indent=2)
    print(f"[INFO] Bar-by-bar log saved: {log_file}")
    print(f"[INFO] Total trades: {len(trades)}")
    print(f"[INFO] Final equity: {equity[-1]:.4f}")
    return results, trades, equity


if __name__ == "__main__":
    # 默认跑最近两天
    end = TimeUtil.today_market_date()
    start = TimeUtil.days_ago_market_date(2)
    run_bar_by_bar(start, end) 
