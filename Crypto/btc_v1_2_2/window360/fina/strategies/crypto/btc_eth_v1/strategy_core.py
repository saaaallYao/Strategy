#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Core for BTC-ETH Statistical Arbitrage
- Single source of truth for strategy logic
- Shared by both live trading and backtesting
- Ensures complete consistency between live and historical testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class StatArbStrategyCore:
    """
    Core strategy logic for BTC-ETH Statistical Arbitrage
    
    This is the single source of truth for all strategy calculations.
    Both live trading and backtesting must use this exact same implementation.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.window = config.get('window', 720)  # 12 hours * 60 minutes
        self.z_enter = config.get('z_enter', 1.2)
        self.z_exit = config.get('z_exit', 0.4)
        self.signal_persistence = config.get('signal_persistence', 1)
        self.min_hold_bars = config.get('min_hold_bars', 0)
        self.cooldown_bars = config.get('cooldown_bars', 0)
        self.vol_filter_pct_low = config.get('vol_filter_pct_low')
        self.vol_filter_pct_high = config.get('vol_filter_pct_high')
        self.entry_trend_lookback_min = config.get('entry_trend_lookback_min')
        self.entry_trend_slope_threshold = config.get('entry_trend_slope_threshold')
        self.min_edge_return = config.get('min_edge_return')
        self.trend_entry_scale = config.get('trend_entry_scale', 1.0)
        self.trend_only = config.get('trend_only', False)
        self.regime_switch_enabled = config.get('regime_switch_enabled', False)
        self.regime_vol_threshold_pct = config.get('regime_vol_threshold_pct')
        self.scale_base = config.get('scale_base', 0.05)
        self.inv_cap = config.get('inv_cap', 0.15)
        self.fee = config.get('fee', 5e-4)
        self.clip_resid0 = config.get('clip_resid0', 800)
        self.clip_beta = config.get('clip_beta', 6)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.007)  # 0.7% stop loss
        self.dyn_z_enabled = config.get('dyn_z_enabled', False)
        self.dyn_z_lookback = config.get('dyn_z_lookback', 1440)
        self.dyn_z_clip_low = config.get('dyn_z_clip_low', 0.8)
        self.dyn_z_clip_high = config.get('dyn_z_clip_high', 1.3)
        self.dyn_edge_enabled = config.get('dyn_edge_enabled', False)
        self.dyn_edge_fee_mult = config.get('dyn_edge_fee_mult', 2.0)
        self.dyn_edge_vol_mult = config.get('dyn_edge_vol_mult', 0.0)
        self.fee_exit_enabled = config.get('fee_exit_enabled', False)
        self.fee_exit_mult = config.get('fee_exit_mult', 2.0)
        self.scale_in_enabled = config.get('scale_in_enabled', False)
        self.scale_in_min = config.get('scale_in_min', 0.04)
        self.scale_in_slope = config.get('scale_in_slope', 0.038)
        self.scale_update_min_delta = config.get('scale_update_min_delta', 0.01)
        self.beta_update_step = config.get('beta_update_step', 1)
        self.dyn_exit_enabled = config.get('dyn_exit_enabled', False)
        self.dyn_exit_min_mult = config.get('dyn_exit_min_mult', 0.5)
        self.dyn_exit_decay_per_bar = config.get('dyn_exit_decay_per_bar', 0.001)
        self.vol_regime_enabled = config.get('vol_regime_enabled', False)
        self.vol_regime_low_pct = config.get('vol_regime_low_pct', 0.35)
        self.vol_regime_high_pct = config.get('vol_regime_high_pct', 0.65)
        self.z_enter_low_mult = config.get('z_enter_low_mult', 0.9)
        self.z_enter_high_mult = config.get('z_enter_high_mult', 1.1)
        self.z_exit_low_mult = config.get('z_exit_low_mult', 0.9)
        self.z_exit_high_mult = config.get('z_exit_high_mult', 1.1)
        self.tiered_entry_enabled = config.get('tiered_entry_enabled', False)
        self.tier1_enter = config.get('tier1_enter', 1.2)
        self.tier2_enter = config.get('tier2_enter', 1.8)
        self.tier1_size = config.get('tier1_size', 0.6)
        self.tier2_size = config.get('tier2_size', 1.0)
        self.tier2_requires_confirmation = config.get('tier2_requires_confirmation', True)
        self.resid_ema_enabled = config.get('resid_ema_enabled', False)
        self.resid_ema_span = config.get('resid_ema_span', 30)
        self.nonlinear_pos_enabled = config.get('nonlinear_pos_enabled', False)
        self.nonlinear_pos_alpha = config.get('nonlinear_pos_alpha', 1.5)
        self.nonlinear_pos_min = config.get('nonlinear_pos_min', 0.4)
        self.nonlinear_pos_max = config.get('nonlinear_pos_max', 1.0)
        
        print(f"[STRATEGY_CORE] Initialized with window={self.window}, z_enter={self.z_enter}, z_exit={self.z_exit}")
    
    def rolling_beta_resid(self, px: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Calculate rolling beta and residuals using OLS regression
        
        Args:
            px: DataFrame with price data (must contain BTC_USD and ETH_USD columns)
            
        Returns:
            Tuple of (cleaned_data, residuals, beta_series)
        """
        print("[STRATEGY_CORE] Calculating rolling beta and residuals...")
        
        sub = px.dropna()
        n = len(sub)
        betas = np.full((n, 2), np.nan)  # For BTC vs ETH (2 variables: constant + ETH)
        
        # Use the crypto asset columns
        X = sub["ETH_USD"].values
        Y = sub["BTC_USD"].values
        
        for i in range(self.window, n):
            if self.beta_update_step > 1:
                step_ok = (i - self.window) % self.beta_update_step == 0
                if not step_ok and i > self.window:
                    betas[i] = betas[i - 1]
                    continue
            Xi = np.column_stack([np.ones(self.window), X[i-self.window:i]])
            yi = Y[i-self.window:i]
            try:
                betas[i], *_ = np.linalg.lstsq(Xi, yi, rcond=None)
            except np.linalg.LinAlgError:
                continue
        
        alpha = pd.Series(betas[:, 0], index=sub.index)
        beta = pd.Series(betas[:, 1], index=sub.index)
        
        # Clip extreme beta values
        med = beta.median()
        iqr = beta.quantile(0.75) - beta.quantile(0.25)
        beta.clip(lower=med - self.clip_beta * iqr,
                 upper=med + self.clip_beta * iqr,
                 inplace=True)
        
        # Calculate residuals
        resid = sub["BTC_USD"] - (alpha + beta * sub["ETH_USD"])
        
        print(f"[STRATEGY_CORE] Beta and residual calculation complete")
        return sub, resid, beta
    
    def calculate_dynamic_adjustments(self, resid: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate dynamic volatility-based adjustments
        
        Args:
            resid: Residual series
            
        Returns:
            Tuple of (dynamic_clipping, dynamic_scaling)
        """
        # Dynamic volatility-based adjustments
        sigma_ref = resid.rolling(1440).std().median()  # Reference volatility (daily rolling median)
        vol_day = resid.rolling(1440).std()  # Daily rolling volatility
        
        # Dynamic clipping based on current volatility vs reference
        clip_dyn = self.clip_resid0 * (vol_day / sigma_ref).clip(lower=0.5, upper=4)
        
        # Dynamic scaling - reduce position size when volatility is high
        scale_dyn = self.scale_base * (sigma_ref / vol_day).clip(upper=1.5)
        
        return clip_dyn, scale_dyn
    
    def generate_signals(
        self,
        resid: pd.Series,
        scale_dyn: pd.Series,
        px: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate position signals using Z-score logic
        
        Args:
            resid: Residual series
            scale_dyn: Dynamic scaling factors
            
        Returns:
            Tuple of (position_series, z_score_series)
        """
        # Z-score calculation (optional EMA smoothing on residuals)
        resid_signal = resid
        if self.resid_ema_enabled:
            resid_signal = resid.ewm(span=self.resid_ema_span, adjust=False).mean()
        mu = resid_signal.rolling(self.window).mean()
        sig = resid_signal.rolling(self.window).std()
        z = (resid_signal - mu) / sig
        
        z = z.ffill()
        vol_day = resid.rolling(1440).std()
        vol_ref = resid.rolling(self.dyn_z_lookback).std() if self.dyn_z_enabled else None
        if vol_ref is not None:
            sigma_ref = vol_ref.median()
            if sigma_ref == 0 or np.isnan(sigma_ref):
                sigma_ref = None
        else:
            sigma_ref = None
        vol_regime_low = None
        vol_regime_high = None
        if self.vol_regime_enabled and self.vol_regime_low_pct is not None and self.vol_regime_high_pct is not None:
            vol_regime_low = vol_day.quantile(self.vol_regime_low_pct)
            vol_regime_high = vol_day.quantile(self.vol_regime_high_pct)
        vol_low = vol_day.quantile(self.vol_filter_pct_low) if self.vol_filter_pct_low is not None else None
        vol_high = vol_day.quantile(self.vol_filter_pct_high) if self.vol_filter_pct_high is not None else None
        regime_vol_th = vol_day.quantile(self.regime_vol_threshold_pct) if self.regime_switch_enabled and self.regime_vol_threshold_pct is not None else None

        btc_price = None
        if px is not None and "BTC_USD" in px.columns:
            btc_price = px["BTC_USD"].reindex(resid.index).ffill()
        trend_slope = None
        if btc_price is not None and self.entry_trend_lookback_min:
            trend_slope = btc_price.pct_change(int(self.entry_trend_lookback_min)).fillna(0)
        scale_dyn_filled = scale_dyn.ffill().fillna(0)
        pos_vals = []
        current_pos = 0
        entry_idx = None
        cooldown_until = -1
        long_count = 0
        short_count = 0

        for i, z_val in enumerate(z.values):
            if np.isnan(z_val):
                pos_vals.append(current_pos)
                continue

            trend_strong = False
            trend_dir = 0
            if trend_slope is not None and self.entry_trend_slope_threshold is not None:
                slope_val = trend_slope.iloc[i]
                if slope_val >= self.entry_trend_slope_threshold:
                    trend_strong = True
                    trend_dir = 1
                elif slope_val <= -self.entry_trend_slope_threshold:
                    trend_strong = True
                    trend_dir = -1

            z_scale = 1.0
            if self.dyn_z_enabled and vol_ref is not None and sigma_ref is not None:
                z_scale_val = vol_ref.iloc[i] / sigma_ref if sigma_ref else 1.0
                if np.isnan(z_scale_val) or z_scale_val <= 0:
                    z_scale_val = 1.0
                z_scale = float(np.clip(z_scale_val, self.dyn_z_clip_low, self.dyn_z_clip_high))
            regime_enter_mult = 1.0
            regime_exit_mult = 1.0
            if self.vol_regime_enabled and vol_regime_low is not None and vol_regime_high is not None:
                vol_val = vol_day.iloc[i]
                if vol_val <= vol_regime_low:
                    regime_enter_mult = self.z_enter_low_mult
                    regime_exit_mult = self.z_exit_low_mult
                elif vol_val >= vol_regime_high:
                    regime_enter_mult = self.z_enter_high_mult
                    regime_exit_mult = self.z_exit_high_mult
            z_enter_eff = self.z_enter * z_scale * regime_enter_mult * (self.trend_entry_scale if trend_strong else 1.0)
            z_exit_eff = self.z_exit * z_scale * regime_exit_mult
            if self.dyn_exit_enabled and entry_idx is not None and current_pos != 0:
                hold_bars = i - entry_idx
                decay = 1.0 - (self.dyn_exit_decay_per_bar * hold_bars)
                decay = max(self.dyn_exit_min_mult, decay)
                z_exit_eff = z_exit_eff * decay
            expected_return = None
            if btc_price is not None:
                sig_val = sig.iloc[i]
                if sig_val == sig_val and btc_price.iloc[i]:
                    expected_move = abs(z_val) * sig_val
                    expected_return = expected_move / btc_price.iloc[i]

            if z_val < -z_enter_eff:
                long_count += 1
            else:
                long_count = 0

            if z_val > z_enter_eff:
                short_count += 1
            else:
                short_count = 0

            if i < cooldown_until:
                pos_vals.append(current_pos)
                continue

            exit_signal = abs(z_val) < z_exit_eff
            if current_pos != 0 and entry_idx is not None:
                if (i - entry_idx) < self.min_hold_bars:
                    exit_signal = False

            desired_pos = current_pos
            if current_pos == 0:
                if self.trend_only and not trend_strong:
                    pos_vals.append(current_pos)
                    continue
                # Volatility filter (only block entries)
                if vol_low is not None and vol_day.iloc[i] < vol_low:
                    pos_vals.append(current_pos)
                    continue
                if vol_high is not None and vol_day.iloc[i] > vol_high:
                    pos_vals.append(current_pos)
                    continue

                edge_ok = True
                if expected_return is not None:
                    if self.dyn_edge_enabled:
                        required_edge = (self.dyn_edge_fee_mult * self.fee) + (self.dyn_edge_vol_mult * (sig.iloc[i] / btc_price.iloc[i]))
                        edge_ok = expected_return >= required_edge
                    elif self.min_edge_return is not None:
                        edge_ok = expected_return >= self.min_edge_return

                use_momentum = False
                if self.regime_switch_enabled and regime_vol_th is not None:
                    use_momentum = vol_day.iloc[i] >= regime_vol_th

                if edge_ok:
                    if use_momentum:
                        if short_count >= self.signal_persistence and trend_dir in (0, 1):
                            desired_pos = -1
                        elif long_count >= self.signal_persistence and trend_dir in (0, -1):
                            desired_pos = 1
                    else:
                        if long_count >= self.signal_persistence and trend_dir in (0, 1):
                            desired_pos = 1
                        elif short_count >= self.signal_persistence and trend_dir in (0, -1):
                            desired_pos = -1
                    if desired_pos != 0:
                        if self.tiered_entry_enabled:
                            tier_size = self._tiered_size_from_z(abs(z_val))
                            if self.tier2_requires_confirmation and abs(z_val) >= self.tier2_enter:
                                if (long_count < self.signal_persistence and desired_pos > 0) or (
                                    short_count < self.signal_persistence and desired_pos < 0
                                ):
                                    tier_size = self.tier1_size
                            scale_val = scale_dyn_filled.iloc[i]
                            desired_size = tier_size * scale_val
                            desired_pos = np.sign(desired_pos) * desired_size
                        elif self.scale_in_enabled:
                            scale_val = scale_dyn_filled.iloc[i]
                            desired_size = self._position_size_from_z(abs(z_val), scale_val)
                            desired_pos = np.sign(desired_pos) * desired_size
                        elif self.nonlinear_pos_enabled:
                            scale_val = scale_dyn_filled.iloc[i]
                            size_mult = self._nonlinear_mult_from_z(abs(z_val))
                            desired_size = size_mult * scale_val
                            desired_pos = np.sign(desired_pos) * desired_size
                        current_pos = desired_pos
                        entry_idx = i
            else:
                if self.fee_exit_enabled and expected_return is not None:
                    if expected_return > (self.fee_exit_mult * self.fee):
                        exit_signal = False
                if exit_signal:
                    current_pos = 0
                    entry_idx = None
                    cooldown_until = i + self.cooldown_bars
                elif self.scale_in_enabled:
                    scale_val = scale_dyn_filled.iloc[i]
                    desired_size = self._position_size_from_z(abs(z_val), scale_val)
                    desired_pos = np.sign(current_pos) * desired_size
                    if abs(desired_pos - current_pos) >= self.scale_update_min_delta:
                        current_pos = desired_pos
                elif self.tiered_entry_enabled:
                    tier_size = self._tiered_size_from_z(abs(z_val))
                    scale_val = scale_dyn_filled.iloc[i]
                    desired_size = tier_size * scale_val
                    desired_pos = np.sign(current_pos) * desired_size
                    if abs(desired_pos - current_pos) >= self.scale_update_min_delta:
                        current_pos = desired_pos
                elif self.nonlinear_pos_enabled:
                    scale_val = scale_dyn_filled.iloc[i]
                    size_mult = self._nonlinear_mult_from_z(abs(z_val))
                    desired_size = size_mult * scale_val
                    desired_pos = np.sign(current_pos) * desired_size
                    if abs(desired_pos - current_pos) >= self.scale_update_min_delta:
                        current_pos = desired_pos

            pos_vals.append(current_pos)

        pos = pd.Series(pos_vals, index=resid.index)
        if not self.scale_in_enabled and not self.tiered_entry_enabled and not self.nonlinear_pos_enabled:
            pos = pos * scale_dyn_filled
        pos = pos.clip(-self.inv_cap, self.inv_cap)

        return pos, z

    def _tiered_size_from_z(self, z_abs: float) -> float:
        if z_abs >= self.tier2_enter:
            return self.tier2_size
        return self.tier1_size

    def _position_size_from_z(self, z_abs: float, scale_val: float) -> float:
        base_size = min(self.inv_cap, z_abs * self.scale_in_slope)
        base_size = max(self.scale_in_min, base_size)
        if self.scale_base and scale_val == scale_val:
            base_size *= scale_val / self.scale_base
        return float(min(self.inv_cap, base_size))

    def _nonlinear_mult_from_z(self, z_abs: float) -> float:
        if z_abs <= self.z_enter:
            return self.nonlinear_pos_min
        z_ratio = z_abs / self.z_enter
        size = self.nonlinear_pos_min + (z_ratio ** self.nonlinear_pos_alpha - 1.0) * (
            self.nonlinear_pos_max - self.nonlinear_pos_min
        )
        return float(np.clip(size, self.nonlinear_pos_min, self.nonlinear_pos_max))
    
    def calculate_pnl_series(self, pos: pd.Series, resid: pd.Series, px: pd.DataFrame, clip_dyn: pd.Series) -> pd.Series:
        """
        Calculate PnL series with dynamic residual clipping
        
        Args:
            pos: Position series
            resid: Residual series
            px: Price data
            clip_dyn: Dynamic clipping factors
            
        Returns:
            Total PnL series
        """
        # Dynamic residual clipping and PnL calculation
        resid_diff = resid.diff().fillna(0).clip(lower=-clip_dyn, upper=clip_dyn)
        pnl_series = pos.shift() * resid_diff / px["BTC_USD"].shift()
        fee_series = -self.fee * pos.diff().abs().fillna(0)
        total_pnl_series = (pnl_series + fee_series).dropna()
        
        return total_pnl_series
    
    def calculate_pnl(self, position_size: float, entry_price: float, exit_price: float) -> float:
        """
        Calculate PnL for a single trade
        
        Args:
            position_size: Size of position (1 for long, -1 for short)
            entry_price: Entry price
            exit_price: Exit price
            
        Returns:
            PnL as percentage
        """
        if position_size > 0:  # Long position
            return (exit_price - entry_price) / entry_price
        else:  # Short position
            return (entry_price - exit_price) / entry_price
    
    def run_strategy(self, px: pd.DataFrame, user_start_date: str, user_end_date: str) -> Dict:
        """
        Run the complete strategy - SINGLE SOURCE OF TRUTH
        
        This method contains the complete strategy logic and must be used by both
        live trading and backtesting to ensure consistency.
        
        Args:
            px: Price data DataFrame
            user_start_date: Start date for analysis
            user_end_date: End date for analysis
            
        Returns:
            Dictionary with complete strategy results
        """
        print("[STRATEGY_CORE] Running statistical arbitrage strategy...")
        
        # Convert user date range to datetime for filtering
        user_start_dt = pd.to_datetime(user_start_date)
        user_end_dt = pd.to_datetime(user_end_date)
        
        # Handle timezone consistently
        if user_start_dt.tz is None:
            user_start_dt = user_start_dt.tz_localize('UTC')
        if user_end_dt.tz is None:
            user_end_dt = user_end_dt.tz_localize('UTC')
        
        # Step 1: Calculate rolling beta and residuals
        sub, resid, beta = self.rolling_beta_resid(px)
        
        # Step 2: Calculate dynamic adjustments
        clip_dyn, scale_dyn = self.calculate_dynamic_adjustments(resid)
        
        # Step 3: Generate signals
        pos, z = self.generate_signals(resid, scale_dyn, sub)
        
        # Step 4: Calculate PnL series
        total_pnl_series = self.calculate_pnl_series(pos, resid, sub, clip_dyn)
        
        # Step 5: Track trades and generate detailed logs
        trades = []
        trade_details = []
        position_size = 0
        entry_price = 0
        entry_time = None
        current_day = None
        daily_pnl = 0.0
        daily_returns = []
        equity = [1.0]
        equity_dates = []
        
        # Generate detailed trade logs
        for i in range(1, len(pos)):
            timestamp = pos.index[i]
            # Ensure timestamp is tz-aware (UTC)
            if getattr(timestamp, 'tz', None) is None:
                timestamp = pd.Timestamp(timestamp).tz_localize('UTC')
            current_pos = pos.iloc[i]
            prev_pos = pos.iloc[i-1]
            price = sub["BTC_USD"].iloc[i]
            z_score = z.iloc[i]
            cur_day = timestamp.date()
            
            if np.isnan(z_score):
                continue
                
            # Check for new trading day
            if current_day is None:
                current_day = cur_day
            elif cur_day != current_day:
                # End of day settlement
                if position_size != 0:
                    pnl = self.calculate_pnl(position_size, entry_price, price)
                    exit_fee = self.fee * abs(position_size)
                    net_pnl = pnl - exit_fee
                    daily_pnl += net_pnl
                    trades.append((timestamp, 'close_eod', price))
                    trade_details.append({
                        'timestamp': timestamp, 'action': 'close_eod', 'price': price, 
                        'pnl': net_pnl, 'fee': exit_fee, 'gross_pnl': pnl,
                        'zscore': z_score, 'position': position_size,
                        'entry_price': entry_price, 'entry_time': entry_time
                    })
                    # Only print trades within user-specified range
                    if user_start_dt <= timestamp <= user_end_dt:
                        print(f"{timestamp} CLOSE (EOD) @ {price:.2f}, PnL: {net_pnl:.4%}, Position: {position_size}")
                    position_size = 0
                    
                # Daily settlement
                equity.append(equity[-1] * (1 + daily_pnl))
                equity_dates.append(current_day)
                daily_returns.append(daily_pnl)
                daily_pnl = 0.0
                current_day = cur_day
            
            # Check for stop loss before position changes
            if position_size != 0:
                current_pnl = self.calculate_pnl(position_size, entry_price, price)
                if current_pnl <= -self.stop_loss_pct:
                    # Stop loss triggered - force close position
                    pnl = current_pnl
                    exit_fee = self.fee * abs(position_size)
                    net_pnl = pnl - exit_fee
                    daily_pnl += net_pnl
                    trades.append((timestamp, 'stop_loss', price))
                    trade_details.append({
                        'timestamp': timestamp, 'action': 'stop_loss', 'price': price, 
                        'pnl': net_pnl, 'fee': exit_fee, 'gross_pnl': pnl,
                        'zscore': z_score, 'position': position_size,
                        'entry_price': entry_price, 'entry_time': entry_time
                    })
                    # Only print trades within user-specified range
                    if user_start_dt <= timestamp <= user_end_dt:
                        print(f"{timestamp} STOP LOSS @ {price:.2f}, PnL: {net_pnl:.4%}, Position: {position_size}")
                    position_size = 0
                    # Skip normal position logic since we've closed
                    continue
            
            # Position changes
            if abs(current_pos - prev_pos) > 1e-6:
                if position_size == 0 and abs(current_pos) > 1e-6:
                    # Opening position
                    position_size = 1 if current_pos > 0 else -1
                    entry_price = price
                    entry_time = timestamp
                    entry_fee = self.fee * abs(position_size)
                    daily_pnl -= entry_fee
                    action = 'buy' if position_size > 0 else 'sell'
                    trades.append((timestamp, action, price))
                    trade_details.append({
                        'timestamp': timestamp, 'action': action, 'price': price, 
                        'pnl': -entry_fee, 'fee': entry_fee, 'gross_pnl': 0.0,
                        'zscore': z_score, 'position': position_size,
                        'entry_price': entry_price, 'entry_time': entry_time
                    })
                    # Only print trades within user-specified range
                    if user_start_dt <= timestamp <= user_end_dt:
                        print(f"{timestamp} {action.upper()} @ {price:.2f}, Z-Score: {z_score:.2f}, Position: {position_size}")
                    
                elif position_size != 0 and abs(current_pos) < 1e-6:
                    # Closing position
                    pnl = self.calculate_pnl(position_size, entry_price, price)
                    exit_fee = self.fee * abs(position_size)
                    net_pnl = pnl - exit_fee
                    daily_pnl += net_pnl
                    trades.append((timestamp, 'close', price))
                    trade_details.append({
                        'timestamp': timestamp, 'action': 'close', 'price': price, 
                        'pnl': net_pnl, 'fee': exit_fee, 'gross_pnl': pnl,
                        'zscore': z_score, 'position': position_size,
                        'entry_price': entry_price, 'entry_time': entry_time
                    })
                    # Only print trades within user-specified range
                    if user_start_dt <= timestamp <= user_end_dt:
                        print(f"{timestamp} CLOSE @ {price:.2f}, PnL: {net_pnl:.4%}, Position: {position_size}")
                    position_size = 0
        
        # Final settlement
        if position_size != 0:
            final_price = sub["BTC_USD"].iloc[-1]
            pnl = self.calculate_pnl(position_size, entry_price, final_price)
            exit_fee = self.fee * abs(position_size)
            net_pnl = pnl - exit_fee
            daily_pnl += net_pnl
            final_time = pos.index[-1]
            trades.append((final_time, 'close_final', final_price))
            trade_details.append({
                'timestamp': final_time, 'action': 'close_final', 'price': final_price, 
                'pnl': net_pnl, 'fee': exit_fee, 'gross_pnl': pnl,
                'zscore': z.iloc[-1], 'position': position_size,
                'entry_price': entry_price, 'entry_time': entry_time
            })
            # Only print trades within user-specified range
            if user_start_dt <= final_time <= user_end_dt:
                print(f"{final_time} CLOSE (FINAL) @ {final_price:.2f}, PnL: {net_pnl:.4%}, Position: {position_size}")
        
        if current_day is not None:
            equity.append(equity[-1] * (1 + daily_pnl))
            equity_dates.append(current_day)
            daily_returns.append(daily_pnl)
        
        print("[STRATEGY_CORE] Strategy calculation complete")
        
        # Create equity curve
        equity_curve = pd.Series(equity, index=pd.to_datetime([pos.index[0].date()] + equity_dates))
        
        # Filter results to user-requested time range
        total_pnl_filtered = total_pnl_series.loc[user_start_dt:user_end_dt]
        
        # Filter trades and trade_details to user-requested time range
        trades_filtered = [(t, action, price) for t, action, price in trades 
                          if user_start_dt <= t <= user_end_dt]
        
        trade_details_filtered = [td for td in trade_details 
                                 if user_start_dt <= td['timestamp'] <= user_end_dt]
        
        print(f"[STRATEGY_CORE] Filtered to user period: {len(trades_filtered)} trades, PnL series length: {len(total_pnl_filtered)}")
        
        return {
            'pnl_series': total_pnl_filtered,
            'trades': trades_filtered,
            'trade_details': trade_details_filtered,
            'equity_curve': equity_curve,
            'daily_returns': daily_returns,
            'residuals': resid,
            'positions': pos,
            'z_scores': z,
            'beta': beta,
            'raw_data': sub,
            'full_pnl_series': total_pnl_series,
            'full_trades': trades,
            'full_trade_details': trade_details
        }
    
    def should_trigger_stop_loss_price_based(self, entry_price: float, current_price: float, is_long: bool) -> bool:
        """
        Check if price-based stop loss should trigger (shared logic for live/backtest)
        
        Args:
            entry_price: Position entry price
            current_price: Current market price
            is_long: True for long position, False for short
            
        Returns:
            True if stop loss should trigger
        """
        if is_long:
            # For long: loss when price goes down
            price_pct_change = (current_price - entry_price) / entry_price
        else:
            # For short: loss when price goes up
            price_pct_change = (entry_price - current_price) / entry_price
            
        # Stop loss triggers when price moves against position by threshold percentage
        return price_pct_change <= -self.stop_loss_pct
    
    def should_exit_sage_mode(self, zscore: float, sage_mode_signal: int) -> bool:
        """
        Check if sage mode should exit based on corrected logic (shared for live/backtest)
        
        Args:
            zscore: Current Z-score
            sage_mode_signal: Original position direction (1 for long, -1 for short)
            
        Returns:
            True if sage mode should exit
        """
        if sage_mode_signal > 0:
            # After LONG stop loss, wait for LONG close signal
            # LONG closes when Z-score > -exit_threshold (returns to neutral from negative)
            return zscore > -self.z_exit
        elif sage_mode_signal < 0:
            # After SHORT stop loss, wait for SHORT close signal
            # SHORT closes when Z-score < exit_threshold (returns to neutral from positive)
            return zscore < self.z_exit
            
        return False
    
    def get_position_size_from_zscore(self, zscore: float) -> float:
        """
        Calculate position size based on Z-score (shared logic for live/backtest)
        
        Args:
            zscore: Current Z-score
            
        Returns:
            Position size (positive for long, negative for short, 0 for flat)
        """
        if zscore > self.z_enter:
            base_size = min(self.inv_cap, abs(zscore) * self.scale_in_slope)
            base_size = max(self.scale_in_min, base_size)
            return -base_size
        elif zscore < -self.z_enter:
            base_size = min(self.inv_cap, abs(zscore) * self.scale_in_slope)
            base_size = max(self.scale_in_min, base_size)
            return base_size
        else:
            # No signal or exit signal
            return 0.0
