# @ command: cd /Users/zhihaoouyang/Desktop/code/when2buy/github/fina && python -m fina.strategies.crypto.btc_eth_v1.live_monitor
#!/usr/bin/env python3
"""
实时监控脚本 - 每分钟更新数据并执行策略
直接调用run_strategy_bar_by_bar.py的逻辑，确保一致性
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import pandas as pd
import pytz

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fina.consts import LOG_PATH
from fina.strategies.crypto.btc_eth_v1.data_manager import CryptoDataManager
from fina.strategies.crypto.btc_eth_v1.strategy_engine import StatArbEngine

# 确保日志目录存在
monitor_dir = LOG_PATH / "monitor"
monitor_dir.mkdir(parents=True, exist_ok=True)
print("monitor_dir:", monitor_dir)

# 设置日志 - 必须在使用logger之前
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(monitor_dir / "live_monitor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ETF执行模块 - 现在可以安全使用logger了
try:
    from fina.strategies.crypto.btc_eth_v1.alpaca_trade.etf_executor import ETFExecutor
    ETF_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ETF模块不可用: {e}")
    ETF_AVAILABLE = False


class LiveMonitor:
    def __init__(self):
        # 初始化数据管理器
        config = {"symbols": ["BTC/USD", "ETH/USD", "SOL/USD"]}
        self.data_manager = CryptoDataManager(config)

        # 策略参数 (与run_strategy_bar_by_bar.py保持一致)
        self.strategy_config = {
            "window": 360,
            "z_enter": 1.2,
            "z_exit": 0.4,
            "signal_persistence": 3,
            "min_hold_bars": 30,
            "cooldown_bars": 30,
            "vol_filter_pct_low": None,
            "vol_filter_pct_high": None,
            "entry_trend_lookback_min": None,
            "entry_trend_slope_threshold": None,
            "trend_entry_scale": 1.0,
            "trend_only": False,
            "min_edge_return": 0.0014,
            "dyn_edge_enabled": True,
            "dyn_edge_fee_mult": 5.0,
            "dyn_edge_vol_mult": 0.5,
            "fee_exit_enabled": True,
            "fee_exit_mult": 2.0,
            "beta_update_step": 1,
            "dyn_exit_enabled": False,
            "dyn_exit_min_mult": 0.5,
            "dyn_exit_decay_per_bar": 0.001,
            "vol_regime_enabled": False,
            "vol_regime_low_pct": 0.35,
            "vol_regime_high_pct": 0.65,
            "z_enter_low_mult": 0.9,
            "z_enter_high_mult": 1.1,
            "z_exit_low_mult": 0.9,
            "z_exit_high_mult": 1.1,
            "tiered_entry_enabled": False,
            "tier1_enter": 1.2,
            "tier2_enter": 1.8,
            "tier1_size": 0.6,
            "tier2_size": 1.0,
            "tier2_requires_confirmation": True,
            "resid_ema_enabled": False,
            "resid_ema_span": 30,
            "nonlinear_pos_enabled": False,
            "nonlinear_pos_alpha": 1.5,
            "nonlinear_pos_min": 0.4,
            "nonlinear_pos_max": 1.0,
            "scale_base": 0.05,
            "inv_cap": 0.15,
            "fee": 5e-4,
            "stop_loss_pct": 0.01,
            "clip_resid0": 800,
            "clip_beta": 6,
        }
        self.engine = StatArbEngine(self.strategy_config)

        self.monitor_dir = LOG_PATH / "monitor"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)

        # 监控状态
        self.current_position = 0
        self.current_equity = 1.0
        self.trade_history = []
        self.last_update = None
        self.entry_price = 0
        self.entry_time = None
        self.last_pnl = 0
        self.last_signal = 0
        self.long_count = 0
        self.short_count = 0
        self.last_exit_time = None

        # 贤者模式状态 - 止损后的保护机制
        self.sage_mode = False  # 是否处于贤者模式
        self.sage_mode_signal = 0  # 触发止损的原始信号方向
        self.sage_mode_entry_time = None  # 进入贤者模式的时间

        # 订单执行状态跟踪
        self.order_execution_state = "NONE"  # NONE, PENDING_OPEN, PENDING_CLOSE
        self.pending_orders = {}  # 跟踪待执行的限价单
        self.order_timeout_minutes = 30  # 订单超时时间（分钟）

        # ETF执行器初始化
        self.etf_executor = None
        self.etf_enabled = False
        if ETF_AVAILABLE:
            try:
                etf_log_dir = str(self.monitor_dir / "etf_logs")
                self.etf_executor = ETFExecutor(log_dir=etf_log_dir)
                self.etf_enabled = True
                logger.info("✅ ETF执行器初始化成功")
            except Exception as e:
                logger.error(f"❌ ETF执行器初始化失败: {e}")
                self.etf_enabled = False

        # Market close handler for automatic position flattening
        self.market_close_handler = None
        if self.etf_enabled:
            try:
                from fina.strategies.crypto.btc_eth_v1.market_close_handler import MarketCloseHandler
                self.market_close_handler = MarketCloseHandler(self.etf_executor)
                logger.info("✅ Market close handler initialized")
            except Exception as e:
                logger.warning(f"Market close handler not available: {e}")

        # 加载历史状态
        self.load_state()

    def _safe_float(self, value):
        """安全转换为JSON可序列化的float"""
        if pd.isna(value) or np.isinf(value):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _calc_fee_cost(self, new_position: float) -> float:
        """Calculate fee cost based on position delta."""
        delta = new_position - self.current_position
        return self.engine.fee * abs(delta)
    
    def _check_pending_orders(self):
        """检查并管理待执行的限价单"""
        if not self.pending_orders:
            return
            
        current_time = datetime.now()
        orders_to_remove = []
        
        for order_id, order_info in self.pending_orders.items():
            order_time = datetime.fromisoformat(order_info['submitted_at'])
            elapsed_minutes = (current_time - order_time).total_seconds() / 60
            
            if elapsed_minutes >= self.order_timeout_minutes:
                # 订单超时
                order_type = order_info['order_type']
                if order_type == 'open':
                    # 开仓订单超时：放弃交易
                    logger.warning(f"⏰ Open order timeout ({elapsed_minutes:.1f}min): Giving up on {order_info['symbol']} {order_info['side']}")
                    self.order_execution_state = "NONE"
                    orders_to_remove.append(order_id)
                    
                elif order_type == 'close':
                    # 平仓订单超时：改用市价单强制平仓
                    logger.warning(f"⏰ Close order timeout ({elapsed_minutes:.1f}min): Switching to market order for {order_info['symbol']}")
                    if self.etf_executor:
                        try:
                            # 使用市价单强制平仓
                            success = self.etf_executor.execute_market_order_timeout(order_info)
                            if success:
                                logger.info(f"✅ Market order executed successfully after timeout")
                                self.order_execution_state = "NONE"
                                orders_to_remove.append(order_id)
                            else:
                                logger.error(f"❌ Market order failed after timeout")
                        except Exception as e:
                            logger.error(f"❌ Market order timeout execution failed: {e}")
            else:
                # 检查订单是否已成交
                if self.etf_executor:
                    try:
                        is_filled = self.etf_executor.check_order_status(order_id)
                        if is_filled:
                            logger.info(f"✅ Limit order filled: {order_info['symbol']} {order_info['side']} after {elapsed_minutes:.1f}min")
                            self.order_execution_state = "NONE"
                            orders_to_remove.append(order_id)
                    except Exception as e:
                        logger.error(f"Error checking order status: {e}")
        
        # 移除已完成或超时的订单
        for order_id in orders_to_remove:
            del self.pending_orders[order_id]

    def _should_cancel_pending_orders(self, new_position: float) -> bool:
        """Check if pending orders should be cancelled due to conflicting new signal
        
        Args:
            new_position: The new position from the latest signal
            
        Returns:
            True if orders should be cancelled
        """
        if not self.pending_orders:
            return False
        
        # Determine position directions
        current_pos_sign = 1 if self.current_position > 0 else (-1 if self.current_position < 0 else 0)
        new_pos_sign = 1 if new_position > 0 else (-1 if new_position < 0 else 0)
        
        # Cancel orders if the new signal conflicts with what we're currently trying to do
        
        # Case 1: Signal direction reversal (long->short or short->long)
        if current_pos_sign != 0 and new_pos_sign != 0 and current_pos_sign != new_pos_sign:
            logger.info(f"🔄 Direction reversal detected: {current_pos_sign} -> {new_pos_sign}")
            return True
            
        # Case 2: We have position and new signal wants to close (pos->flat)
        if current_pos_sign != 0 and new_pos_sign == 0:
            if self.order_execution_state == "PENDING_OPEN":
                # We're trying to add to position but signal wants to close
                logger.info(f"🔄 Position->Flat signal while pending open orders")
                return True
                
        # Case 3: We're flat and trying to open, but new signal wants different direction
        if current_pos_sign == 0 and self.order_execution_state == "PENDING_OPEN":
            # Check if any pending order direction conflicts with new signal
            for order_info in self.pending_orders.values():
                order_direction = 1 if order_info['side'] == 'buy' else -1
                if new_pos_sign != 0 and order_direction != new_pos_sign:
                    logger.info(f"🔄 Pending order direction {order_direction} conflicts with new signal {new_pos_sign}")
                    return True
                elif new_pos_sign == 0:
                    logger.info(f"🔄 New signal wants flat while trying to open position")
                    return True
                    
        # Case 4: We're trying to close but new signal wants to enter in different direction
        if current_pos_sign != 0 and self.order_execution_state == "PENDING_CLOSE":
            if new_pos_sign != 0 and new_pos_sign != current_pos_sign:
                logger.info(f"🔄 Trying to close {current_pos_sign} but new signal wants {new_pos_sign}")
                return True
                
        return False

    def _cancel_conflicting_orders(self):
        """Cancel all pending orders that conflict with new signal"""
        if not self.pending_orders or not self.etf_executor:
            return
            
        order_ids = list(self.pending_orders.keys())
        logger.info(f"🚫 Cancelling {len(order_ids)} conflicting orders: {order_ids}")
        
        # Cancel orders through ETF executor
        cancel_results = self.etf_executor.cancel_pending_orders(order_ids)
        
        # Clear pending orders and reset state
        cancelled_count = sum(1 for success in cancel_results.values() if success)
        logger.info(f"✅ Successfully cancelled {cancelled_count}/{len(order_ids)} orders")
        
        # Reset order execution state
        self.pending_orders.clear()
        self.order_execution_state = "NONE"
        logger.info("🔄 Order execution state reset to NONE - ready for new signals")

    def _execute_etf_trade(self, btc_position, signal_timestamp, signal_price, z_score, etf_prices=None):
        """执行ETF交易（基于BTC仓位变化）"""
        logger.info(f"🔍 ETF Trade Debug: enabled={self.etf_enabled}, executor={self.etf_executor is not None}, position={btc_position:.3f}")
        if etf_prices:
            logger.info(f"📊 ETF Prices from signal minute: {etf_prices}")
        
        if not self.etf_enabled or not self.etf_executor:
            logger.warning(f"⚠️  ETF trading disabled: enabled={self.etf_enabled}, executor={self.etf_executor is not None}")
            return
        
        # Check if we're in the market close window (15:57-16:00 ET)
        import pytz
        from datetime import time
        et_tz = pytz.timezone('US/Eastern')
        current_et = datetime.now(et_tz)
        current_time = current_et.time()
        
        # Block new position opening during close window (allow closing only)
        if time(15, 57) <= current_time < time(16, 0):
            if self.current_position == 0 and btc_position != 0:
                logger.warning(f"🚫 Blocking new position opening during market close window (15:57-16:00 ET)")
                return
            elif self.current_position != 0 and btc_position != 0 and btc_position * self.current_position < 0:
                logger.warning(f"🚫 Blocking position reversal during market close window (15:57-16:00 ET)")
                return
            elif self.current_position != 0 and btc_position == 0:
                logger.info(f"✅ Allowing position close during market close window")
                # Continue with close logic
        
        # 检查是否有待执行的订单
        if self.order_execution_state != "NONE":
            logger.info(f"🚫 Signal ignored: Currently in {self.order_execution_state} state, waiting for order completion")
            return
        
        try:
            logger.info(f"🎯 Calling ETF executor: position={btc_position:.3f}, price={signal_price:.2f}, z_score={z_score:.3f}")
            
            # 确定订单类型
            is_opening = (self.current_position == 0 and btc_position != 0)
            is_closing = (self.current_position != 0 and btc_position == 0)
            is_reversing = (self.current_position != 0 and btc_position != 0 and btc_position * self.current_position < 0)
            
            order_type = "open" if is_opening else ("close" if is_closing else ("reverse" if is_reversing else "none"))
            
            result = self.etf_executor.execute_btc_position_change_limit(
                btc_position=btc_position,
                signal_timestamp=str(signal_timestamp),
                signal_price=signal_price,
                z_score=z_score,
                order_type=order_type,
                etf_prices=etf_prices
            )
            
            if result['success']:
                # 将待执行订单添加到跟踪系统
                self.pending_orders.update(result['pending_orders'])
                
                # 设置相应的执行状态
                if is_opening:
                    self.order_execution_state = "PENDING_OPEN"
                elif is_closing or is_reversing:
                    self.order_execution_state = "PENDING_CLOSE"
                    
                orders_count = result.get('orders_submitted', 0)
                logger.info(f"💰 ETF限价单提交成功: {orders_count} orders for BTC仓位 -> {btc_position:.3f}, 状态: {self.order_execution_state}")
                
                # 记录限价单详情
                for order_id, order_info in result['pending_orders'].items():
                    logger.info(f"📝 Limit order tracking: {order_info['side']} {order_info['qty']} {order_info['symbol']} @ ${order_info['limit_price']:.2f} (ID: {order_id})")
                    
            else:
                reason = result.get('reason', 'unknown')
                logger.info(f"⏸️  ETF limit orders skipped: {reason}")
            
        except Exception as e:
            logger.error(f"❌ ETF交易执行失败: {e}")

    def load_state(self):
        """加载历史状态"""
        state_file = self.monitor_dir / "monitor_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    self.current_position = state.get("position", 0)
                    self.current_equity = state.get("equity", 1.0)
                    self.trade_history = state.get("trade_history", [])
                    self.last_update = state.get("last_update")
                    self.entry_price = state.get("entry_price", 0)
                    self.entry_time = state.get("entry_time")

                    # 加载贤者模式状态
                    self.sage_mode = state.get("sage_mode", False)
                    self.sage_mode_signal = state.get("sage_mode_signal", 0)
                    self.sage_mode_entry_time = state.get("sage_mode_entry_time")

                    # 加载订单执行状态
                    self.order_execution_state = state.get("order_execution_state", "NONE")
                    self.pending_orders = state.get("pending_orders", {})
                    logger.info(
                        f"Loaded historical state: position={self.current_position}, equity={self.current_equity}, sage_mode={'ON' if self.sage_mode else 'OFF'}"
                    )
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def save_state(self):
        """保存当前状态"""
        state = {
            "position": self.current_position,
            "equity": self.current_equity,
            "trade_history": self.trade_history,
            "last_update": datetime.now().isoformat(),
            "entry_price": self.entry_price,
            "entry_time": str(self.entry_time) if self.entry_time is not None else None,
            # 保存贤者模式状态
            "sage_mode": self.sage_mode,
            "sage_mode_signal": self.sage_mode_signal,
            "sage_mode_entry_time": (
                str(self.sage_mode_entry_time)
                if self.sage_mode_entry_time is not None
                else None
            ),
            # 保存订单执行状态
            "order_execution_state": self.order_execution_state,
            "pending_orders": self.pending_orders,
        }

        state_file = self.monitor_dir / "monitor_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def update_data(self):
        """更新最新数据"""
        try:
            logger.info("Updating latest data...")
            self.data_manager.ensure_latest_data()

            # 计算窗口
            window = self.strategy_config["window"]
            now = datetime.now(pytz.timezone("US/Eastern"))
            start_date = (now - timedelta(minutes=window + 10)).strftime("%Y-%m-%d")
            end_date = now.strftime("%Y-%m-%d")

            # Get both crypto and ETF data
            px, etf_px, _ = self.data_manager.load_and_align_data_with_etf(start_date, end_date)
            if px.empty:
                logger.error("No data available!")
                return None, None

            logger.info(f"Data update completed, latest time: {px.index[-1]}")
            if not etf_px.empty:
                logger.info(f"ETF data available: {etf_px.columns.tolist()}")
            return px, etf_px
        except Exception as e:
            logger.error(f"Data update failed: {e}")
            return None, None

    def execute_strategy(self, px, etf_px=None):
        """执行策略 (与run_strategy_bar_by_bar.py逻辑完全一致)"""
        try:

            # 预先计算rolling beta和resid
            sub, resid, beta = self.engine.rolling_beta_resid(px)
            clip_dyn, scale_dyn = self.engine.calculate_dynamic_adjustments(resid)
            mu = resid.rolling(self.engine.window).mean()
            sig = resid.rolling(self.engine.window).std()
            z = (resid - mu) / sig
            vol_day = resid.rolling(1440).std()
            vol_low = None
            vol_high = None
            if self.strategy_config.get("vol_filter_pct_low") is not None:
                vol_low = vol_day.quantile(self.strategy_config["vol_filter_pct_low"])
            if self.strategy_config.get("vol_filter_pct_high") is not None:
                vol_high = vol_day.quantile(self.strategy_config["vol_filter_pct_high"])
            vol_regime_low = None
            vol_regime_high = None
            if self.strategy_config.get("vol_regime_enabled"):
                low_pct = self.strategy_config.get("vol_regime_low_pct", 0.35)
                high_pct = self.strategy_config.get("vol_regime_high_pct", 0.65)
                vol_regime_low = vol_day.quantile(low_pct)
                vol_regime_high = vol_day.quantile(high_pct)
            btc_price = sub["BTC_USD"].reindex(resid.index).ffill()
            trend_slope = None
            lookback = self.strategy_config.get("entry_trend_lookback_min")
            if lookback:
                trend_slope = btc_price.pct_change(int(lookback)).fillna(0)

            # 获取最新数据点
            i = len(sub) - 1
            ts = sub.index[i]
            price = sub["BTC_USD"].iloc[i]
            z_score = z.iloc[i]
            scale = scale_dyn.iloc[i]
            expected_return = None
            if sig.iloc[i] == sig.iloc[i] and price:
                expected_move = abs(z_score) * sig.iloc[i]
                expected_return = expected_move / price
            
            # Get ETF prices from the same minute (if available)
            etf_prices = {}
            if etf_px is not None and not etf_px.empty and ts in etf_px.index:
                if "BITO_close" in etf_px.columns:
                    etf_prices["BITO"] = etf_px.loc[ts, "BITO_close"]
                if "BITI_close" in etf_px.columns:
                    etf_prices["BITI"] = etf_px.loc[ts, "BITI_close"]

            # 信号生成（加入持仓持久性、最小持仓和冷却时间以降低换手）
            trend_strong = False
            trend_dir = 0
            slope_th = self.strategy_config.get("entry_trend_slope_threshold")
            if trend_slope is not None and slope_th is not None:
                slope_val = trend_slope.iloc[i]
                if slope_val >= slope_th:
                    trend_strong = True
                    trend_dir = 1
                elif slope_val <= -slope_th:
                    trend_strong = True
                    trend_dir = -1

            z_enter_eff = self.engine.z_enter * (
                self.strategy_config.get("trend_entry_scale", 1.0) if trend_strong else 1.0
            )
            if self.strategy_config.get("vol_regime_enabled") and vol_regime_low is not None and vol_regime_high is not None:
                vol_val = vol_day.iloc[i]
                if vol_val <= vol_regime_low:
                    z_enter_eff *= self.strategy_config.get("z_enter_low_mult", 0.9)
                elif vol_val >= vol_regime_high:
                    z_enter_eff *= self.strategy_config.get("z_enter_high_mult", 1.1)

            if z_score < -z_enter_eff:
                self.long_count += 1
            else:
                self.long_count = 0

            if z_score > z_enter_eff:
                self.short_count += 1
            else:
                self.short_count = 0

            signal = 0
            z_exit_eff = self.engine.z_exit
            if self.strategy_config.get("dyn_exit_enabled") and self.current_position != 0 and self.entry_time is not None:
                held_minutes = (ts - self.entry_time).total_seconds() / 60.0
                decay = 1.0 - (self.strategy_config.get("dyn_exit_decay_per_bar", 0.001) * held_minutes)
                decay = max(self.strategy_config.get("dyn_exit_min_mult", 0.5), decay)
                z_exit_eff = z_exit_eff * decay
            if self.strategy_config.get("vol_regime_enabled") and vol_regime_low is not None and vol_regime_high is not None:
                vol_val = vol_day.iloc[i]
                if vol_val <= vol_regime_low:
                    z_exit_eff *= self.strategy_config.get("z_exit_low_mult", 0.9)
                elif vol_val >= vol_regime_high:
                    z_exit_eff *= self.strategy_config.get("z_exit_high_mult", 1.1)

            if self.current_position == 0:
                if self.long_count >= self.engine.signal_persistence:
                    signal = 1
                elif self.short_count >= self.engine.signal_persistence:
                    signal = -1
            else:
                if abs(z_score) < z_exit_eff:
                    signal = 0
                else:
                    signal = 1 if self.current_position > 0 else -1

            if self.strategy_config.get("fee_exit_enabled") and expected_return is not None and self.current_position != 0:
                if expected_return > (self.strategy_config.get("fee_exit_mult", 2.0) * self.engine.fee):
                    signal = 1 if self.current_position > 0 else -1

            if self.last_exit_time is not None and self.current_position == 0:
                cooldown_end = self.last_exit_time + pd.Timedelta(minutes=self.engine.cooldown_bars)
                if ts < cooldown_end:
                    signal = 0

            if self.current_position != 0 and self.entry_time is not None:
                held_minutes = (ts - self.entry_time).total_seconds() / 60.0
                if held_minutes < self.engine.min_hold_bars and signal == 0:
                    signal = 1 if self.current_position > 0 else -1

            # Optional entry filters (only block new entries)
            if self.current_position == 0 and signal != 0:
                if self.strategy_config.get("trend_only") and not trend_strong:
                    signal = 0
                if signal != 0 and vol_low is not None and vol_day.iloc[i] < vol_low:
                    signal = 0
                if signal != 0 and vol_high is not None and vol_day.iloc[i] > vol_high:
                    signal = 0
                if signal != 0 and self.strategy_config.get("min_edge_return") is not None:
                    expected_move = abs(z_score) * sig.iloc[i]
                    expected_return = expected_move / price if price else 0
                    if self.strategy_config.get("dyn_edge_enabled"):
                        required_edge = (self.strategy_config.get("dyn_edge_fee_mult", 2.0) * self.engine.fee)
                        required_edge += self.strategy_config.get("dyn_edge_vol_mult", 0.0) * (sig.iloc[i] / price if price else 0)
                        if expected_return < required_edge:
                            signal = 0
                    elif expected_return < self.strategy_config["min_edge_return"]:
                        signal = 0

            # 动态持仓
            pos = signal * scale
            if signal != 0 and self.strategy_config.get("tiered_entry_enabled"):
                tier1 = self.strategy_config.get("tier1_enter", 1.2)
                tier2 = self.strategy_config.get("tier2_enter", 1.8)
                tier1_size = self.strategy_config.get("tier1_size", 0.6)
                tier2_size = self.strategy_config.get("tier2_size", 1.0)
                tier_size = tier2_size if abs(z_score) >= tier2 else tier1_size
                pos = signal * scale * tier_size
            elif signal != 0 and self.strategy_config.get("nonlinear_pos_enabled"):
                z_enter = self.engine.z_enter
                z_abs = abs(z_score)
                if z_abs <= z_enter:
                    size_mult = self.strategy_config.get("nonlinear_pos_min", 0.4)
                else:
                    z_ratio = z_abs / z_enter
                    alpha = self.strategy_config.get("nonlinear_pos_alpha", 1.5)
                    size_mult = self.strategy_config.get("nonlinear_pos_min", 0.4) + (
                        (z_ratio ** alpha) - 1.0
                    ) * (
                        self.strategy_config.get("nonlinear_pos_max", 1.0)
                        - self.strategy_config.get("nonlinear_pos_min", 0.4)
                    )
                size_mult = max(self.strategy_config.get("nonlinear_pos_min", 0.4),
                                min(self.strategy_config.get("nonlinear_pos_max", 1.0), size_mult))
                pos = signal * scale * size_mult
            pos = max(min(pos, self.engine.inv_cap), -self.engine.inv_cap)

            logger.info(
                f"[SIGNAL] time={ts} price={price:.2f} z={z_score:.3f} "
                f"signal={signal} pos={pos:.3f}"
            )

            # 检查是否需要取消冲突的待执行订单
            if self.order_execution_state != "NONE":
                should_cancel_orders = self._should_cancel_pending_orders(pos)
                if should_cancel_orders:
                    logger.info(f"🔄 New signal conflicts with pending orders - cancelling existing orders")
                    self._cancel_conflicting_orders()

            # 交易逻辑 - 增强版本，包含贤者模式保护
            trade = None

            # 1. 检查止损 - 在其他交易逻辑之前
            if self.current_position != 0:
                current_pnl = (
                    (price - self.entry_price) / self.entry_price
                    if self.current_position > 0
                    else (self.entry_price - price) / self.entry_price
                )

                # 使用策略引擎的共享止损逻辑
                is_long = self.current_position > 0
                if self.engine.should_trigger_stop_loss_price_based(
                    self.entry_price, price, is_long
                ):
                    # 触发止损 - 强制平仓并进入贤者模式
                    original_position = self.current_position  # 记录原始持仓方向

                    fee_cost = self._calc_fee_cost(0)
                    net_pnl = current_pnl - fee_cost
                    self.last_pnl = net_pnl
                    self.current_equity *= 1 + net_pnl

                    # 进入贤者模式
                    self.sage_mode = True
                    self.sage_mode_signal = (
                        1 if original_position > 0 else -1
                    )  # 记录原始信号方向
                    self.sage_mode_entry_time = ts

                    trade = {
                        "timestamp": str(ts),
                        "action": "stop_loss",
                        "price": self._safe_float(price),
                        "position": self._safe_float(original_position),
                        "pnl": self._safe_float(net_pnl),
                        "gross_pnl": self._safe_float(current_pnl),
                        "fee": self._safe_float(fee_cost),
                        "zscore": self._safe_float(z_score),
                    }

                    logger.info(
                        f"🛑 Stop loss triggered: price={price:.2f}, Z-score={z_score:.3f}, "
                        f"gross_pnl={current_pnl:.4f} fee={fee_cost:.4f} net_pnl={net_pnl:.4f} "
                        f"({net_pnl*100:.2f}%), original_position={'Long' if original_position > 0 else 'Short'}({original_position:.3f})"
                    )
                    logger.info(
                        f"🧘 Entering sage mode: waiting for {'Long' if self.sage_mode_signal > 0 else 'Short'} exit conditions"
                    )

                    # 强制平仓 - 确保仓位状态正确
                    self.current_position = 0
                    self.entry_price = 0
                    self.entry_time = None
                    self.last_exit_time = ts
                    
                    # 执行ETF交易（止损平仓）
                    self._execute_etf_trade(0, ts, price, z_score, etf_prices)

                    # 记录交易
                    if trade:
                        self.trade_history.append(trade)
                        trades_file = self.monitor_dir / "live_trades.json"
                        with open(trades_file, "w") as f:
                            json.dump(self.trade_history, f, indent=2)

                    # 更新状态并生成报告
                    self.save_state()
                    self.generate_status_report(
                        ts, price, z_score, 0
                    )  # signal=0 after stop loss
                    self.last_signal = 0
                    return True  # 表示发生了交易

            # 2. 检查贤者模式退出条件
            if self.sage_mode:
                # 检查是否满足原始信号的平仓条件
                should_exit_sage_mode = False

                # 使用策略引擎的共享贤者模式退出逻辑
                should_exit_sage_mode = self.engine.should_exit_sage_mode(
                    z_score, self.sage_mode_signal
                )
                if should_exit_sage_mode:
                    if self.sage_mode_signal > 0:
                        exit_reason = f"Long close signal met (Z-score {z_score:.3f} > {-self.engine.z_exit})"
                    else:
                        exit_reason = f"Short close signal met (Z-score {z_score:.3f} < {self.engine.z_exit})"

                if should_exit_sage_mode:
                    # 退出贤者模式
                    sage_duration = (
                        ts - self.sage_mode_entry_time
                    ).total_seconds() / 60

                    logger.info(
                        f"🌅 Exiting sage mode: {exit_reason}, Z-score={z_score:.3f}, duration={sage_duration:.1f}min"
                    )

                    self.sage_mode = False
                    self.sage_mode_signal = 0
                    self.sage_mode_entry_time = None
                    self.save_state()
                else:
                    # 仍在贤者模式，不进行任何交易，但仍要记录状态
                    logger.info(
                        f"🧘 In sage mode: Z-score={z_score:.3f}, waiting for {'Long' if self.sage_mode_signal > 0 else 'Short'} exit conditions"
                    )
                    # 在贤者模式期间也要生成状态报告，确保数据连续性
                    self.generate_status_report(ts, price, z_score, signal)
                    return False

            # 3. 正常交易逻辑 - 只有在非贤者模式下才执行
            if not self.sage_mode:
                if self.current_position == 0 and pos != 0:
                    # 开仓
                    fee_cost = self._calc_fee_cost(pos)
                    self.current_equity *= 1 - fee_cost
                    self.current_position = pos
                    self.entry_price = price
                    self.entry_time = ts
                    trade = {
                        "timestamp": str(ts),
                        "action": "open",
                        "price": self._safe_float(price),
                        "position": self._safe_float(pos),
                        "pnl": self._safe_float(-fee_cost),
                        "fee": self._safe_float(fee_cost),
                        "gross_pnl": 0.0,
                        "zscore": self._safe_float(z_score),
                    }
                    logger.info(
                        f"Open position: price={price:.2f}, Z-score={z_score:.3f}, "
                        f"position={pos:.3f}, fee={fee_cost:.4f}"
                    )
                    
                    # 执行ETF交易
                    self._execute_etf_trade(pos, ts, price, z_score, etf_prices)

                elif self.current_position != 0 and pos == 0:
                    # 平仓
                    pnl = (
                        (price - self.entry_price) / self.entry_price
                        if self.current_position > 0
                        else (self.entry_price - price) / self.entry_price
                    )
                    fee_cost = self._calc_fee_cost(0)
                    net_pnl = pnl - fee_cost
                    self.last_pnl = net_pnl
                    self.current_equity *= 1 + net_pnl
                    trade = {
                        "timestamp": str(ts),
                        "action": "close",
                        "price": self._safe_float(price),
                        "position": self._safe_float(self.current_position),
                        "pnl": self._safe_float(net_pnl),
                        "gross_pnl": self._safe_float(pnl),
                        "fee": self._safe_float(fee_cost),
                        "zscore": self._safe_float(z_score),
                    }
                    self.current_position = 0
                    self.entry_price = 0
                    self.entry_time = None
                    self.last_exit_time = ts
                    logger.info(
                        f"Close position: price={price:.2f}, Z-score={z_score:.3f}, "
                        f"gross_pnl={pnl:.4f} fee={fee_cost:.4f} net_pnl={net_pnl:.4f}"
                    )
                    
                    # 执行ETF交易（平仓）
                    self._execute_etf_trade(0, ts, price, z_score, etf_prices)

                elif self.current_position != 0 and (pos * self.current_position < 0):
                    # 反向平仓再开仓
                    pnl = (
                        (price - self.entry_price) / self.entry_price
                        if self.current_position > 0
                        else (self.entry_price - price) / self.entry_price
                    )
                    fee_cost = self._calc_fee_cost(pos)
                    net_pnl = pnl - fee_cost
                    self.last_pnl = net_pnl
                    self.current_equity *= 1 + net_pnl
                    trade = {
                        "timestamp": str(ts),
                        "action": "reverse",
                        "price": self._safe_float(price),
                        "position": self._safe_float(self.current_position),
                        "pnl": self._safe_float(net_pnl),
                        "gross_pnl": self._safe_float(pnl),
                        "fee": self._safe_float(fee_cost),
                        "zscore": self._safe_float(z_score),
                    }
                    self.current_position = pos
                    self.entry_price = price
                    self.entry_time = ts
                    logger.info(
                        f"Reverse: price={price:.2f}, Z-score={z_score:.3f}, "
                        f"new_position={pos:.3f}, gross_pnl={pnl:.4f} fee={fee_cost:.4f} "
                        f"net_pnl={net_pnl:.4f}"
                    )
                    
                    # 执行ETF交易（反向）
                    self._execute_etf_trade(pos, ts, price, z_score, etf_prices)

            # 记录交易
            if trade:
                self.trade_history.append(trade)

                # 保存交易记录
                trades_file = self.monitor_dir / "live_trades.json"
                with open(trades_file, "w") as f:
                    json.dump(self.trade_history, f, indent=2)

            # 更新状态
            self.save_state()

            # 生成状态报告
            self.generate_status_report(ts, price, z_score, signal)

            self.last_signal = signal
            self.last_pnl = 0  # 只在平仓/反向时计入

            return trade is not None

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            return False

    def generate_status_report(
        self, current_time, current_price, current_zscore, signal
    ):
        """生成状态报告"""
        # 计算统计信息 (与analyze_bar_by_bar_log.py保持一致)
        total_trades = len(
            [t for t in self.trade_history if t["action"] in ["close", "reverse"]]
        )
        winning_trades = len(
            [
                t
                for t in self.trade_history
                if t["action"] in ["close", "reverse"] and t["pnl"] > 0
            ]
        )
        win_rate = winning_trades / max(1, total_trades)
        avg_pnl = sum(
            [
                t["pnl"]
                for t in self.trade_history
                if t["action"] in ["close", "reverse"]
            ]
        ) / max(1, total_trades)

        status = {
            "timestamp": current_time.isoformat(),
            "current_price": current_price,
            "current_zscore": current_zscore,
            "current_signal": signal,
            "current_position": self.current_position,
            "current_equity": self.current_equity,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "return_pct": (self.current_equity - 1.0) * 100,
            "order_execution_state": self.order_execution_state,
            "pending_orders_count": len(self.pending_orders),
        }
        
        # 添加ETF状态信息
        if self.etf_enabled and self.etf_executor:
            try:
                etf_status = self.etf_executor.get_account_status()
                etf_summary = self.etf_executor.get_trade_summary(days=1)
                status.update({
                    "etf_enabled": True,
                    "etf_position": etf_status.get('current_etf_position', 'unknown'),
                    "etf_portfolio_value": etf_status.get('portfolio_value', 0),
                    "etf_positions": etf_status.get('etf_positions', {}),
                    "etf_trades_today": etf_summary.get('recent_trades', 0)
                })
            except Exception as e:
                logger.warning(f"ETF状态获取失败: {e}")
                status["etf_enabled"] = False
        else:
            status["etf_enabled"] = False

        # 保存状态报告
        status_file = self.monitor_dir / "current_status.json"
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)

        # 打印状态
        status_msg = (
            f"Status update: price={current_price:.2f}, Z-score={current_zscore:.3f}, "
            f"signal={signal}, position={self.current_position:.3f}, "
            f"equity={self.current_equity:.4f}, return={status['return_pct']:.2f}%"
        )
        
        # 添加订单执行状态信息
        if self.order_execution_state != "NONE":
            status_msg += f" | Order State: {self.order_execution_state}"
            if self.pending_orders:
                status_msg += f" ({len(self.pending_orders)} pending)"
        
        # 每10分钟添加ETF状态信息
        if self.etf_enabled and current_time.minute % 10 == 0:
            try:
                etf_info = f" | ETF: {status['etf_position']}, ${status['etf_portfolio_value']:,.0f}, Trades: {status['etf_trades_today']}"
                status_msg += etf_info
            except:
                pass
        
        logger.info(status_msg)

    def run(self):
        """运行监控"""
        logger.info("Starting live monitoring... (waiting for data)")
        
        # 显示ETF状态
        if self.etf_enabled:
            try:
                etf_status = self.etf_executor.get_account_status()
                logger.info(f"🏦 ETF账户初始状态: ${etf_status['portfolio_value']:,.2f} (市场 {'开盘' if etf_status['market_open'] else '收盘'})")
                if etf_status['etf_positions']:
                    pos_info = ", ".join([f"{sym}: {qty}" for sym, qty in etf_status['etf_positions'].items()])
                    logger.info(f"📊 ETF持仓: {pos_info}")
            except Exception as e:
                logger.warning(f"ETF状态检查失败: {e}")

        while True:
            try:
                current_time = datetime.now()

                # 检查待执行的限价单状态
                self._check_pending_orders()

                # Check for market close position flattening (every minute)
                if self.market_close_handler and current_time.second < 5:  # Check once per minute
                    try:
                        close_status = self.market_close_handler.check_and_flatten()
                        if close_status.get('positions_flattened'):
                            logger.warning("🚨 Positions flattened due to market close")
                            # Reset strategy position tracking
                            self.current_position = 0
                            self.entry_price = 0
                            self.entry_time = None
                            self.last_exit_time = current_time
                    except Exception as e:
                        logger.error(f"Market close handler error: {e}")

                # 检查是否需要更新（每分钟）
                if self.last_update is None or current_time - datetime.fromisoformat(
                    self.last_update
                ) >= timedelta(minutes=1):

                    # 更新数据 - now returns both crypto and ETF data
                    px, etf_px = self.update_data()
                    if px is not None:
                        # 执行策略 with ETF data
                        self.execute_strategy(px, etf_px)
                        self.last_update = current_time.isoformat()

                # 等待到下一分钟
                time.sleep(30)  # 每30秒检查一次

            except KeyboardInterrupt:
                logger.info("Monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # 异常时等待1分钟


def main():
    """主函数"""
    monitor = LiveMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
