import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from .data_manager import DataManager

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.server.secret_key = "trading-agent-dashboard"
if os.environ.get("DASH_AUTH_DISABLED", "false").lower() not in {"true", "1", "yes"}:
    import dash_auth
    dash_auth.BasicAuth(app, {"paw": "paw12345"})

data_manager = DataManager(LOG_DIR)

app.layout = html.Div([
    dcc.Interval(id="interval-refresh", interval=10 * 1000, n_intervals=0),
    html.Div([
        html.P("trading agent", className="header-title"),
        html.P("paper trading dashboard", className="header-subtitle"),
    ], className="header"),
    html.Div([
        html.Div(id="metrics-cards", className="metrics-row"),
        html.Div([
            html.Div([
                html.Label("Strategy", className="input-label"),
                dcc.Dropdown(id="strategy-dropdown", options=[], value=None),
            ], className="control-item"),
            html.Div([
                html.Label("Window", className="input-label"),
                dcc.Dropdown(
                    id="window-dropdown",
                    options=[
                        {"label": "1H", "value": "1H"},
                        {"label": "6H", "value": "6H"},
                        {"label": "24H", "value": "24H"},
                        {"label": "7D", "value": "7D"},
                    ],
                    value="1H",
                    clearable=False,
                ),
            ], className="control-item"),
            html.Div([
                html.Label("Granularity", className="input-label"),
                dcc.Dropdown(
                    id="granularity-dropdown",
                    options=[
                        {"label": "1 Min", "value": "1min"},
                        {"label": "5 Min", "value": "5min"},
                        {"label": "1 Hour", "value": "1H"},
                        {"label": "1 Day", "value": "1D"},
                    ],
                    value="1min",
                    clearable=False,
                ),
            ], className="control-item"),
            html.Div([
                html.Label("Symbols", className="input-label"),
                dcc.Dropdown(id="symbols-dropdown", options=[], value=[], multi=True),
            ], className="control-item"),
        ], className="controls-row"),
        html.Div([
            dcc.Graph(id="equity-chart"),
            dcc.Graph(id="drawdown-chart"),
        ], className="chart-grid"),
        html.Div([
            dcc.Graph(id="price-signal-chart"),
        ], className="chart-grid"),
    ], className="container"),
])


def _metric_card(label: str, value: str, tone: str = "neutral") -> html.Div:
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className=f"metric-value {tone}"),
    ], className="metric-card")


@app.callback(
    Output("strategy-dropdown", "options"),
    Output("strategy-dropdown", "value"),
    Input("interval-refresh", "n_intervals"),
    State("strategy-dropdown", "value"),
)
def refresh_strategies(_, current_value):
    data = data_manager.load_all()
    prices = data["prices"]
    if prices.empty or "strategy" not in prices.columns:
        return no_update, no_update
    strategies = sorted(prices["strategy"].unique().tolist())
    options = [{"label": s, "value": s} for s in strategies]
    if current_value in strategies:
        return options, current_value
    return options, strategies[0] if strategies else None


@app.callback(
    Output("symbols-dropdown", "options"),
    Output("symbols-dropdown", "value"),
    Input("interval-refresh", "n_intervals"),
    Input("strategy-dropdown", "value"),
    State("symbols-dropdown", "value"),
)
def refresh_symbols(_, strategy, current_values):
    data = data_manager.load_all()
    prices = data["prices"]
    if prices.empty or "symbol" not in prices.columns:
        return no_update, no_update
    if strategy and "strategy" in prices.columns:
        prices = prices[prices["strategy"] == strategy]
    symbols = sorted(prices["symbol"].unique().tolist())
    options = [{"label": s, "value": s} for s in symbols]
    current_values = current_values or []
    kept = [s for s in current_values if s in symbols]
    if kept:
        return options, kept
    return options, symbols


@app.callback(
    Output("metrics-cards", "children"),
    Output("equity-chart", "figure"),
    Output("drawdown-chart", "figure"),
    Output("price-signal-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
    Input("strategy-dropdown", "value"),
    Input("window-dropdown", "value"),
    Input("granularity-dropdown", "value"),
    Input("symbols-dropdown", "value"),
)
def refresh_dashboard(_, strategy, window, granularity, symbols):
    data = data_manager.load_all()
    equity = data["equity"]
    prices = data["prices"]
    trades = data["trades"]
    signals = data["signals"]
    prices_full = prices.copy()
    if "volume" in prices_full.columns:
        prices_full["volume"] = pd.to_numeric(prices_full["volume"], errors="coerce")
        prices_full = prices_full[prices_full["volume"] != 0]
    prices = prices_full.copy()

    def _resample_close(df: pd.DataFrame, ts_col: str, freq: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df = df.dropna(subset=[ts_col])
        if df.empty:
            return pd.DataFrame()
        df = df.set_index(ts_col)
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        df = df[~df.index.isna()]
        if df.empty:
            return pd.DataFrame()
        out = (
            df.resample(freq)
            .agg({"close": "last"})
            .dropna(subset=["close"])
            .reset_index()
            .rename(columns={ts_col: "ts"})
        )
        return out

    if strategy:
        if "strategy" in equity.columns:
            equity = equity[equity["strategy"] == strategy]
        if "strategy" in prices.columns:
            prices = prices[prices["strategy"] == strategy]
        if "strategy" in trades.columns:
            trades = trades[trades["strategy"] == strategy]
        if "strategy" in signals.columns:
            signals = signals[signals["strategy"] == strategy]

    cutoff_ts = None
    max_ts = None
    if not prices.empty and "timestamp" in prices.columns:
        ts = pd.to_datetime(prices["timestamp"], errors="coerce", utc=True)
        ts = ts.dropna()
        if not ts.empty:
            max_ts = ts.max()
            if window == "1H":
                cutoff_ts = max_ts - pd.Timedelta(hours=1)
            elif window == "6H":
                cutoff_ts = max_ts - pd.Timedelta(hours=6)
            elif window == "24H":
                cutoff_ts = max_ts - pd.Timedelta(hours=24)
            elif window == "7D":
                cutoff_ts = max_ts - pd.Timedelta(days=7)

    # Zoom mode: keep full data, only set chart x-axis range later.

    metrics = data_manager.compute_metrics(equity, trades)
    sharpe = f"{metrics['sharpe']:.2f}"
    total_ret = f"{metrics['return'] * 100:.2f}%"
    max_dd = f"{metrics['max_drawdown'] * 100:.2f}%"
    win_rate = f"{metrics['win_rate'] * 100:.1f}%"
    total_pnl = f"{metrics['total_pnl']:.2f}"
    avg_hold = f"{metrics['avg_hold_minutes']:.1f}m"
    trade_freq = f"{metrics['trade_freq_per_day']:.2f}/day"

    cards = [
        _metric_card("Sharpe", sharpe, "good" if metrics["sharpe"] > 0 else "bad"),
        _metric_card("Return", total_ret, "good" if metrics["return"] >= 0 else "bad"),
        _metric_card("Max Drawdown", max_dd, "bad"),
        _metric_card("Win Rate", win_rate, "good" if metrics["win_rate"] >= 0.5 else "bad"),
        _metric_card("Total PnL", total_pnl, "good" if metrics["total_pnl"] >= 0 else "bad"),
        _metric_card("Avg Hold", avg_hold, "neutral"),
        _metric_card("Trade Freq", trade_freq, "neutral"),
    ]

    equity_fig = go.Figure()
    equity_series = equity
    if not equity.empty:
        equity_series = equity.copy()
        equity_series["ts"] = pd.to_datetime(equity_series["timestamp"], errors="coerce", utc=True)
        equity_series["equity"] = pd.to_numeric(equity_series["equity"], errors="coerce")
        equity_series = equity_series.dropna(subset=["ts", "equity"]).sort_values("ts")
        resampled_eq = (
            equity_series.set_index("ts")
            .resample(granularity)
            .agg({"equity": "last"})
            .dropna(subset=["equity"])
            .reset_index()
        )
        if not resampled_eq.empty:
            equity_series = resampled_eq
        equity_fig.add_trace(go.Scatter(
            x=equity_series["ts"],
            y=equity_series["equity"],
            mode="lines",
            name="Equity",
        ))
    equity_fig.update_layout(title="Equity Curve", margin=dict(l=40, r=20, t=40, b=40))
    equity_fig.update_xaxes(tickmode="auto", nticks=8, tickformat="%m-%d %H:%M")
    if cutoff_ts is not None and max_ts is not None:
        equity_fig.update_xaxes(range=[cutoff_ts, max_ts])

    drawdown_fig = go.Figure()
    if not equity_series.empty:
        eq = equity_series["equity"].astype(float).to_numpy()
        peaks = np.maximum.accumulate(eq)
        drawdown = (eq - peaks) / np.maximum(peaks, 1e-9)
        drawdown_fig.add_trace(go.Scatter(
            x=equity_series["ts"],
            y=drawdown * 100,
            mode="lines",
            name="Drawdown %",
        ))
    drawdown_fig.update_layout(title="Drawdown", margin=dict(l=40, r=20, t=40, b=40))
    drawdown_fig.update_xaxes(tickmode="auto", nticks=8, tickformat="%m-%d %H:%M")
    if cutoff_ts is not None and max_ts is not None:
        drawdown_fig.update_xaxes(range=[cutoff_ts, max_ts])

    price_fig = make_subplots(specs=[[{"secondary_y": True}]])
    symbol_list = []
    if not prices.empty and "symbol" in prices.columns:
        available = prices["symbol"].unique().tolist()
        if symbols:
            symbol_list = [s for s in symbols if s in available]
        else:
            symbol_list = available

    price_series = {}
    # z-score series for pair strategy (use BTC/USD if present)
    sig_z = signals.copy()
    if "zscore" in sig_z.columns:
        sig_z["ts"] = pd.to_datetime(sig_z["timestamp"], errors="coerce", utc=True)
        sig_z["zscore"] = pd.to_numeric(sig_z["zscore"], errors="coerce")
        if "symbol" in sig_z.columns and "BTC/USD" in sig_z["symbol"].unique():
            sig_z = sig_z[sig_z["symbol"] == "BTC/USD"]
        sig_z = sig_z.dropna(subset=["ts", "zscore"]).sort_values("ts")
        sig_z = (
            sig_z.set_index("ts")[["zscore"]]
            .resample("1min")
            .mean()
            .dropna()
            .reset_index()
        )
    else:
        sig_z = pd.DataFrame()

    z_map = {}
    if not sig_z.empty:
        sig_z["ts_min"] = sig_z["ts"].dt.floor("min")
        sig_z = sig_z.drop_duplicates(subset=["ts_min"], keep="last")
        z_map = dict(zip(sig_z["ts_min"], sig_z["zscore"]))
    color_map = {
        "BTC/USD": "rgba(59,130,246,1)",
        "ETH/USD": "rgba(16,185,129,1)",
        "SOL/USD": "rgba(249,115,22,1)",
    }
    def _fade_color(sym: str) -> str:
        base = color_map.get(sym, "rgba(107,114,128,1)")
        return base.replace(",1)", ",0.35)")
    for symbol in symbol_list:
        prices_sym = prices[prices["symbol"] == symbol].copy()
        prices_sym["close"] = pd.to_numeric(prices_sym["close"], errors="coerce")
        prices_sym = prices_sym.dropna(subset=["close"])
        resampled = _resample_close(prices_sym, "timestamp", granularity)
        if not resampled.empty:
            prices_sym = resampled
        else:
            prices_sym["ts"] = pd.to_datetime(prices_sym["timestamp"], errors="coerce", utc=True)
            prices_sym = prices_sym.dropna(subset=["ts"]).sort_values("ts")
        price_series[symbol] = prices_sym[["ts", "close"]].copy()
        price_fig.add_trace(go.Scatter(
            x=prices_sym["ts"],
            y=prices_sym["close"],
            mode="lines",
            name=f"{symbol} close",
        ), secondary_y=False)
        # signal markers disabled (z-score line is sufficient)

    # trade markers (paw-style: map trades to z-score minute index)
    trade_mode = os.environ.get("TRADE_MODE", "offline").lower()
    if not sig_z.empty:
        symbol_for_trades = "BTC/USD"
        if symbol_for_trades not in (trades["symbol"].unique().tolist() if not trades.empty and "symbol" in trades.columns else []):
            symbol_for_trades = symbol_list[0] if symbol_list else "BTC/USD"

        if trade_mode == "online":
            if not trades.empty and "timestamp" in trades.columns:
                trades_sym = trades.copy()
                if "symbol" in trades_sym.columns and symbol_for_trades in trades_sym["symbol"].unique():
                    trades_sym = trades_sym[trades_sym["symbol"] == symbol_for_trades]
                trades_sym["ts"] = pd.to_datetime(trades_sym["timestamp"], errors="coerce", utc=True)
                trades_sym = trades_sym.dropna(subset=["ts"])
                trades_sym["ts_min"] = trades_sym["ts"].dt.floor("min")
                trades_sym["zscore"] = trades_sym["ts_min"].map(z_map)
                trades_sym = trades_sym.dropna(subset=["zscore"])
                for side, color, marker in [("buy", "green", "triangle-up"), ("sell", "red", "triangle-down")]:
                    side_df = trades_sym[trades_sym["side"] == side].copy()
                    if side_df.empty:
                        continue
                    price_fig.add_trace(go.Scatter(
                        x=side_df["ts_min"],
                        y=side_df["zscore"],
                        mode="markers",
                        name=f"{symbol_for_trades} {side}",
                        marker=dict(color=color, size=9, symbol=marker),
                        hovertemplate="time=%{x}<br>z=%{y:.3f}<br>qty=%{customdata}<extra>trade</extra>",
                        customdata=side_df.get("qty"),
                    ), secondary_y=True)
        else:
            # offline mode: mark entries/exits from target_pos changes
            if not signals.empty and "target_pos" in signals.columns:
                sig_tp = signals.copy()
                sig_tp["ts"] = pd.to_datetime(sig_tp["timestamp"], errors="coerce", utc=True)
                sig_tp["target_pos"] = pd.to_numeric(sig_tp["target_pos"], errors="coerce")
                if "symbol" in sig_tp.columns and "BTC/USD" in sig_tp["symbol"].unique():
                    sig_tp = sig_tp[sig_tp["symbol"] == "BTC/USD"]
                sig_tp = sig_tp.dropna(subset=["ts", "target_pos"]).sort_values("ts")
                sig_tp = sig_tp.set_index("ts").resample("1min").last().dropna().reset_index()
                sig_tp["prev_pos"] = sig_tp["target_pos"].shift(1).fillna(0.0)
                entries = sig_tp[(sig_tp["prev_pos"] == 0) & (sig_tp["target_pos"] != 0)]
                exits = sig_tp[(sig_tp["prev_pos"] != 0) & (sig_tp["target_pos"] == 0)]

                for df, color, marker, name in [
                    (entries, "green", "triangle-up", "entry"),
                    (exits, "red", "triangle-down", "exit"),
                ]:
                    if df.empty:
                        continue
                    df = df.copy()
                    df["ts_min"] = df["ts"].dt.floor("min")
                    df["zscore"] = df["ts_min"].map(z_map)
                    df = df.dropna(subset=["zscore"])
                    if df.empty:
                        continue
                    price_fig.add_trace(go.Scatter(
                        x=df["ts_min"],
                        y=df["zscore"],
                        mode="markers",
                        name=f"{symbol_for_trades} {name}",
                        marker=dict(color=color, size=9, symbol=marker),
                        hovertemplate="time=%{x}<br>z=%{y:.3f}<extra>{name}</extra>",
                    ), secondary_y=True)

    # z-score overlay on secondary axis for BTC/ETH pair if available
    if not sig_z.empty:
        price_fig.add_trace(go.Scatter(
            x=sig_z["ts"],
            y=sig_z["zscore"],
            mode="lines",
            name="z-score (strategy)",
            line=dict(color="#111827", width=1.5, dash="dot"),
        ), secondary_y=True)

        for level, name, color, show in [
            (1.2, "z_enter (entry)", "rgba(220,38,38,0.8)", True),
            (-1.2, "-z_enter", "rgba(220,38,38,0.8)", False),
            (0.4, "z_exit (exit)", "rgba(22,163,74,0.8)", True),
            (-0.4, "-z_exit", "rgba(22,163,74,0.8)", False),
        ]:
            price_fig.add_trace(go.Scatter(
                x=[sig_z["ts"].min(), sig_z["ts"].max()],
                y=[level, level],
                mode="lines",
                name=name,
                line=dict(color=color, width=1.2, dash="dash"),
                showlegend=show,
            ), secondary_y=True)
    price_fig.update_layout(title="Price + Signals", margin=dict(l=40, r=20, t=40, b=40))
    price_fig.update_xaxes(tickmode="auto", nticks=8, tickformat="%m-%d %H:%M", rangeslider=dict(visible=True))
    if cutoff_ts is not None and max_ts is not None:
        price_fig.update_xaxes(range=[cutoff_ts, max_ts])
    price_fig.update_yaxes(title_text="Price", secondary_y=False)
    price_fig.update_yaxes(title_text="Z-score", secondary_y=True)
    # mark preload -> live boundary if volume column exists

    return cards, equity_fig, drawdown_fig, price_fig


if __name__ == "__main__":
    host = os.environ.get("DASH_HOST", "0.0.0.0")
    debug = os.environ.get("DASH_DEBUG", "True").lower() == "true"
    app.run(debug=debug, host=host, port=8050)
