# ====================================== #
#      python -m fina.log_utils.utils    #
# ====================================== #


import pandas as pd
import numpy as np
from typing import List, TypedDict


class Trade(TypedDict):
    timestamp: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float


# ====================================== #
#              Sharpe Ratio              #
# ====================================== #


def calculate_sharpe_ratio(
    trades: List[Trade],
    initial_capital: float = 10000,
    risk_free_rate: float = 0.045,
) -> float:
    """
    Calculate Sharpe Ratio from trades data for a single symbol.

    Args:
        trades: List of trade dictionaries for a single symbol
        initial_capital: Starting capital amount
        risk_free_rate: Annual risk-free rate (e.g., 0.045 for 4.5%)

    Returns:
        Sharpe ratio (annualized)
    """

    if not trades:
        return 0.0

    # Convert trades to DataFrame
    df_trades = pd.DataFrame(trades)
    df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
    df_trades = df_trades.sort_values("timestamp")

    # Calculate portfolio value after each trade
    portfolio_values = []
    current_position = 0
    current_cash = initial_capital

    for _, trade in df_trades.iterrows():
        action = trade["action"]
        quantity = trade["quantity"]
        price = trade["price"]

        # Update cash and position
        if action == "BUY":
            current_cash -= quantity * price
            current_position += quantity
        else:  # SELL
            current_cash += quantity * price
            current_position -= quantity

        # Calculate total portfolio value (cash + position at current price)
        portfolio_value = current_cash + current_position * price

        portfolio_values.append(
            {"timestamp": trade["timestamp"], "portfolio_value": portfolio_value}
        )

    # Convert to DataFrame
    df_portfolio = pd.DataFrame(portfolio_values)

    # Calculate daily returns
    df_portfolio["date"] = df_portfolio["timestamp"].dt.date
    daily_portfolio = (
        df_portfolio.groupby("date")["portfolio_value"].last().reset_index()
    )

    if len(daily_portfolio) < 2:
        return 0.0

    # Calculate daily returns
    daily_portfolio["daily_return"] = daily_portfolio["portfolio_value"].pct_change()
    daily_returns = daily_portfolio["daily_return"].dropna()

    if len(daily_returns) < 2:
        return 0.0

    # Calculate metrics
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    # Convert annual risk-free rate to daily
    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1

    # Calculate Sharpe ratio
    if std_daily_return == 0:
        return 0.0

    sharpe_ratio = (avg_daily_return - daily_risk_free_rate) / std_daily_return

    # Annualize the Sharpe ratio
    return sharpe_ratio * np.sqrt(252)


def calculate_annualized_gross_return(
    initial_capital: float, trades: List[Trade]
) -> float:
    """
    Calculate Annualized Gross Return from initial capital and trades data.

    Args:
        initial_capital: Starting capital amount
        trades: List of trade dictionaries for a single symbol

    Returns:
        Annualized gross return as a decimal (e.g., 0.15 for 15%)
    """

    if not trades or initial_capital <= 0:
        return 0.0

    # Convert trades to DataFrame
    df_trades = pd.DataFrame(trades)
    df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
    df_trades = df_trades.sort_values("timestamp")

    # Calculate final capital by tracking cash and position
    current_position = 0
    current_cash = initial_capital

    for _, trade in df_trades.iterrows():
        action = trade["action"]
        quantity = trade["quantity"]
        price = trade["price"]

        if action == "BUY":
            current_cash -= quantity * price
            current_position += quantity
        else:  # SELL
            current_cash += quantity * price
            current_position -= quantity

    # Final capital = cash + position value at last trade price
    last_price = df_trades["price"].iloc[-1]
    final_capital = current_cash + current_position * last_price

    # Calculate years between first and last trade
    start_time = df_trades["timestamp"].iloc[0]
    end_time = df_trades["timestamp"].iloc[-1]
    time_diff = end_time - start_time
    years = time_diff.total_seconds() / (365.25 * 24 * 3600)  # Convert to years

    if years <= 0:
        return 0.0

    # Calculate annualized return: (Final/Initial)^(1/years) - 1
    return (final_capital / initial_capital) ** (1 / years) - 1


# ====================================== #
#           Maximum Drawdown             #
# ====================================== #


def calculate_max_drawdown(initial_capital: float, trades: List[Trade]) -> float:
    """
    Calculate Maximum Drawdown from initial capital and trades data.

    Args:
        initial_capital: Starting capital amount
        trades: List of trade dictionaries for a single symbol

    Returns:
        Maximum drawdown as a decimal (e.g., -0.15 for -15%)

    Raises:
        ValueError: If trades data is invalid
    """

    if not trades or initial_capital <= 0:
        return 0.0

    # Convert trades to DataFrame
    df_trades = pd.DataFrame(trades)
    df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
    df_trades = df_trades.sort_values("timestamp")

    # Calculate portfolio value after each trade and track drawdown
    current_position = 0
    current_cash = initial_capital
    peak_value = initial_capital
    max_drawdown = 0.0

    for _, trade in df_trades.iterrows():
        action = trade["action"]
        quantity = trade["quantity"]
        price = trade["price"]

        # Update cash and position
        if action == "BUY":
            current_cash -= quantity * price
            current_position += quantity
        else:  # SELL
            current_cash += quantity * price
            current_position -= quantity

        # Calculate current portfolio value
        portfolio_value = current_cash + current_position * price

        # Update peak value
        if portfolio_value > peak_value:
            peak_value = portfolio_value

        # Calculate current drawdown
        if peak_value > 0:
            current_drawdown = (portfolio_value - peak_value) / peak_value

            # Update max drawdown (most negative value)
            if current_drawdown < max_drawdown:
                max_drawdown = current_drawdown

    return max_drawdown


# ====================================== #
#              Example Usage             #
# ====================================== #

if __name__ == "__main__":
    # Example usage with your data structure
    example_data = {
        "strategy": "btc_eth_v1_simple",
        "config": {"symbols": ["BTC-USD"], "interval": "5min"},
        "performance": {
            "total_trades": 3,
            "total_pnl": -1000.0,
            "positions": {"BTC-USD": -0.02191570673276384},
        },
        "trades": [
            {
                "timestamp": "2025-07-05T14:12:20.823029",
                "action": "SELL",
                "symbol": "BTC-USD",
                "quantity": 0.02191570673276384,
                "price": 45629.37495896523,
                "position": -0.02191570673276384,
            },
            {
                "timestamp": "2025-07-06T14:12:30.847069",
                "action": "BUY",
                "symbol": "BTC-USD",
                "quantity": 0.40654330150316736,
                "price": 2459.7625795396584,
                "position": 0.40654330150316736,
            },
            {
                "timestamp": "2025-07-07T14:12:40.872980",
                "action": "BUY",
                "symbol": "BTC-USD",
                "quantity": 0.40736513564707055,
                "price": 2454.800159595325,
                "position": 0.8139084371502379,
            },
            {
                "timestamp": "2025-07-08T14:12:50.897980",
                "action": "SELL",
                "symbol": "BTC-USD",
                "quantity": 0.8139084371502379,
                "price": 2454.800159595325,
                "position": 0.0,
            },
        ],
    }

    # Calculate Sharpe ratio
    result = calculate_sharpe_ratio(
        example_data.get("trades", []), initial_capital=10000, risk_free_rate=0.045
    )
    print("[Example] Sharpe Ratio:", result)

    # Calculate Annualized Gross Return
    annualized_return = calculate_annualized_gross_return(
        initial_capital=10000, trades=example_data.get("trades", [])
    )
    print("[Example] Annualized Gross Return:", annualized_return)

    # Calculate Maximum Drawdown
    max_drawdown = calculate_max_drawdown(
        initial_capital=10000, trades=example_data.get("trades", [])
    )
    print("[Example] Maximum Drawdown:", max_drawdown)
