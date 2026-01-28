# ===================================== #
#    python -m fina.core.trading_api    #
# ===================================== #


import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeApi
import pandas as pd
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from ..log_utils import get_logger

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Validate that required environment variables are set
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError(
        "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment variables or .env file"
    )


# Initialize logger
logger = get_logger()


class MarketTrader:
    """
    A class to handle stock trading operations using Alpaca Trade API
    with simulation mode support for testing and development.
    """

    def __init__(
        self,
        enable_simulation: bool = False,
    ):
        """
        Initialize the MarketTrader

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: Alpaca API base URL (paper trading by default)
            enable_simulation: Enable simulation mode for testing
        """
        self.enable_simulation = enable_simulation
        self.api_key = ALPACA_API_KEY
        self.secret_key = ALPACA_SECRET_KEY
        self.base_url = ALPACA_BASE_URL

        if not enable_simulation:
            try:
                self.api = tradeApi.REST(
                    self.api_key, self.secret_key, self.base_url, api_version="v2"
                )
                logger.info("Connected to Alpaca API")
            except Exception as e:
                logger.error(f"Failed to connect to Alpaca API: {e}")
                raise
        else:
            logger.info("Running in simulation mode")
            self.api = None
            # Simulation data
            self.sim_portfolio = {"cash": 10000.0, "positions": {}}
            self.sim_orders = []
            self.sim_order_id = 1000

    def get_account_info(self) -> Dict:
        """Get account information"""
        if self.enable_simulation:
            return {
                "id": "sim_account_123",
                "account_number": "SIM123456",
                "status": "ACTIVE",
                "currency": "USD",
                "cash": self.sim_portfolio["cash"],
                "portfolio_value": self.sim_portfolio["cash"]
                + sum(
                    pos["qty"] * pos["market_value"]
                    for pos in self.sim_portfolio["positions"].values()
                ),
                "pattern_day_trader": False,
                "trading_blocked": False,
                "transfers_blocked": False,
                "account_blocked": False,
                "created_at": "2024-01-01T00:00:00Z",
                "equity": self.sim_portfolio["cash"],
            }

        try:
            account = self.api.get_account()
            return {
                "id": account.id,
                "account_number": account.account_number,
                "status": account.status,
                "currency": account.currency,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
                "created_at": account.created_at,
                "equity": float(account.equity),
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            raise

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if self.enable_simulation:
            positions = []
            for symbol, pos in self.sim_portfolio["positions"].items():
                positions.append(
                    {
                        "symbol": symbol,
                        "qty": pos["qty"],
                        "market_value": pos["market_value"],
                        "cost_basis": pos["cost_basis"],
                        "unrealized_pl": pos["market_value"] - pos["cost_basis"],
                        "side": "long" if pos["qty"] > 0 else "short",
                    }
                )
            return positions

        try:
            positions = self.api.list_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "side": pos.side,
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Dict:
        """
        Place a trading order

        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
        """
        if self.enable_simulation:
            # Simulate order placement
            order_id = str(self.sim_order_id)
            self.sim_order_id += 1

            # Simulate market price (random between 50-200)
            market_price = random.uniform(50, 200)

            order = {
                "id": order_id,
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "order_type": order_type,
                "time_in_force": time_in_force,
                "limit_price": limit_price,
                "stop_price": stop_price,
                "status": "filled",
                "filled_qty": qty,
                "filled_price": market_price,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            # Update simulation portfolio
            if side == "buy":
                cost = qty * market_price
                if cost <= self.sim_portfolio["cash"]:
                    self.sim_portfolio["cash"] -= cost
                    if symbol in self.sim_portfolio["positions"]:
                        self.sim_portfolio["positions"][symbol]["qty"] += qty
                        self.sim_portfolio["positions"][symbol]["cost_basis"] += cost
                        self.sim_portfolio["positions"][symbol]["market_value"] = (
                            self.sim_portfolio["positions"][symbol]["qty"]
                            * market_price
                        )
                    else:
                        self.sim_portfolio["positions"][symbol] = {
                            "qty": qty,
                            "cost_basis": cost,
                            "market_value": qty * market_price,
                        }
                else:
                    order["status"] = "rejected"
                    order["filled_qty"] = 0

            elif side == "sell":
                if (
                    symbol in self.sim_portfolio["positions"]
                    and self.sim_portfolio["positions"][symbol]["qty"] >= qty
                ):
                    self.sim_portfolio["cash"] += qty * market_price
                    self.sim_portfolio["positions"][symbol]["qty"] -= qty
                    if self.sim_portfolio["positions"][symbol]["qty"] == 0:
                        del self.sim_portfolio["positions"][symbol]
                    else:
                        self.sim_portfolio["positions"][symbol]["market_value"] = (
                            self.sim_portfolio["positions"][symbol]["qty"]
                            * market_price
                        )
                else:
                    order["status"] = "rejected"
                    order["filled_qty"] = 0

            self.sim_orders.append(order)
            logger.info(f"Simulated order placed: {order}")
            return order

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
            )

            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": int(order.qty),
                "side": order.side,
                "order_type": order.type,
                "time_in_force": order.time_in_force,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
                "status": order.status,
                "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
                "filled_price": (
                    float(order.filled_avg_price) if order.filled_avg_price else None
                ),
                "created_at": order.created_at,
                "updated_at": order.updated_at,
            }
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    def get_orders(self, status: str = "all", limit: int = 100) -> List[Dict]:
        """Get order history"""
        if self.enable_simulation:
            orders = (
                self.sim_orders[-limit:]
                if limit < len(self.sim_orders)
                else self.sim_orders
            )
            if status != "all":
                orders = [order for order in orders if order["status"] == status]
            return orders

        try:
            orders = self.api.list_orders(status=status, limit=limit)
            return [
                {
                    "id": order.id,
                    "symbol": order.symbol,
                    "qty": int(order.qty),
                    "side": order.side,
                    "order_type": order.type,
                    "status": order.status,
                    "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
                    "filled_price": (
                        float(order.filled_avg_price)
                        if order.filled_avg_price
                        else None
                    ),
                    "created_at": order.created_at,
                    "updated_at": order.updated_at,
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if self.enable_simulation:
            # Find and cancel the order in simulation
            for order in self.sim_orders:
                if order["id"] == order_id and order["status"] == "new":
                    order["status"] = "canceled"
                    logger.info(f"Simulated order canceled: {order_id}")
                    return True
            return False

        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order canceled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False


class MarketObserver:
    """
    A class to observe market data and fetch historical/real-time stock information
    using Alpaca Trade API with robust error handling and retry mechanisms.
    """

    def __init__(
        self,
    ):
        """
        Initialize the MarketObserver

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: Alpaca API base URL
        """
        self.api_key = ALPACA_API_KEY
        self.secret_key = ALPACA_SECRET_KEY
        self.base_url = ALPACA_BASE_URL

        try:
            self.api = tradeApi.REST(
                self.api_key, self.secret_key, self.base_url, api_version="v2"
            )
            logger.info("MarketObserver connected to Alpaca API")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            raise

    def get_latest_price(self, symbol: str, retry_count: int = 3) -> Optional[Dict]:
        """
        Get the latest price for a stock symbol

        Args:
            symbol: Stock symbol
            retry_count: Number of retry attempts
        """
        for attempt in range(retry_count):
            try:
                # Get latest trade
                latest_trade = self.api.get_latest_trade(symbol)

                # Get latest quote
                latest_quote = self.api.get_latest_quote(symbol)

                return {
                    "symbol": symbol,
                    "price": float(latest_trade.price),
                    "timestamp": latest_trade.timestamp,
                    "bid": float(latest_quote.bid_price),
                    "ask": float(latest_quote.ask_price),
                    "bid_size": int(latest_quote.bid_size),
                    "ask_size": int(latest_quote.ask_size),
                    "spread": float(latest_quote.ask_price)
                    - float(latest_quote.bid_price),
                }

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed to get latest price for {symbol}: {e}"
                )
                if attempt < retry_count - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    logger.error(
                        f"Failed to get latest price for {symbol} after {retry_count} attempts"
                    )
                    return None

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
        retry_count: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical bar data for a stock symbol

        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 30Min, 1Hour, 1Day)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of bars to return
            retry_count: Number of retry attempts
        """
        for attempt in range(retry_count):
            try:
                # Set default dates if not provided
                if not end_date:
                    end_date = datetime.now().strftime("%Y-%m-%d")
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=365)).strftime(
                        "%Y-%m-%d"
                    )

                # Get historical bars
                bars = self.api.get_bars(
                    symbol, timeframe, start=start_date, end=end_date, limit=limit
                ).df

                if bars.empty:
                    logger.warning(f"No data found for {symbol}")
                    return None

                # Reset index to get timestamp as a column
                bars = bars.reset_index()

                logger.info(f"Retrieved {len(bars)} bars for {symbol}")
                return bars

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed to get historical data for {symbol}: {e}"
                )
                if attempt < retry_count - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    logger.error(
                        f"Failed to get historical data for {symbol} after {retry_count} attempts"
                    )
                    return None

    def get_multiple_latest_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get latest prices for multiple symbols

        Args:
            symbols: List of stock symbols
        """
        results = {}
        for symbol in symbols:
            price_data = self.get_latest_price(symbol)
            if price_data:
                results[symbol] = price_data
            else:
                results[symbol] = None
        return results

    def get_market_status(self) -> Dict:
        """Get current market status"""
        try:
            clock = self.api.get_clock()
            return {
                "timestamp": clock.timestamp,
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
                "timezone": "America/New_York",
            }
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            raise

    def get_stock_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Get basic stock information (this is a placeholder as Alpaca doesn't provide fundamentals)
        You would need to integrate with another service for fundamental data
        """
        try:
            # This is a basic implementation - you'd need to integrate with
            # services like Alpha Vantage, Yahoo Finance, or similar for fundamentals
            asset = self.api.get_asset(symbol)
            return {
                "symbol": asset.symbol,
                "name": asset.name,
                "exchange": asset.exchange,
                "asset_class": asset.asset_class,
                "status": asset.status,
                "tradable": asset.tradable,
                "marginable": asset.marginable,
                "shortable": asset.shortable,
                "easy_to_borrow": asset.easy_to_borrow,
                "fractionable": asset.fractionable,
            }
        except Exception as e:
            logger.error(f"Error getting stock fundamentals for {symbol}: {e}")
            return None

    def search_assets(self, query: str, asset_class: str = "us_equity") -> List[Dict]:
        """
        Search for assets by name or symbol

        Args:
            query: Search query
            asset_class: Asset class to search in
        """
        try:
            assets = self.api.list_assets(status="active", asset_class=asset_class)

            # Filter assets based on query
            matching_assets = []
            query_lower = query.lower()

            for asset in assets:
                if (
                    query_lower in asset.symbol.lower()
                    or query_lower in asset.name.lower()
                ):
                    matching_assets.append(
                        {
                            "symbol": asset.symbol,
                            "name": asset.name,
                            "exchange": asset.exchange,
                            "asset_class": asset.asset_class,
                            "status": asset.status,
                            "tradable": asset.tradable,
                        }
                    )

            return matching_assets[:50]  # Limit to 50 results

        except Exception as e:
            logger.error(f"Error searching assets: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Example 1: Using MarketTrader in simulation mode
    print("=== MarketTrader Example (Simulation Mode) ===\n")
    trader = MarketTrader(enable_simulation=True)

    # Get account info
    account = trader.get_account_info()
    print(f"Account cash: ${account['cash']:.2f}")

    # Place a buy order
    order = trader.place_order("AAPL", 10, "buy", "market")
    print(f"Order placed: {order['id']}, Status: {order['status']}")

    # Get positions
    positions = trader.get_positions()
    print(f"Current positions: {len(positions)}")

    # Example 2: Using MarketObserver
    print("\n=== MarketObserver Example ===\n")
    observer = MarketObserver()

    # Get latest price
    price_data = observer.get_latest_price("AAPL")
    if price_data:
        print(f"AAPL latest price: ${price_data['price']:.2f}")

    # Get historical data
    historical_data = observer.get_historical_data("AAPL", "1Day", limit=5)
    if historical_data is not None:
        print(f"Historical data shape: {historical_data.shape}")
        print(historical_data.head())

    # Get market status
    market_status = observer.get_market_status()
    print(f"Market is open: {market_status['is_open']}")
