#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Manager for BTC-ETH Statistical Arbitrage Strategy
- Downloads real crypto data from Alpaca API
- Caches data for efficiency
- Handles data alignment and preprocessing
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fina.consts  # Import entire module to ensure load_dotenv() is called
import numpy as np
import pandas as pd
from fina.consts import CACHE_PATH
from fina.core.time_util import TimeUtil

warnings.filterwarnings("ignore")


class CryptoDataManager:
    """Crypto data manager for downloading and managing historical data"""

    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config.get("symbols", ["BTC/USD", "ETH/USD", "SOL/USD"])
        # Add ETF symbols for synchronized data fetching
        self.etf_symbols = ["BITO", "BITI"]
        self.cache_dir = CACHE_PATH
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DATA] CryptoDataManager initialized with symbols: {self.symbols}")
        print(f"[DATA] Also fetching ETF data for: {self.etf_symbols}")

    def get_or_download_crypto_data(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Download or get cached crypto data from Alpaca"""
        print(
            f"[DATA] Fetching crypto data for {self.symbols} from {start_date} to {end_date}"
        )

        # Create cache filename
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")
        cache_file = self.cache_dir / f"crypto_data_{start_str}_to_{end_str}.csv"

        # Check if cache exists
        if cache_file.exists():
            print(f"[DATA] Found cached data: {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=["timestamp"])
            print(f"[DATA] Cached data loaded, records: {len(df)}")

            # Check if data is complete - FIX: Use current time instead of calculated end_date
            last_ts = df["timestamp"].max()
            last_ts_naive = pd.to_datetime(last_ts).tz_localize(None)
            
            # FIX: Compare against current UTC time to handle date boundary correctly
            import datetime
            current_utc = datetime.datetime.now(datetime.timezone.utc)
            end_date_naive = current_utc.replace(tzinfo=None)

            if last_ts_naive < end_date_naive:
                # FIX: Use current UTC time for incremental end instead of calculated date
                incremental_end = current_utc
                print(
                    f"[DATA] Cached data incomplete, downloading incremental data from {TimeUtil.format_market_datetime(last_ts)} to {TimeUtil.format_market_datetime(incremental_end)}"
                )
                try:
                    new_data = self.download_crypto_data_alpaca(
                        self.symbols, last_ts, incremental_end
                    )
                    if not new_data.empty:
                        df = pd.concat([df, new_data], ignore_index=True)
                        df = df.drop_duplicates(subset=["timestamp", "symbol"])
                        df = df.sort_values(["timestamp", "symbol"])
                        df.to_csv(cache_file, index=False)
                        print(
                            f"[DATA] Incremental data merged and cached, total records: {len(df)}"
                        )
                except Exception as e:
                    print(f"[ERROR] Incremental download failed: {e}")
            else:
                print("[DATA] Cached data is complete")
        else:
            print(f"[DATA] No cache found, downloading full dataset from Alpaca")
            try:
                df = self.download_crypto_data_alpaca(
                    self.symbols, start_date, end_date
                )
                if not df.empty:
                    df.to_csv(cache_file, index=False)
                    print(
                        f"[DATA] Data downloaded and cached to {cache_file}, records: {len(df)}"
                    )
            except Exception as e:
                print(f"[ERROR] Data download failed: {e}")
                return pd.DataFrame()

        return df

    def download_crypto_data_alpaca(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Download crypto data from Alpaca API using Eastern Time"""
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame

            API_KEY = os.environ.get("ALPACA_API_KEY")
            SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

            if not API_KEY or not SECRET_KEY:
                print(
                    "[ERROR] ALPACA API keys not set. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables"
                )
                return pd.DataFrame()

            client = CryptoHistoricalDataClient()

            # Convert dates to Eastern Time for Alpaca API
            market_tz = TimeUtil.get_market_timezone()

            # Parse dates and ensure they're in Eastern Time
            if isinstance(start_date, str):
                start_dt = pd.to_datetime(start_date).tz_localize(market_tz)
            else:
                start_dt = TimeUtil.convert_to_market_time(start_date)

            if isinstance(end_date, str):
                end_dt = (
                    pd.to_datetime(end_date)
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(seconds=1)
                )
                end_dt = end_dt.tz_localize(market_tz)
            else:
                end_dt = TimeUtil.convert_to_market_time(end_date)
                end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

            print(
                f"[DATA] Requesting data from Alpaca: {TimeUtil.format_market_datetime(start_dt)} to {TimeUtil.format_market_datetime(end_dt)}"
            )

            request = CryptoBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=start_dt,
                end=end_dt,
            )

            bars = client.get_crypto_bars(request).df

            if bars.empty:
                print("[WARNING] No data received from Alpaca")
                return pd.DataFrame()

            df = bars.reset_index()
            print(f"[DATA] Downloaded {len(df)} records from Alpaca")
            return df

        except Exception as e:
            print(f"[ERROR] Failed to download data from Alpaca: {e}")
            return pd.DataFrame()

    def download_etf_data_alpaca(
        self, symbols: List[str], start_date, end_date
    ) -> pd.DataFrame:
        """Download ETF data from Alpaca"""
        try:
            from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame
            
            if os.getenv("ALPACA_API_KEY") is None:
                print(
                    "[WARNING] No Alpaca API key found, skipping ETF data download"
                )
                return pd.DataFrame()

            client = StockHistoricalDataClient()

            # Convert dates to Eastern Time for Alpaca API
            market_tz = TimeUtil.get_market_timezone()

            # Parse dates and ensure they're in Eastern Time
            if isinstance(start_date, str):
                start_dt = pd.to_datetime(start_date).tz_localize(market_tz)
            else:
                start_dt = TimeUtil.convert_to_market_time(start_date)

            if isinstance(end_date, str):
                end_dt = (
                    pd.to_datetime(end_date)
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(seconds=1)
                )
                end_dt = end_dt.tz_localize(market_tz)
            else:
                end_dt = TimeUtil.convert_to_market_time(end_date)
                end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

            print(
                f"[DATA] Requesting ETF data from Alpaca: {TimeUtil.format_market_datetime(start_dt)} to {TimeUtil.format_market_datetime(end_dt)}"
            )

            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=start_dt,
                end=end_dt,
            )

            bars = client.get_stock_bars(request).df

            if bars.empty:
                print("[WARNING] No ETF data received from Alpaca")
                return pd.DataFrame()

            df = bars.reset_index()
            print(f"[DATA] Downloaded {len(df)} ETF records from Alpaca")
            return df

        except Exception as e:
            print(f"[ERROR] Failed to download ETF data from Alpaca: {e}")
            return pd.DataFrame()

    def load_and_align_data_with_etf(
        self, start_date: str, end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """Load and align crypto data with synchronized ETF data"""
        print("[DATA] Loading and aligning crypto and ETF data...")
        
        # First get crypto data using existing method
        crypto_data, original_start = self.load_and_align_data(start_date, end_date)
        
        if crypto_data.empty:
            return crypto_data, pd.DataFrame(), original_start
            
        # Calculate required buffer for ETF data (same as crypto)
        window = self.config.get("window", 720)
        max_window = max(window, 1440)
        buffer_days = (max_window // 1440) + 2
        
        # Extend start date to include historical data
        start_date_dt = pd.to_datetime(start_date)
        extended_start = (start_date_dt - pd.Timedelta(days=buffer_days)).strftime(
            "%Y-%m-%d"
        )
        
        # Download ETF data
        print(f"[DATA] Fetching ETF data for {self.etf_symbols}")
        etf_data = self.download_etf_data_alpaca(self.etf_symbols, extended_start, end_date)
        
        if etf_data.empty:
            print("[WARNING] No ETF data available, using empty DataFrame")
            return crypto_data, pd.DataFrame(), original_start
            
        # Process ETF data similar to crypto data
        etf_data["timestamp"] = pd.to_datetime(etf_data["timestamp"])
        
        # Pivot ETF data
        etf_dfs = []
        for symbol in self.etf_symbols:
            symbol_data = etf_data[etf_data["symbol"] == symbol].copy()
            if symbol_data.empty:
                print(f"[WARNING] No data for ETF {symbol}")
                continue
            symbol_data = symbol_data[["timestamp", "close"]].rename(
                columns={"close": f"{symbol}_close"}
            )
            etf_dfs.append(symbol_data)
        
        if etf_dfs:
            # Merge ETF data
            etf_aligned = etf_dfs[0]
            for df in etf_dfs[1:]:
                etf_aligned = etf_aligned.merge(df, on="timestamp", how="outer")
            
            # Align ETF data with crypto timestamps
            etf_aligned = etf_aligned.set_index("timestamp").sort_index()
            etf_aligned = etf_aligned.reindex(crypto_data.index).ffill()
            
            print(f"[DATA] ETF data aligned, shape: {etf_aligned.shape}")
        else:
            etf_aligned = pd.DataFrame(index=crypto_data.index)
            
        return crypto_data, etf_aligned, original_start
    
    def load_and_align_data(
        self, start_date: str, end_date: str
    ) -> Tuple[pd.DataFrame, str]:
        """Load and align crypto data for the assets"""
        print("[DATA] Loading and aligning crypto data...")

        # Calculate required buffer for rolling windows
        window = self.config.get("window", 720)
        max_window = max(window, 1440)
        buffer_days = (max_window // 1440) + 2

        # Extend start date to include historical data
        start_date_dt = pd.to_datetime(start_date)
        extended_start = (start_date_dt - pd.Timedelta(days=buffer_days)).strftime(
            "%Y-%m-%d"
        )

        print(f"[DATA] Extended start date from {start_date} to {extended_start}")

        raw_data = self.get_or_download_crypto_data(extended_start, end_date)
        original_start = start_date

        if raw_data.empty:
            print("[ERROR] No data available")
            return pd.DataFrame(), original_start

        # Ensure timestamp column
        if "timestamp" not in raw_data.columns:
            raw_data = raw_data.rename(columns={raw_data.columns[0]: "timestamp"})

        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])

        # Pivot to get each asset as a column
        dfs = []
        for symbol in self.symbols:
            asset_name = symbol.replace("/", "_")
            asset_data = raw_data[raw_data["symbol"] == symbol].copy()
            if asset_data.empty:
                print(f"[WARNING] No data for {symbol}")
                continue
            asset_data = asset_data[["timestamp", "close"]].rename(
                columns={"close": asset_name}
            )
            dfs.append(asset_data)

        if not dfs:
            print("[ERROR] No valid asset data found")
            return pd.DataFrame(), original_start

        # Merge all assets
        data = dfs[0]
        for df in dfs[1:]:
            data = data.merge(df, on="timestamp", how="outer")

        # Create complete time index and forward fill missing values
        data = data.set_index("timestamp").sort_index()

        start_time = data.index.min()
        end_time = data.index.max()

        if start_time.tz is not None:
            idx = pd.date_range(start_time, end_time, freq="1min")
        else:
            idx = pd.date_range(start_time, end_time, freq="1min", tz="UTC")

        data = data.reindex(idx).ffill()

        print(f"[DATA] Data aligned, final shape: {data.shape}")
        return data, original_start

    def ensure_latest_data(self) -> pd.DataFrame:
        """
        Ensure latest 1-minute crypto data is available locally
        Returns the latest data DataFrame
        """
        print("[DATA] Ensuring latest crypto data...")

        # Get today's date range
        end_date = TimeUtil.today_market_date()
        start_date = TimeUtil.days_ago_market_date(1)  # Last 1 day

        # Get or download data
        df = self.get_or_download_crypto_data(start_date, end_date)

        if not df.empty:
            latest_time = df["timestamp"].max()
            print(
                f"[DATA] ✅ Latest data: {TimeUtil.format_market_datetime(latest_time)}"
            )
        else:
            print("[DATA] ❌ Failed to get latest data")

        return df

    def test_recent_data_download(self, days: int = 7) -> bool:
        """Test recent data download"""
        print(f"\n{'='*60}")
        print("TEST 1: Recent Data Download")
        print(f"{'='*60}")

        end_date = TimeUtil.today_market_date()
        start_date = TimeUtil.days_ago_market_date(days)

        print(f"[TEST] Downloading data from {start_date} to {end_date}")

        try:
            df = self.get_or_download_crypto_data(start_date, end_date)

            if not df.empty:
                print(f"[TEST] ✅ Data download successful!")
                print(f"[TEST] Records: {len(df)}")
                print(
                    f"[TEST] Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
                )
                return True
            else:
                print("[TEST] ❌ Data download failed")
                return False

        except Exception as e:
            print(f"[TEST] ❌ Error during data download: {e}")
            return False

    def test_incremental_update(self, days: int = 7) -> bool:
        """Test incremental data update"""
        print(f"\n{'='*60}")
        print("TEST 2: Incremental Data Update")
        print(f"{'='*60}")

        end_date = TimeUtil.today_market_date()
        start_date = TimeUtil.days_ago_market_date(days)

        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")
        cache_file = self.cache_dir / f"crypto_data_{start_str}_to_{end_str}.csv"

        print(f"[TEST] Testing incremental update for cache: {cache_file}")

        try:
            # Step 1: Download full dataset
            print("[TEST] Step 1: Downloading full dataset...")
            df_full = self.get_or_download_crypto_data(start_date, end_date)

            if df_full.empty:
                print("[TEST] ❌ Cannot download initial data")
                return False

            original_count = len(df_full)
            print(f"[TEST] Original data count: {original_count}")

            # Step 2: Artificially truncate the cache
            print("[TEST] Step 2: Artificially truncating cache...")
            if cache_file.exists():
                df_truncated = pd.read_csv(cache_file, parse_dates=["timestamp"])
                cutoff_time = df_truncated["timestamp"].max() - pd.Timedelta(hours=6)
                df_truncated = df_truncated[df_truncated["timestamp"] <= cutoff_time]
                df_truncated.to_csv(cache_file, index=False)
                print(
                    f"[TEST] Truncated to {len(df_truncated)} records (removed last 6 hours)"
                )

            # Step 3: Try to download again
            print("[TEST] Step 3: Triggering incremental update...")
            df_updated = self.get_or_download_crypto_data(start_date, end_date)

            if not df_updated.empty:
                updated_count = len(df_updated)
                print(f"[TEST] Updated data count: {updated_count}")

                if updated_count >= original_count:
                    print("[TEST] ✅ Incremental update successful!")
                    return True
                else:
                    print(
                        "[TEST] ❌ Incremental update failed - data not fully restored"
                    )
                    return False
            else:
                print("[TEST] ❌ Incremental update failed - no data returned")
                return False

        except Exception as e:
            print(f"[TEST] ❌ Error during incremental update test: {e}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all data manager tests"""
        print("\n" + "=" * 80)
        print("DATA MANAGER TESTS")
        print("=" * 80)

        results = {}
        results["recent_download"] = self.test_recent_data_download()
        results["incremental_update"] = self.test_incremental_update()

        # Summary
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:20s}: {status}")

        all_passed = all(results.values())
        print(
            f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
        )

        return results


if __name__ == "__main__":
    """Run data manager tests when executed directly"""
    import argparse

    parser = argparse.ArgumentParser(description="Crypto Data Manager Tests")
    parser.add_argument(
        "--test",
        choices=["download", "incremental", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days for testing"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USD", "ETH/USD", "SOL/USD"],
        help="Symbols to test",
    )

    args = parser.parse_args()

    config = {"symbols": args.symbols}
    dm = CryptoDataManager(config)

    print(f"Running tests for symbols: {args.symbols}")

    if args.test == "download":
        print("\nRunning download test only...")
        success = dm.test_recent_data_download(args.days)
        print(f"\nDownload test: {'✅ PASSED' if success else '❌ FAILED'}")

    elif args.test == "incremental":
        print("\nRunning incremental update test only...")
        success = dm.test_incremental_update(args.days)
        print(f"\nIncremental test: {'✅ PASSED' if success else '❌ FAILED'}")

    else:  # all
        print("\nRunning all tests...")
        results = dm.run_all_tests()

        if all(results.values()):
            print("\n🎉 All tests passed!")
            exit(0)
        else:
            print("\n❌ Some tests failed!")
            exit(1)
