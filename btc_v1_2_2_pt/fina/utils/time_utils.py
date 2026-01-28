"""
时间工具模块 - 统一时间处理
使用 pd.Timestamp 作为标准时间格式，统一使用 US/Eastern 时区
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
import pytz

# 设置纽约时区
NY_TZ = pytz.timezone("US/Eastern")


def get_now() -> pd.Timestamp:
    """获取当前纽约时间"""
    return pd.Timestamp.now(tz=NY_TZ)


def convert_to_ny_time(timestamp) -> pd.Timestamp:
    """将任意时间格式转换为纽约时间的pd.Timestamp

    Args:
        timestamp: 支持的格式:
            - pd.Timestamp
            - datetime.datetime
            - str (ISO格式)
            - float/int (Unix timestamp)
    """
    if isinstance(timestamp, pd.Timestamp):
        if timestamp.tz is None:
            return timestamp.tz_localize(NY_TZ)
        return timestamp.tz_convert(NY_TZ)

    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            return pd.Timestamp(timestamp).tz_localize(NY_TZ)
        return pd.Timestamp(timestamp).tz_convert(NY_TZ)

    if isinstance(timestamp, (int, float)):
        # 如果是纳秒级时间戳，转换为秒
        if timestamp > 1e15:
            timestamp = timestamp / 1e9
        return pd.Timestamp.fromtimestamp(timestamp, tz=NY_TZ)

    if isinstance(timestamp, str):
        ts = pd.Timestamp(timestamp)
        # 如果时间戳没有时区信息，先本地化再转换
        if ts.tz is None:
            return ts.tz_localize(NY_TZ)
        # 如果有时区信息，转换到NY时区
        return ts.tz_convert(NY_TZ)

    raise ValueError(f"Unsupported timestamp format: {type(timestamp)}")


def format_timestamp(timestamp, fmt: str = None) -> str:
    """格式化时间戳为字符串

    Args:
        timestamp: 任意支持的时间格式
        fmt: 输出格式，默认为 'YYYY-MM-DD HH:mm:ss'
    """
    ts = convert_to_ny_time(timestamp)
    if fmt:
        return ts.strftime(fmt)
    return str(ts)


def is_market_hours(timestamp=None) -> bool:
    """判断是否为市场交易时间（美东时间 9:30 - 16:00）

    Args:
        timestamp: 要判断的时间点，默认为当前时间
    """
    if timestamp is None:
        timestamp = get_now()
    else:
        timestamp = convert_to_ny_time(timestamp)

    # 获取时间部分
    time = timestamp.time()
    # 判断是否在交易时间内
    return time >= pd.Timestamp("09:30").time() and time <= pd.Timestamp("16:00").time()


def get_previous_trading_day(timestamp=None) -> pd.Timestamp:
    """获取上一个交易日（简化版，仅考虑周末）

    Args:
        timestamp: 参考时间点，默认为当前时间
    """
    if timestamp is None:
        timestamp = get_now()
    else:
        timestamp = convert_to_ny_time(timestamp)

    # 简单处理，向前推一个工作日
    prev_day = timestamp - pd.Timedelta(days=1)
    while prev_day.weekday() > 4:  # 5和6代表周六和周日
        prev_day = prev_day - pd.Timedelta(days=1)

    return prev_day


if __name__ == "__main__":
    # 测试代码
    print(f"当前纽约时间: {get_now()}")
    print(f"是否交易时间: {is_market_hours()}")

    # 测试时间转换
    test_times = [
        datetime.now(),
        "2024-01-01 10:30:00",
        1704123600,  # Unix timestamp
        pd.Timestamp("2024-01-01 10:30:00"),
    ]

    for t in test_times:
        ny_time = convert_to_ny_time(t)
        print(f"\n原始时间: {t}")
        print(f"转换后: {ny_time}")
        print(f"格式化: {format_timestamp(ny_time)}")
