from typing import List
import numpy as np


def sharpe_ratio(returns: List[float], bars_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return 0.0
    return (mean / std) * (bars_per_year ** 0.5)


def max_drawdown(equity: List[float]) -> float:
    if not equity:
        return 0.0
    arr = np.array(equity, dtype=float)
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / np.where(peaks == 0, 1, peaks)
    return float(drawdowns.min())


def total_return(equity: List[float]) -> float:
    if len(equity) < 2:
        return 0.0
    start = equity[0]
    end = equity[-1]
    if start == 0:
        return 0.0
    return (end - start) / start
