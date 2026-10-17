"""Candidate strategies. Each returns target exposure in [0, 1] per bar.

Exposure at bar t may only use data up to and including bar t's close;
the backtester applies it from bar t+1. Robinhood crypto can't be shorted,
so "bearish" means 0 (cash).
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

StrategyFn = Callable[..., pd.Series]


def _latch(enter: pd.Series, exit_: pd.Series, index: pd.Index) -> pd.Series:
    """1 after an enter signal until an exit signal."""
    state = pd.Series(np.nan, index=index)
    state[exit_] = 0.0
    state[enter] = 1.0
    return state.ffill().fillna(0.0)


def buy_hold(df: pd.DataFrame, bpy: int) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sma_cross(df: pd.DataFrame, bpy: int, fast: int, slow: int) -> pd.Series:
    c = df["close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)


def ema_trend(df: pd.DataFrame, bpy: int, span: int) -> pd.Series:
    c = df["close"]
    ema = c.ewm(span=span, adjust=False, min_periods=span).mean()
    return (c > ema).astype(float)


def donchian(df: pd.DataFrame, bpy: int, entry: int, exit: int) -> pd.Series:
    upper = df["high"].rolling(entry).max().shift(1)
    lower = df["low"].rolling(exit).min().shift(1)
    return _latch(df["close"] > upper, df["close"] < lower, df.index)


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def rsi_reversion(df: pd.DataFrame, bpy: int, period: int, low: int, high: int) -> pd.Series:
    r = rsi(df["close"], period)
    return _latch(r < low, r > high, df.index)


def tsmom_vol(df: pd.DataFrame, bpy: int, lookback: int, target_vol: float) -> pd.Series:
    """Time-series momentum, sized so realized volatility ~= target_vol (capped at 100%)."""
    c = df["close"]
    trend_up = c.pct_change(lookback) > 0
    realized = c.pct_change().rolling(30).std() * np.sqrt(bpy)
    size = (target_vol / realized).clip(upper=1.0)
    return (size * trend_up).fillna(0.0)


@dataclass(frozen=True)
class Strategy:
    name: str
    fn: StrategyFn
    grid: dict[str, list]

    def configs(self) -> list[dict]:
        keys = list(self.grid)
        combos = [dict(zip(keys, vals)) for vals in product(*self.grid.values())] if keys else [{}]
        if self.name == "sma_cross":
            combos = [c for c in combos if c["fast"] < c["slow"]]
        return combos


STRATEGIES: list[Strategy] = [
    Strategy("buy_hold", buy_hold, {}),
    Strategy("sma_cross", sma_cross, {"fast": [10, 20, 50], "slow": [50, 100, 200]}),
    Strategy("ema_trend", ema_trend, {"span": [20, 50, 100, 200]}),
    Strategy("donchian", donchian, {"entry": [20, 55], "exit": [10, 20]}),
    Strategy("rsi_reversion", rsi_reversion, {"period": [14], "low": [25, 30, 35], "high": [55, 65, 75]}),
    Strategy("tsmom_vol", tsmom_vol, {"lookback": [30, 90, 180], "target_vol": [0.4, 0.8]}),
]

BY_NAME = {s.name: s for s in STRATEGIES}

TREND_FAMILIES = ("sma_cross", "ema_trend", "donchian", "tsmom_vol")


def trend_ensemble(df: pd.DataFrame, bpy: int) -> pd.Series:
    """Average of every trend config. Nothing is fitted, so there is nothing to overfit.

    In testing this was more robust than picking a 'best' strategy from recent
    performance: walk-forward picking chased past winners and lost to buy & hold.
    """
    exps = [s.fn(df, bpy, **p) for s in STRATEGIES if s.name in TREND_FAMILIES for p in s.configs()]
    return (sum(exps) / len(exps)).clip(0, 1)
