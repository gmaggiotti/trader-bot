"""Backtesting and walk-forward strategy selection.

The selection process itself is what gets tested: at each fold we pick the best
config using only the training window, then trade it on the next (unseen) window.
Stitching those unseen windows together gives an honest out-of-sample record.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategies import STRATEGIES, Strategy


def strategy_returns(df: pd.DataFrame, exposure: pd.Series, cost_bps: float) -> pd.Series:
    """Per-bar returns of holding `exposure` (decided at t, applied from t+1), net of costs."""
    ret = df["close"].pct_change().fillna(0.0)
    pos = exposure.shift(1).fillna(0.0)
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    return pos * ret - turnover * cost_bps / 1e4


def metrics(r: pd.Series, bpy: int, exposure: pd.Series | None = None) -> dict:
    if len(r) == 0:
        return {"total_return": 0.0, "cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0, "exposure": 0.0}
    eq = (1 + r).cumprod()
    years = len(r) / bpy
    std = r.std()
    final = float(eq.iloc[-1])
    out = {
        "total_return": final - 1,
        "cagr": final ** (1 / years) - 1 if years > 0 and final > 0 else -1.0,
        "vol": float(std * np.sqrt(bpy)),
        "sharpe": float(r.mean() / std * np.sqrt(bpy)) if std > 0 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
    }
    if exposure is not None:
        e = exposure.reindex(r.index).fillna(0.0)
        out["trades"] = int((e.diff().abs() > 0.05).sum())
        out["exposure"] = float(e.mean())
    return out


@dataclass
class Config:
    strategy: str
    params: dict
    exposure: pd.Series
    returns: pd.Series

    @property
    def label(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.strategy}({p})" if p else self.strategy


def all_configs(df: pd.DataFrame, bpy: int, cost_bps: float, strategies: list[Strategy] = STRATEGIES) -> list[Config]:
    out = []
    for s in strategies:
        for params in s.configs():
            exp = s.fn(df, bpy, **params).clip(0, 1)
            out.append(Config(s.name, params, exp, strategy_returns(df, exp, cost_bps)))
    return out


def _sharpe(r: pd.Series) -> float:
    std = r.std()
    return float(r.mean() / std) if std > 0 else -np.inf


@dataclass
class WalkForwardResult:
    oos_metrics: dict
    buyhold_oos_metrics: dict
    family_oos_sharpe: dict[str, float]
    current: Config
    current_exposure: float
    oos_start: pd.Timestamp
    folds: list[dict] = field(default_factory=list)
    oos_returns: pd.Series | None = None


def walk_forward(
    df: pd.DataFrame, bpy: int, cost_bps: float, train_bars: int, test_bars: int,
    strategies: list[Strategy] = STRATEGIES,
) -> WalkForwardResult:
    if len(df) < train_bars + test_bars:
        raise ValueError(f"need at least {train_bars + test_bars} bars, have {len(df)}")
    configs = all_configs(df, bpy, cost_bps, strategies)
    families = sorted({c.strategy for c in configs})

    meta_exposure = pd.Series(0.0, index=df.index)
    family_exposure = {f: pd.Series(0.0, index=df.index) for f in families}
    folds = []

    start = train_bars
    while start < len(df):
        end = min(start + test_bars, len(df))
        scores = [_sharpe(c.returns.iloc[start - train_bars:start]) for c in configs]
        best = configs[int(np.argmax(scores))]
        meta_exposure.iloc[start:end] = best.exposure.iloc[start:end]
        folds.append({"test_start": df.index[start], "pick": best.label, "train_sharpe": max(scores) * np.sqrt(bpy)})
        for f in families:
            idx = [i for i, c in enumerate(configs) if c.strategy == f]
            fb = configs[max(idx, key=lambda i: scores[i])]
            family_exposure[f].iloc[start:end] = fb.exposure.iloc[start:end]
        start = end

    # Exposure for bar t is applied at t+1, so the first OOS return is at train_bars + 1.
    oos = slice(train_bars + 1, None)
    meta_r = strategy_returns(df, meta_exposure, cost_bps).iloc[oos]
    bh = next(c for c in configs if c.strategy == "buy_hold")
    family_sharpe = {
        f: metrics(strategy_returns(df, e, cost_bps).iloc[oos], bpy)["sharpe"] for f, e in family_exposure.items()
    }

    # Live pick: best config on the most recent training window.
    recent = [_sharpe(c.returns.iloc[-train_bars:]) for c in configs]
    current = configs[int(np.argmax(recent))]

    return WalkForwardResult(
        oos_metrics=metrics(meta_r, bpy, meta_exposure),
        buyhold_oos_metrics=metrics(bh.returns.iloc[oos], bpy, bh.exposure),
        family_oos_sharpe=family_sharpe,
        current=current,
        current_exposure=float(current.exposure.iloc[-1]),
        oos_start=df.index[train_bars + 1],
        folds=folds,
        oos_returns=meta_r,
    )
