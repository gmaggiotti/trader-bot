"""Historical candles from Coinbase's free public API (Robinhood's API has no history).

Candles are cached on disk and topped up incrementally.
"""

import time
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import DATA_DIR

COINBASE = "https://api.exchange.coinbase.com/products/{product}/candles"
MAX_CANDLES = 300  # Coinbase limit per request


def _fetch_range(product: str, granularity: int, start: int, end: int) -> list[list]:
    rows: list[list] = []
    step = granularity * MAX_CANDLES
    t = start
    while t < end:
        t_end = min(t + step, end)
        params = {
            "granularity": granularity,
            "start": datetime.fromtimestamp(t, timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(t_end, timezone.utc).isoformat(),
        }
        for attempt in range(4):
            r = requests.get(COINBASE.format(product=product), params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1 + attempt)
                continue
            if r.status_code == 404:
                raise ValueError(f"No Coinbase market data for {product}")
            r.raise_for_status()
            rows.extend(r.json())
            break
        t = t_end
        time.sleep(0.25)
    return rows


def load_candles(product: str, granularity: int, history_days: int = 2000) -> pd.DataFrame:
    """Return OHLCV indexed by UTC bar open time, completed bars only."""
    cache = DATA_DIR / "cache" / f"{product}_{granularity}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    now = int(time.time())

    df = pd.read_csv(cache, index_col="time") if cache.exists() else pd.DataFrame()
    start = int(df.index.max()) + granularity if len(df) else now - history_days * 86400
    if now - start >= granularity:
        rows = _fetch_range(product, granularity, start, now)
        if rows:
            new = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]).set_index("time")
            df = pd.concat([df, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            # drop the bar that is still forming
            df = df[df.index + granularity <= now]
            df.to_csv(cache)

    if df.empty:
        raise ValueError(f"No candle data for {product}")
    out = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    out.index = pd.to_datetime(out.index.astype(int), unit="s", utc=True)
    return out
