"""Settings loaded from environment / .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KILL_SWITCH = ROOT / "KILL"

load_dotenv(ROOT / ".env")

TIMEFRAMES = {
    # name: (coinbase granularity seconds, bars per year)
    "1d": (86400, 365),
    "6h": (21600, 1460),
}


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    rh_api_key: str
    rh_private_key: str
    symbols: list[str]
    timeframe: str
    min_oos_sharpe: float
    cost_bps: float
    max_position_pct: float
    max_total_exposure_pct: float
    max_order_usd: float
    min_trade_usd: float
    rebalance_threshold_pct: float
    daily_loss_limit_pct: float
    watch_interval_min: float
    watch_realert_min: float

    @property
    def granularity(self) -> int:
        return TIMEFRAMES[self.timeframe][0]

    @property
    def bars_per_year(self) -> int:
        return TIMEFRAMES[self.timeframe][1]


def load_settings() -> Settings:
    timeframe = os.getenv("TIMEFRAME", "1d").split("#")[0].strip()
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"TIMEFRAME must be one of {list(TIMEFRAMES)}, got {timeframe!r}")
    symbols = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTC-USD,ETH-USD,SOL-USD").split(",") if s.strip()]
    return Settings(
        rh_api_key=os.getenv("RH_API_KEY", "").strip(),
        rh_private_key=os.getenv("RH_PRIVATE_KEY", "").strip(),
        symbols=symbols,
        timeframe=timeframe,
        min_oos_sharpe=_float("MIN_OOS_SHARPE", 0.5),
        cost_bps=_float("COST_BPS", 50),
        max_position_pct=_float("MAX_POSITION_PCT", 0.25),
        max_total_exposure_pct=_float("MAX_TOTAL_EXPOSURE_PCT", 0.75),
        max_order_usd=_float("MAX_ORDER_USD", 1000),
        min_trade_usd=_float("MIN_TRADE_USD", 10),
        rebalance_threshold_pct=_float("REBALANCE_THRESHOLD_PCT", 0.03),
        daily_loss_limit_pct=_float("DAILY_LOSS_LIMIT_PCT", 0.05),
        watch_interval_min=_float("WATCH_INTERVAL_MIN", 5),
        watch_realert_min=_float("WATCH_REALERT_MIN", 60),
    )
