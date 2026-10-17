"""Turns target weights into concrete orders, enforcing hard risk limits."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal

from .config import DATA_DIR, KILL_SWITCH, Settings

STATE_FILE = DATA_DIR / "state.json"


@dataclass
class Market:
    symbol: str
    bid: float            # what we'd get selling (spread included)
    ask: float            # what we'd pay buying (spread included)
    quantity: float       # coins held
    available: float      # coins available to sell
    asset_increment: str = "0.00000001"
    min_order_size: str = "0"


@dataclass
class Proposal:
    symbol: str
    side: str
    quantity: Decimal
    price: float
    notional: float
    current_value: float
    target_value: float
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quantity"] = format(self.quantity, "f")
        return d


def round_down(value: float, increment: str) -> Decimal:
    inc = Decimal(increment)
    if inc <= 0:
        return Decimal(str(value))
    return (Decimal(str(value)) / inc).to_integral_value(rounding=ROUND_DOWN) * inc


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def daily_loss(equity: float) -> float:
    """Fractional loss since the first equity snapshot of the UTC day (0 if up)."""
    today = datetime.now(timezone.utc).date().isoformat()
    state = _load_state()
    if state.get("day") != today or not state.get("start_equity"):
        state = {"day": today, "start_equity": equity}
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state))
    return max(0.0, 1 - equity / state["start_equity"])


def cap_weights(targets: dict[str, float], s: Settings, unmanaged_pct: float = 0.0) -> dict[str, float]:
    """Scale strategy exposures into portfolio weights. Coins the bot doesn't manage
    still count toward the total-exposure cap."""
    w = {k: max(0.0, min(v, 1.0)) * s.max_position_pct for k, v in targets.items()}
    budget = max(0.0, s.max_total_exposure_pct - unmanaged_pct)
    total = sum(w.values())
    if total > budget:
        w = {k: v * budget / total for k, v in w.items()} if total > 0 else w
    return w


def build_proposals(
    equity: float, cash: float, markets: dict[str, Market], targets: dict[str, float],
    reasons: dict[str, str], s: Settings, loss_today: float = 0.0,
) -> tuple[list[Proposal], list[str]]:
    """targets: symbol -> desired exposure (0..1) from the strategy. Symbols not in
    targets are left alone (no edge found = no opinion)."""
    notes: list[str] = []
    if KILL_SWITCH.exists():
        return [], [f"Kill switch is on ({KILL_SWITCH.name} file exists). No orders."]
    if equity <= 0:
        return [], ["Account equity is zero."]

    buys_blocked = loss_today >= s.daily_loss_limit_pct
    if buys_blocked:
        notes.append(f"Down {loss_today:.1%} today (limit {s.daily_loss_limit_pct:.0%}): new buys blocked, sells allowed.")

    unmanaged_pct = sum(m.quantity * m.bid for k, m in markets.items() if k not in targets) / equity
    for k, m in markets.items():
        w = m.quantity * m.bid / equity
        if k not in targets and w > s.max_position_pct:
            notes.append(f"{k} is {w:.0%} of equity, above your {s.max_position_pct:.0%} per-coin limit. "
                         "The bot isn't managing it (no proven edge), so it won't sell it for you.")
    if unmanaged_pct > 0 and targets:
        notes.append(f"Coins the bot doesn't manage are {unmanaged_pct:.0%} of equity; that counts toward the "
                     f"{s.max_total_exposure_pct:.0%} total crypto cap.")
    weights = cap_weights(targets, s, unmanaged_pct)
    band = max(s.min_trade_usd, s.rebalance_threshold_pct * equity)
    sells, buys = [], []
    cash_left = cash * 0.98  # buffer for price moves between quote and fill

    for sym, w in weights.items():
        m = markets.get(sym)
        if m is None:
            notes.append(f"{sym}: no quote, skipped.")
            continue
        current = m.quantity * m.bid
        target = w * equity
        diff = target - current
        if abs(diff) < band:
            continue
        if diff < 0:
            qty = round_down(min(-diff / m.bid, m.available), m.asset_increment)
            if qty <= 0 or qty < Decimal(m.min_order_size):
                continue
            sells.append(Proposal(sym, "sell", qty, m.bid, float(qty) * m.bid, current, target, reasons.get(sym, "")))
        else:
            if buys_blocked:
                continue
            notional = min(diff, s.max_order_usd, cash_left)
            if notional < s.min_trade_usd:
                notes.append(f"{sym}: wants ${diff:,.0f} more but only ${cash_left:,.0f} cash available.")
                continue
            qty = round_down(notional / m.ask, m.asset_increment)
            if qty <= 0 or qty < Decimal(m.min_order_size):
                continue
            cost = float(qty) * m.ask
            cash_left -= cost
            buys.append(Proposal(sym, "buy", qty, m.ask, cost, current, target, reasons.get(sym, "")))

    return sells + buys, notes
