import base64
from decimal import Decimal

import numpy as np
import pandas as pd
from nacl.signing import VerifyKey

from trader_bot.backtest import strategy_returns, walk_forward
from trader_bot.config import Settings
from trader_bot.risk import Market, build_proposals, round_down
from trader_bot.robinhood import RobinhoodCrypto, generate_keypair
from trader_bot.strategies import STRATEGIES


def settings(**kw) -> Settings:
    base = dict(rh_api_key="k", rh_private_key="", symbols=["BTC-USD"], timeframe="1d", min_oos_sharpe=0.5,
                cost_bps=50, max_position_pct=0.25, max_total_exposure_pct=0.75, max_order_usd=1000,
                min_trade_usd=10, rebalance_threshold_pct=0.03, daily_loss_limit_pct=0.05,
                watch_interval_min=5, watch_realert_min=60)
    base.update(kw)
    return Settings(**base)


def prices(n=1500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1.0}, index=idx)


def test_signature_matches_robinhood_scheme():
    priv, pub = generate_keypair()
    rh = RobinhoodCrypto("my-key", priv)
    h = rh.sign("GET", "/api/v1/crypto/trading/accounts/", "", 1700000000)
    msg = b"my-key1700000000/api/v1/crypto/trading/accounts/GET"
    VerifyKey(base64.b64decode(pub)).verify(msg, base64.b64decode(h["x-signature"]))  # raises if wrong
    assert h["x-timestamp"] == "1700000000" and h["x-api-key"] == "my-key"


def test_strategies_do_not_look_ahead():
    df = prices()
    cut = 1000
    for s in STRATEGIES:
        for params in s.configs():
            full = s.fn(df, 365, **params)
            partial = s.fn(df.iloc[:cut], 365, **params)
            assert np.allclose(full.iloc[:cut].values, partial.values), f"{s.name}{params} uses future data"


def test_exposure_applied_next_bar_with_costs():
    df = prices(10)
    exp = pd.Series(0.0, index=df.index)
    exp.iloc[3] = 1.0  # decide at bar 3 -> hold during bar 4 only
    r = strategy_returns(df, exp, cost_bps=100)
    ret = df["close"].pct_change()
    assert abs(r.iloc[4] - (ret.iloc[4] - 0.01)) < 1e-12
    assert abs(r.iloc[5] - (-0.01)) < 1e-12  # exit cost
    assert r.iloc[3] == 0


def test_walk_forward_runs_and_reports():
    wf = walk_forward(prices(), 365, 50, train_bars=730, test_bars=90)
    assert 0 <= wf.current_exposure <= 1
    assert "sharpe" in wf.oos_metrics and len(wf.folds) > 5


def test_risk_caps_and_rounding():
    s = settings()
    m = {"BTC-USD": Market("BTC-USD", bid=100_000, ask=100_500, quantity=0, available=0, asset_increment="0.00001")}
    props, _ = build_proposals(10_000, 10_000, m, {"BTC-USD": 1.0}, {}, s)
    assert len(props) == 1 and props[0].side == "buy"
    assert props[0].notional <= 0.25 * 10_000 and props[0].notional <= s.max_order_usd
    assert props[0].quantity == round_down(props[0].quantity, "0.00001")


def test_daily_loss_blocks_buys_but_allows_sells():
    s = settings()
    m = {"A-USD": Market("A-USD", 10, 10, 0, 0), "B-USD": Market("B-USD", 10, 10, 300, 300)}
    props, notes = build_proposals(10_000, 7_000, m, {"A-USD": 1.0, "B-USD": 0.0}, {}, s, loss_today=0.06)
    assert [p.side for p in props] == ["sell"] and props[0].quantity == Decimal("300")
    assert any("blocked" in n for n in notes)


def test_small_drift_ignored():
    s = settings()
    m = {"A-USD": Market("A-USD", 10, 10, 245, 245)}  # $2450 held vs $2500 target
    props, _ = build_proposals(10_000, 7_550, m, {"A-USD": 1.0}, {}, s)
    assert props == []


def test_unmanaged_holdings_count_toward_total_cap():
    s = settings()
    m = {"A-USD": Market("A-USD", 10, 10, 0, 0), "B-USD": Market("B-USD", 10, 10, 0, 0),
         "C-USD": Market("C-USD", 10, 10, 600, 600)}  # $6000 unmanaged = 60% of equity
    props, notes = build_proposals(10_000, 4_000, m, {"A-USD": 1.0, "B-USD": 1.0}, {}, s)
    bought = sum(p.notional for p in props)
    assert bought <= 0.15 * 10_000 + 1  # 75% cap - 60% unmanaged
    assert any("per-coin limit" in n for n in notes)
