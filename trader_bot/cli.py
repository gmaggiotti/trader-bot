"""Command line: trader {keygen,status,backtest,suggest,orders,kill,resume}."""

import argparse
import json
import select
import subprocess
import sys
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from . import advisor
from .backtest import metrics, strategy_returns, walk_forward
from .config import DATA_DIR, KILL_SWITCH, Settings, load_settings
from .data import load_candles
from .risk import Market, Proposal, build_proposals, daily_loss
from .robinhood import RobinhoodCrypto, RobinhoodError, generate_keypair
from .strategies import trend_ensemble

console = Console()
JOURNAL = DATA_DIR / "journal.jsonl"


def journal(event: str, **data) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **data}
    with JOURNAL.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def client(s: Settings) -> RobinhoodCrypto:
    return RobinhoodCrypto(s.rh_api_key, s.rh_private_key)


# ---------------- analysis ----------------

def analyze(symbol: str, s: Settings) -> dict:
    """Score the live strategy (trend ensemble) against buy & hold and the adaptive
    picker, all on the same out-of-sample window."""
    df = load_candles(symbol, s.granularity)
    bpy = s.bars_per_year
    train = 2 * bpy
    wf = walk_forward(df, bpy, s.cost_bps, train_bars=train, test_bars=bpy // 4)
    oos = slice(train + 1, None)
    ens = trend_ensemble(df, bpy)
    ens_m = metrics(strategy_returns(df, ens, s.cost_bps).iloc[oos], bpy, ens)
    oos_years = len(wf.oos_returns) / bpy
    return {
        "symbol": symbol,
        "strategy": "trend_ensemble",
        "last_close": float(df["close"].iloc[-1]),
        "last_bar": df.index[-1].isoformat(),
        "signal": float(ens.iloc[-1]),
        "oos_years": round(oos_years, 1),
        "oos": ens_m,
        "buy_hold_oos": wf.buyhold_oos_metrics,
        "adaptive_oos": wf.oos_metrics,
        "adaptive_pick": wf.current.label,
        "family_oos_sharpe": wf.family_oos_sharpe,
        "gate_passed": ens_m["sharpe"] >= s.min_oos_sharpe and oos_years >= 1,
    }


def analyze_many(symbols: list[str], s: Settings) -> tuple[dict[str, dict], list[str]]:
    results, errors = {}, []
    for sym in symbols:
        with console.status(f"Backtesting {sym}..."):
            try:
                results[sym] = analyze(sym, s)
            except Exception as e:  # one bad symbol shouldn't stop the rest
                errors.append(f"{sym}: {e}")
    return results, errors


def print_analysis(results: dict[str, dict], s: Settings) -> None:
    t = Table(title=f"Out-of-sample results ({s.timeframe} bars, {s.cost_bps:.0f} bps cost/side)")
    cols = ["Symbol", "Signal", "Yrs", "Bot Sharpe", "Bot CAGR", "Bot MaxDD",
            "B&H Sharpe", "B&H CAGR", "B&H MaxDD", "Picker Sharpe", "Trade it?"]
    for col in cols:
        t.add_column(col, justify="left" if col == "Symbol" else "right")
    for r in results.values():
        o, b, a = r["oos"], r["buy_hold_oos"], r["adaptive_oos"]
        t.add_row(
            r["symbol"], f"{r['signal']:.0%}", str(r["oos_years"]),
            f"{o['sharpe']:.2f}", f"{o['cagr']:.0%}", f"{o['max_drawdown']:.0%}",
            f"{b['sharpe']:.2f}", f"{b['cagr']:.0%}", f"{b['max_drawdown']:.0%}",
            f"{a['sharpe']:.2f}",
            "[green]yes[/]" if r["gate_passed"] else "[yellow]no edge[/]",
        )
    console.print(t)
    console.print(
        "[dim]Bot = trend ensemble (live strategy). Signal = how much of the max position it wants now. "
        "B&H = buy & hold. Picker = switching to the recent best strategy (shown for comparison, not used).\n"
        f"'Trade it?' needs bot Sharpe ≥ {s.min_oos_sharpe} over ≥1 year. Past results don't guarantee future returns.[/]"
    )


# ---------------- commands ----------------

def cmd_keygen(args, s: Settings) -> None:
    priv, pub = generate_keypair()
    console.print("[bold]1.[/] In Robinhood (web) go to Account → Crypto → API trading and add a key.")
    console.print("[bold]2.[/] Paste this PUBLIC key there:\n")
    console.print(f"    {pub}\n")
    console.print("[bold]3.[/] Put these in your .env (keep the private key secret):\n")
    console.print(f"    RH_PRIVATE_KEY={priv}")
    console.print("    RH_API_KEY=<the API key Robinhood shows you>\n")


def load_portfolio(rh: RobinhoodCrypto, extra_symbols: list[str]) -> dict:
    account = rh.get_account()
    cash = float(account.get("buying_power", 0))
    holdings = {
        f"{h['asset_code']}-USD": h for h in rh.get_holdings() if float(h.get("total_quantity", 0)) > 0
    }
    symbols = sorted(set(extra_symbols) | set(holdings))
    pairs = {p["symbol"]: p for p in rh.get_trading_pairs(symbols)} if symbols else {}
    quotes = {q["symbol"]: q for q in rh.get_best_bid_ask(list(pairs))} if pairs else {}

    markets: dict[str, Market] = {}
    for sym, q in quotes.items():
        h = holdings.get(sym, {})
        p = pairs.get(sym, {})
        markets[sym] = Market(
            symbol=sym,
            bid=float(q.get("bid_inclusive_of_sell_spread") or q["price"]),
            ask=float(q.get("ask_inclusive_of_buy_spread") or q["price"]),
            quantity=float(h.get("total_quantity", 0)),
            available=float(h.get("quantity_available_for_trading", 0)),
            asset_increment=p.get("asset_increment", "0.00000001"),
            min_order_size=p.get("min_order_size", "0"),
        )
    equity = cash + sum(m.quantity * m.bid for m in markets.values())
    return {"account": account, "cash": cash, "holdings": holdings, "pairs": pairs, "markets": markets, "equity": equity}


def print_portfolio(pf: dict) -> None:
    t = Table(title=f"Robinhood crypto — equity ${pf['equity']:,.2f}  (cash ${pf['cash']:,.2f})")
    for col in ["Coin", "Quantity", "Bid", "Value", "Weight"]:
        t.add_column(col, justify="right" if col != "Coin" else "left")
    for sym, m in sorted(pf["markets"].items()):
        if m.quantity <= 0:
            continue
        value = m.quantity * m.bid
        t.add_row(sym, f"{m.quantity:g}", f"${m.bid:,.2f}", f"${value:,.2f}", f"{value / pf['equity']:.1%}")
    console.print(t)


def cmd_status(args, s: Settings) -> None:
    rh = client(s)
    pf = load_portfolio(rh, [])
    console.print(f"Account {pf['account'].get('account_number')} — status: {pf['account'].get('status')}")
    print_portfolio(pf)
    if KILL_SWITCH.exists():
        console.print("[red]Kill switch is ON.[/] Run `trader resume` to allow orders.")


def cmd_backtest(args, s: Settings) -> None:
    symbols = args.symbols.split(",") if args.symbols else s.symbols
    results, errors = analyze_many([x.strip().upper() for x in symbols], s)
    print_analysis(results, s)
    for e in errors:
        console.print(f"[yellow]Skipped {e}[/]")
    if args.verbose:
        for r in results.values():
            fam = ", ".join(f"{k} {v:.2f}" for k, v in sorted(r["family_oos_sharpe"].items(), key=lambda kv: -kv[1]))
            console.print(f"[dim]{r['symbol']} single-strategy OOS Sharpe: {fam}; picker would choose {r['adaptive_pick']}[/]")


# ---------------- planning / execution (shared by suggest and watch) ----------------

_analysis_cache: dict[tuple[str, str], dict] = {}


def analyze_cached(symbols: list[str], s: Settings) -> tuple[dict[str, dict], list[str]]:
    """Re-run the backtest only when a symbol gets a new completed bar."""
    results, errors = {}, []
    for sym in symbols:
        try:
            last_bar = load_candles(sym, s.granularity).index[-1].isoformat()
            key = (sym, last_bar)
            if key not in _analysis_cache:
                _analysis_cache[key] = analyze(sym, s)
            results[sym] = _analysis_cache[key]
        except Exception as e:
            errors.append(f"{sym}: {e}")
    return results, errors


def make_plan(rh: RobinhoodCrypto, s: Settings, quiet: bool = False) -> dict:
    pf = load_portfolio(rh, s.symbols)
    if quiet:
        results, errors = analyze_cached(list(pf["markets"]), s)
    else:
        results, errors = analyze_many(list(pf["markets"]), s)
    targets = {sym: r["signal"] for sym, r in results.items() if r["gate_passed"]}
    reasons = {
        sym: f"trend ensemble wants {r['signal']:.0%} of max position (out-of-sample Sharpe {r['oos']['sharpe']:.2f}, "
             f"max drop {r['oos']['max_drawdown']:.0%} vs buy & hold {r['buy_hold_oos']['max_drawdown']:.0%})"
        for sym, r in results.items()
    }
    loss = daily_loss(pf["equity"])
    proposals, notes = build_proposals(pf["equity"], pf["cash"], pf["markets"], targets, reasons, s, loss)
    return {"pf": pf, "results": results, "errors": errors, "targets": targets,
            "proposals": proposals, "notes": notes}


def print_plan_details(plan: dict, s: Settings) -> None:
    print_portfolio(plan["pf"])
    print_analysis(plan["results"], s)
    for e in plan["errors"]:
        console.print(f"[yellow]Skipped {e}[/]")
    for n in plan["notes"]:
        console.print(f"[yellow]{n}[/]")
    untouched = [sym for sym in plan["results"] if sym not in plan["targets"]]
    if untouched:
        console.print(f"[dim]No proven edge, leaving as-is: {', '.join(untouched)}[/]")


def print_proposals(proposals: list[Proposal]) -> None:
    t = Table(title="Proposed trades")
    for col in ["#", "Side", "Coin", "Quantity", "Est. price", "Est. USD", "Now → target value"]:
        t.add_column(col)
    for i, p in enumerate(proposals, 1):
        color = "green" if p.side == "buy" else "red"
        t.add_row(str(i), f"[{color}]{p.side.upper()}[/]", p.symbol, format(p.quantity, "f"),
                  f"${p.price:,.2f}", f"${p.notional:,.2f}", f"${p.current_value:,.0f} → ${p.target_value:,.0f}")
    console.print(t)


def ai_review(proposals: list[Proposal], results: dict) -> None:
    with console.status("AI is checking recent news on these coins..."):
        try:
            ctx = {sym: {k: results[sym][k] for k in ("strategy", "signal", "oos", "buy_hold_oos")}
                   for sym in {p.symbol for p in proposals}}
            verdict = advisor.review([p.to_dict() for p in proposals], ctx)
        except Exception as e:
            verdict = f"AI review unavailable ({type(e).__name__}: {e})."
    console.rule("AI review")
    console.print(verdict)
    console.rule()
    journal("ai_review", text=verdict)


def timed_input(prompt: str, timeout: float | None) -> str:
    """Read a line; with a timeout, give up and return '' when nobody answers."""
    if timeout is None:
        return console.input(prompt).strip().lower()
    console.print(prompt, end="")
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        console.print("[dim](no answer — skipped)[/]")
        return ""
    return sys.stdin.readline().strip().lower()


MAX_PRICE_MOVE = 0.01  # skip an approved order if the price moved more than this since it was proposed


def approve_and_execute(rh: RobinhoodCrypto, proposals: list[Proposal], live: bool,
                        timeout: float | None = None) -> None:
    if not live:
        console.print("[cyan]Dry run: nothing will be sent. Add --live to place approved orders.[/]")
    for i, p in enumerate(proposals, 1):
        console.print(f"\n[bold]#{i}[/] {p.side.upper()} {format(p.quantity, 'f')} {p.symbol} ≈ ${p.notional:,.2f}\n    why: {p.reason}")
        hint = f" (expires in {timeout:.0f}s)" if timeout else ""
        if timed_input(f"    Approve? \\[y/N]{hint} ", timeout) != "y":
            journal("rejected", proposal=p.to_dict())
            continue
        if not live:
            console.print("    [cyan](dry run — not sent)[/]")
            journal("approved_dry_run", proposal=p.to_dict())
            continue
        if KILL_SWITCH.exists():
            console.print("    [red]Kill switch is on — stopping.[/]")
            return
        try:
            q = rh.get_best_bid_ask([p.symbol])[0]
            now = float(q.get("ask_inclusive_of_buy_spread" if p.side == "buy" else "bid_inclusive_of_sell_spread") or q["price"])
            move = abs(now / p.price - 1)
            if move > MAX_PRICE_MOVE:
                console.print(f"    [yellow]Price moved {move:.1%} since the proposal — skipped. It will be re-proposed if still valid.[/]")
                journal("skipped_price_moved", proposal=p.to_dict(), price_now=now)
                continue
            order = rh.place_market_order(p.symbol, p.side, p.quantity)
            console.print(f"    [green]Order sent:[/] id={order.get('id')} state={order.get('state')}")
            journal("order_placed", proposal=p.to_dict(), order=order)
        except (RobinhoodError, IndexError, KeyError) as e:
            console.print(f"    [red]Order failed:[/] {e}")
            journal("order_failed", proposal=p.to_dict(), error=str(e))


# ---------------- commands: suggest / watch ----------------

def cmd_suggest(args, s: Settings) -> None:
    rh = client(s)
    plan = make_plan(rh, s)
    print_plan_details(plan, s)
    proposals = plan["proposals"]
    journal("suggest", equity=plan["pf"]["equity"], cash=plan["pf"]["cash"],
            analysis=plan["results"], proposals=[p.to_dict() for p in proposals])
    if not proposals:
        console.print("[green]No trades needed right now.[/]")
        return
    print_proposals(proposals)
    if not args.no_ai:
        ai_review(proposals, plan["results"])
    approve_and_execute(rh, proposals, args.live)


def notify(title: str, message: str) -> None:
    """Terminal bell + macOS notification (silently skipped elsewhere)."""
    console.bell()
    if sys.platform == "darwin":
        script = f"display notification {json.dumps(message)} with title {json.dumps(title)} sound name \"Glass\""
        subprocess.run(["osascript", "-e", script], capture_output=True, check=False)


def cmd_watch(args, s: Settings) -> None:
    interval = max(1.0, args.interval if args.interval is not None else s.watch_interval_min) * 60
    realert = s.watch_realert_min * 60
    approve_timeout = args.approve_timeout if args.approve_timeout is not None else min(interval, 300)
    rh = client(s)
    console.print(
        f"[bold]Watching[/] {', '.join(s.symbols)} (+ coins you hold) every {interval / 60:g} min · "
        f"{'[red]LIVE[/]' if args.live else '[cyan]dry run[/]'} · Ctrl+C to stop\n"
        f"[dim]Strategy uses {s.timeframe} bars, so signals change when a bar closes; checks in between "
        "catch price drift past your rebalance band. Same trades are re-alerted after "
        f"{s.watch_realert_min:g} min.[/]"
    )
    first = True
    last_key, last_alert, failures = None, 0.0, 0
    while True:
        tick_start = time.time()
        try:
            if KILL_SWITCH.exists():
                console.print(f"[dim]{datetime.now():%H:%M:%S}[/] [red]kill switch on — not checking[/]")
            else:
                plan = make_plan(rh, s, quiet=not first)
                shown_details = first
                if first:
                    print_plan_details(plan, s)
                    first = False
                proposals = plan["proposals"]
                key = tuple(sorted((p.symbol, p.side) for p in proposals))
                stamp = f"[dim]{datetime.now():%H:%M:%S}[/]"
                if not proposals:
                    console.print(f"{stamp} equity ${plan['pf']['equity']:,.2f} — no trades")
                    last_key = None
                elif key == last_key and tick_start - last_alert < realert:
                    console.print(f"{stamp} same {len(proposals)} trade(s) still pending — already shown")
                else:
                    summary = ", ".join(f"{p.side.upper()} {p.symbol} ${p.notional:,.0f}" for p in proposals)
                    console.rule(f"[bold green]Trade opportunity {datetime.now():%H:%M:%S}")
                    notify("trader-bot", summary)
                    journal("watch_alert", equity=plan["pf"]["equity"], proposals=[p.to_dict() for p in proposals])
                    if not shown_details:
                        for n in plan["notes"]:
                            console.print(f"[yellow]{n}[/]")
                    print_proposals(proposals)
                    if not args.no_ai:
                        ai_review(proposals, plan["results"])
                    approve_and_execute(rh, proposals, args.live, timeout=approve_timeout)
                    last_key, last_alert = key, time.time()
            failures = 0
        except KeyboardInterrupt:
            raise
        except Exception as e:  # network blips, API 5xx: keep watching
            failures += 1
            console.print(f"[yellow]{datetime.now():%H:%M:%S} check failed ({type(e).__name__}: {e}); retrying[/]")
            if isinstance(e, RobinhoodError) and e.status == 401:
                console.print("[red]Robinhood rejected the credentials; stopping. Run `trader status` to debug.[/]")
                return
        delay = interval * min(2 ** max(failures - 1, 0), 4) if failures else interval
        time.sleep(max(0.0, delay - (time.time() - tick_start)))


def cmd_orders(args, s: Settings) -> None:
    orders = client(s).get_orders()
    orders.sort(key=lambda o: o.get("created_at", ""), reverse=True)
    t = Table(title="Recent orders")
    for col in ["Created", "Side", "Symbol", "State", "Filled qty", "Avg price"]:
        t.add_column(col)
    for o in orders[: args.limit]:
        t.add_row(o.get("created_at", "")[:19], o.get("side", ""), o.get("symbol", ""), o.get("state", ""),
                  str(o.get("filled_asset_quantity", "")), str(o.get("average_price") or ""))
    console.print(t)


def cmd_kill(args, s: Settings) -> None:
    KILL_SWITCH.write_text(datetime.now(timezone.utc).isoformat())
    console.print("[red]Kill switch ON.[/] No orders will be sent until `trader resume`.")


def cmd_resume(args, s: Settings) -> None:
    KILL_SWITCH.unlink(missing_ok=True)
    console.print("[green]Kill switch OFF.[/]")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="trader", description="Robinhood crypto trading assistant")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keygen", help="create the key pair Robinhood's API needs")
    sub.add_parser("status", help="show your Robinhood crypto holdings")
    b = sub.add_parser("backtest", help="run the strategy tournament (no account needed)")
    b.add_argument("--symbols", help="comma-separated, e.g. BTC-USD,ETH-USD")
    b.add_argument("-v", "--verbose", action="store_true")
    g = sub.add_parser("suggest", help="propose trades and ask you to approve each")
    g.add_argument("--live", action="store_true", help="actually send approved orders")
    g.add_argument("--no-ai", action="store_true", help="skip the AI news review")
    w = sub.add_parser("watch", help="keep checking and alert you when there's a trade")
    w.add_argument("--interval", type=float, help="minutes between checks (default WATCH_INTERVAL_MIN, 5)")
    w.add_argument("--live", action="store_true", help="actually send approved orders")
    w.add_argument("--no-ai", action="store_true", help="skip the AI news review")
    w.add_argument("--approve-timeout", type=float, help="seconds to answer before a proposal expires")
    o = sub.add_parser("orders", help="list recent orders")
    o.add_argument("--limit", type=int, default=20)
    sub.add_parser("kill", help="block all orders")
    sub.add_parser("resume", help="allow orders again")

    args = ap.parse_args(argv)
    s = load_settings()
    cmds = {"keygen": cmd_keygen, "status": cmd_status, "backtest": cmd_backtest, "suggest": cmd_suggest, "watch": cmd_watch,
            "orders": cmd_orders, "kill": cmd_kill, "resume": cmd_resume}
    try:
        cmds[args.cmd](args, s)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/]")
    except (ValueError, RobinhoodError) as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
