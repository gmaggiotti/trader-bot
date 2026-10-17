# trader-bot

A crypto trading assistant for your Robinhood account. It reads your holdings, decides how much of each coin to hold using a backtested trend strategy, asks an AI model (NVIDIA Nemotron) to check recent news, and **asks you to approve every order**.

## How it decides

1. **Price history** comes from Coinbase's free public API. Robinhood's API has no history.
2. **Six strategy families** (moving-average cross, EMA trend, Donchian breakout, RSI mean-reversion, volatility-sized momentum, buy & hold), each tested over a grid of settings.
3. **Walk-forward testing**: every result shown is *out-of-sample*, measured on data the strategy never saw.
4. **The live strategy is a fixed blend of 18 trend signals** with nothing fitted to past data. Testing showed that switching to whichever strategy did best recently ("Picker" in the report) chases past winners and does worse. The blend held up better.
5. **A coin is traded only if the blend has an out-of-sample Sharpe ≥ `MIN_OOS_SHARPE`** over at least one year. Otherwise the bot leaves that coin alone.
6. **Risk manager**: per-coin cap, total crypto cap (counts coins the bot doesn't manage), max order size, daily-loss buy block, and a kill switch.
7. **AI review** (optional): the bot pulls the last week of Google News headlines for each coin, and `nvidia/nemotron-3.5-super-vl-120b-a12b` (via NVIDIA's API) marks each trade PROCEED / CAUTION / AVOID. It only advises; it never changes or places orders.

**What to expect.** Over the last ~3.5 years the strategy earned *less* than buy & hold but cut the worst drops roughly in half (BTC: -31% vs -53%; ETH: -37% vs -68%). It sells to cash in downtrends because Robinhood crypto can't be shorted. No strategy guarantees profit.

## Setup

```bash
uv sync
cp .env.example .env
uv run trader keygen          # prints a key pair + instructions
```

In Robinhood (web), open Account → Crypto → API trading. Create a key using the **public** key, then put the API key Robinhood gives you and your **private** key into `.env`.

For the AI review, get a free key at https://build.nvidia.com and set `NVIDIA_API_KEY` in `.env`, or pass `--no-ai` to skip it.

## Use

```bash
uv run trader backtest -v     # strategy report, no account needed
uv run trader status          # your holdings
uv run trader suggest         # propose trades, approve each (dry run: nothing is sent)
uv run trader suggest --live  # same, but approved orders are actually placed
uv run trader watch           # keep checking every 5 min; alert + ask when there's a trade
uv run trader watch --interval 1 --live   # check every minute, send approved orders
uv run trader orders          # recent orders
uv run trader kill / resume   # block / unblock all orders
```

**Watch mode** checks every `WATCH_INTERVAL_MIN` minutes (or `--interval`). When there's a trade it beeps, shows a macOS notification, and asks you to approve each order. Unanswered approvals expire (`--approve-timeout`, default up to 5 min) so you never approve a stale price. Approved orders are re-priced first and skipped if the price moved more than 1%. The same pending trades are re-alerted only after `WATCH_REALERT_MIN` minutes. With daily bars, new signals mostly appear once a day, at 00:00 UTC; checks in between catch prices drifting past your rebalance band. Every suggestion, approval and order is logged to `data/journal.jsonl`.

## Tune

All limits live in `.env` (see `.env.example`). Start small: set `MAX_ORDER_USD` low until you trust it.

## Tests

```bash
uv run pytest
```
