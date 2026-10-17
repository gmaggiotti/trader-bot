"""Optional AI review of proposed trades, using an NVIDIA-hosted model.

The bot fetches the last week of headlines for each coin and the model flags
anything the price-based strategy can't see (hacks, delistings, regulation,
token unlocks). Advisory only: it never changes sizes or places orders.
"""

import json
import os
import re

from openai import OpenAI

from . import news

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"

SYSTEM = """You review proposed crypto trades for a single retail trader before they approve them.
The trades come from a rule-based trend strategy tested out-of-sample; your job is NOT to forecast
prices. Your job is to catch things price data can't show: exchange or protocol hacks, delistings,
regulatory actions, large token unlocks, network outages, or any reason a trade looks like a mistake.

You are given recent headlines for each coin. Base your judgment only on them. Reply with one block
per trade, in this exact format:

SYMBOL SIDE: PROCEED | CAUTION | AVOID
- one or two sentences on why, naming the specific headline if any

Finish with one line, "Overall:", summarizing. Be brief and concrete. If nothing notable is in the
headlines, say so plainly rather than inventing concerns."""


def review(proposals: list[dict], context: dict) -> str:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        return "AI review skipped: set NVIDIA_API_KEY in .env (or use --no-ai)."
    model = os.getenv("NVIDIA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    heads = {}
    for sym in sorted({p["symbol"] for p in proposals}):
        try:
            heads[sym] = news.headlines(sym)
        except Exception as e:
            heads[sym] = [{"date": "", "title": f"(could not fetch news: {e})"}]

    prompt = (
        "Proposed trades:\n" + json.dumps(proposals, indent=2, default=str)
        + "\n\nStrategy context (out-of-sample stats per coin):\n" + json.dumps(context, indent=2, default=str)
        + "\n\nHeadlines from the last 7 days:\n" + json.dumps(heads, indent=2)
    )
    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=4096,
    )
    text = resp.choices[0].message.content or ""
    # Reasoning models may include their scratch work in <think> tags; drop it.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text or "AI review returned no text."
