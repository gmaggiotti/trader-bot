"""Recent headlines per coin from Google News RSS (free, no key)."""

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

FEED = "https://news.google.com/rss/search?q={q}+when:7d&hl=en-US&gl=US&ceid=US:en"

NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "DOGE": "Dogecoin", "XRP": "XRP",
    "ADA": "Cardano", "AVAX": "Avalanche", "LINK": "Chainlink", "DOT": "Polkadot", "LTC": "Litecoin",
    "SHIB": "Shiba Inu", "BCH": "Bitcoin Cash", "UNI": "Uniswap", "XLM": "Stellar", "PEPE": "Pepe",
}


def headlines(symbol: str, limit: int = 15) -> list[dict]:
    code = symbol.split("-")[0]
    query = f"{NAMES.get(code, code)} crypto"
    r = requests.get(FEED.format(q=quote_plus(query)), timeout=15)
    r.raise_for_status()
    items = ET.fromstring(r.content).findall("./channel/item")
    out = []
    for it in items:
        try:
            when = parsedate_to_datetime(it.findtext("pubDate", "")).date().isoformat()
        except (TypeError, ValueError):
            when = ""
        out.append({"date": when, "title": it.findtext("title", "")})
    out.sort(key=lambda h: h["date"], reverse=True)
    return out[:limit]
