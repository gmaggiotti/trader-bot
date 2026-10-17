"""Client for Robinhood's official Crypto Trading API.

Every request is signed with Ed25519 over: api_key + timestamp + path + method + body.
Signed requests are only valid for ~30 seconds, so the system clock must be accurate.
"""

import base64
import json
import time
import uuid
from decimal import Decimal
from urllib.parse import urlencode, urlsplit

import requests
from nacl.signing import SigningKey

BASE_URL = "https://trading.robinhood.com"


class RobinhoodError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Robinhood API error {status}: {body[:500]}")
        self.status = status
        self.body = body


def generate_keypair() -> tuple[str, str]:
    """Return (private_key_b64, public_key_b64) for registering with Robinhood."""
    key = SigningKey.generate()
    return (
        base64.b64encode(bytes(key)).decode(),
        base64.b64encode(bytes(key.verify_key)).decode(),
    )


class RobinhoodCrypto:
    def __init__(self, api_key: str, private_key_b64: str, base_url: str = BASE_URL, timeout: float = 15):
        if not api_key or not private_key_b64:
            raise ValueError("RH_API_KEY and RH_PRIVATE_KEY must be set (see .env.example)")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._key = SigningKey(base64.b64decode(private_key_b64))
        self._http = requests.Session()

    # ---- signing / transport ----

    def sign(self, method: str, path: str, body: str, timestamp: int) -> dict[str, str]:
        message = f"{self.api_key}{timestamp}{path}{method}{body}"
        signature = self._key.sign(message.encode("utf-8")).signature
        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
            "Content-Type": "application/json; charset=utf-8",
        }

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        body_str = json.dumps(body) if body is not None else ""
        for attempt in range(4):
            headers = self.sign(method, path, body_str, int(time.time()))
            resp = self._http.request(
                method, self.base_url + path, headers=headers, data=body_str or None, timeout=self.timeout
            )
            if resp.status_code == 429 and attempt < 3:
                time.sleep(2**attempt)
                continue
            if not resp.ok:
                raise RobinhoodError(resp.status_code, resp.text)
            return resp.json() if resp.text else {}
        raise RobinhoodError(429, "rate limited")

    def _get_all(self, path: str) -> list[dict]:
        results: list[dict] = []
        while path:
            data = self._request("GET", path)
            results.extend(data.get("results", []))
            nxt = data.get("next")
            if not nxt:
                break
            parts = urlsplit(nxt)
            path = parts.path + (f"?{parts.query}" if parts.query else "")
        return results

    @staticmethod
    def _query(**params) -> str:
        params = {k: v for k, v in params.items() if v not in (None, [], "")}
        return f"?{urlencode(params, doseq=True)}" if params else ""

    # ---- account ----

    def get_account(self) -> dict:
        return self._request("GET", "/api/v1/crypto/trading/accounts/")

    def get_holdings(self, asset_codes: list[str] | None = None) -> list[dict]:
        return self._get_all("/api/v1/crypto/trading/holdings/" + self._query(asset_code=asset_codes))

    # ---- market data ----

    def get_best_bid_ask(self, symbols: list[str]) -> list[dict]:
        return self._get_all("/api/v1/crypto/marketdata/best_bid_ask/" + self._query(symbol=symbols))

    def get_estimated_price(self, symbol: str, side: str, quantity: Decimal) -> list[dict]:
        path = "/api/v1/crypto/marketdata/estimated_price/" + self._query(
            symbol=symbol, side=side, quantity=str(quantity)
        )
        return self._request("GET", path).get("results", [])

    def get_trading_pairs(self, symbols: list[str] | None = None) -> list[dict]:
        return self._get_all("/api/v1/crypto/trading/trading_pairs/" + self._query(symbol=symbols))

    # ---- orders ----

    def get_orders(self, **filters) -> list[dict]:
        return self._get_all("/api/v1/crypto/trading/orders/" + self._query(**filters))

    def place_market_order(self, symbol: str, side: str, asset_quantity: Decimal) -> dict:
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        body = {
            "client_order_id": str(uuid.uuid4()),
            "side": side,
            "symbol": symbol,
            "type": "market",
            "market_order_config": {"asset_quantity": format(asset_quantity, "f")},
        }
        return self._request("POST", "/api/v1/crypto/trading/orders/", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._request("POST", f"/api/v1/crypto/trading/orders/{order_id}/cancel/")
