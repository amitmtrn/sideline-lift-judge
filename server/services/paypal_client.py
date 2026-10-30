"""
The PayPal REST client: the only place a PayPal URL is spelled.

Orders v2 creates the order with intent AUTHORIZE and authorizes it after the
buyer approves; Payments v2 captures, voids or refunds the result. Every
write carries a PayPal-Request-Id so a retried request cannot double-charge
or double-void. The OAuth token is fetched with the client credentials and
kept until shortly before it expires.

Sandbox only: pro_config.assert_sandbox() runs before any request, so a
PAYPAL_ENV of anything but `sandbox` fails before money is touched.

The invoice helpers at the bottom are the fallback the ops agent uses when
`paypal_agent_toolkit` is not installed (its pins do not resolve on Python
3.13 without --no-deps); they hit the same Invoicing v2 endpoints.
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import requests

from server.services import pro_config
from server.services.db import new_id

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 25.0  # inside Heroku's 30 s router window
TOKEN_SAFETY_SECONDS = 60.0


class PayPalError(RuntimeError):
    """A non-2xx reply (or no reply) from PayPal, with what the state machine needs to decide."""

    def __init__(self, status: Optional[int], body: str, *, name: str = "") -> None:
        super().__init__(f"PayPal error {status}: {body[:300]}")
        self.status = status
        self.body = body
        self.name = name

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status in (429, 500, 502, 503, 504)


def _excerpt(payload: Any) -> str:
    if payload is None:
        return ""
    text = payload if isinstance(payload, str) else str(payload)
    return text[:2000]


def money(cents: int, currency: str = "USD") -> Dict[str, str]:
    return {"currency_code": currency, "value": f"{cents / 100:.2f}"}


class PayPalClient:
    """
    Thin, synchronous, injectable. Routes call it through asyncio.to_thread;
    the tests hand in a fake with the same methods (see reset_paypal_for_tests).
    """

    def __init__(
        self,
        client_id: str,
        secret: str,
        *,
        base_url: Optional[str] = None,
        http: Any = requests,
    ) -> None:
        if not client_id or not secret:
            raise pro_config.ProConfigError("PayPal needs PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET")
        self.client_id = client_id
        self.secret = secret
        self.base_url = (base_url or pro_config.paypal_base_url()).rstrip("/")
        self._http = http
        self._token: Optional[str] = None
        self._token_expires_at = 0.0
        self._token_lock = threading.Lock()

    # ----- auth -----

    def access_token(self) -> str:
        with self._token_lock:
            if self._token and time.time() < self._token_expires_at - TOKEN_SAFETY_SECONDS:
                return self._token
            basic = base64.b64encode(f"{self.client_id}:{self.secret}".encode("utf-8")).decode("ascii")
            try:
                response = self._http.post(
                    f"{self.base_url}/v1/oauth2/token",
                    headers={"Authorization": f"Basic {basic}", "Accept": "application/json"},
                    data={"grant_type": "client_credentials"},
                    timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                )
            except requests.exceptions.RequestException as exc:
                raise PayPalError(None, f"token request failed: {exc}") from exc
            if response.status_code != 200:
                raise PayPalError(response.status_code, _excerpt(response.text), name="TOKEN")
            data = response.json()
            self._token = str(data.get("access_token") or "")
            if not self._token:
                raise PayPalError(response.status_code, "no access_token in reply", name="TOKEN")
            self._token_expires_at = time.time() + float(data.get("expires_in") or 300)
            return self._token

    # ----- transport -----

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        request_id: Optional[str] = None,
        representation: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        pro_config.assert_sandbox()
        headers = {
            "Authorization": f"Bearer {self.access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if request_id:
            headers["PayPal-Request-Id"] = request_id
        if representation:
            headers["Prefer"] = "return=representation"
        try:
            response = self._http.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                json=json,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            )
        except requests.exceptions.RequestException as exc:
            raise PayPalError(None, f"{method} {path} failed: {exc}") from exc
        status = response.status_code
        body: Dict[str, Any] = {}
        text = response.text or ""
        if text.strip():
            try:
                parsed = response.json()
                body = parsed if isinstance(parsed, dict) else {"data": parsed}
            except ValueError:
                body = {"raw": text[:2000]}
        if status >= 400:
            raise PayPalError(status, _excerpt(text), name=str(body.get("name") or ""))
        return status, body

    # ----- Orders v2 -----

    def create_order(
        self,
        *,
        amount_cents: int,
        custom_id: str,
        description: str,
        request_id: Optional[str] = None,
        currency: str = "USD",
    ) -> Dict[str, Any]:
        payload = {
            "intent": "AUTHORIZE",
            "purchase_units": [
                {
                    "custom_id": custom_id,
                    "description": description[:127],
                    "amount": money(amount_cents, currency),
                }
            ],
            "payment_source": {
                "paypal": {
                    "experience_context": {
                        "brand_name": "Sideline Pro (sandbox)",
                        "user_action": "PAY_NOW",
                        "shipping_preference": "NO_SHIPPING",
                    }
                }
            },
        }
        _, body = self._request("POST", "/v2/checkout/orders", json=payload, request_id=request_id or new_id())
        return body

    def authorize_order(self, order_id: str, *, request_id: Optional[str] = None) -> Dict[str, Any]:
        _, body = self._request(
            "POST", f"/v2/checkout/orders/{order_id}/authorize", json={}, request_id=request_id or new_id()
        )
        return body

    def get_order(self, order_id: str) -> Dict[str, Any]:
        _, body = self._request("GET", f"/v2/checkout/orders/{order_id}")
        return body

    # ----- Payments v2 -----

    def capture_authorization(
        self, authorization_id: str, *, request_id: Optional[str] = None, final: bool = True
    ) -> Dict[str, Any]:
        _, body = self._request(
            "POST",
            f"/v2/payments/authorizations/{authorization_id}/capture",
            json={"final_capture": bool(final)},
            request_id=request_id or new_id(),
            representation=True,
        )
        return body

    def void_authorization(self, authorization_id: str, *, request_id: Optional[str] = None) -> Dict[str, Any]:
        status, body = self._request(
            "POST",
            f"/v2/payments/authorizations/{authorization_id}/void",
            json={},
            request_id=request_id or new_id(),
            representation=True,
        )
        body.setdefault("status", "VOIDED" if status in (200, 204) else body.get("status"))
        return body

    def get_authorization(self, authorization_id: str) -> Dict[str, Any]:
        _, body = self._request("GET", f"/v2/payments/authorizations/{authorization_id}")
        return body

    def refund_capture(
        self,
        capture_id: str,
        *,
        request_id: Optional[str] = None,
        amount_cents: Optional[int] = None,
        currency: str = "USD",
        note: str = "",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if amount_cents is not None:
            payload["amount"] = money(amount_cents, currency)
        if note:
            payload["note_to_payer"] = note[:255]
        _, body = self._request(
            "POST",
            f"/v2/payments/captures/{capture_id}/refund",
            json=payload,
            request_id=request_id or new_id(),
            representation=True,
        )
        return body

    # ----- Webhooks (P1) -----

    def verify_webhook_signature(self, *, webhook_id: str, headers: Dict[str, str], event: Dict[str, Any]) -> bool:
        def header(name: str) -> str:
            for key, value in headers.items():
                if key.lower() == name.lower():
                    return value
            return ""

        payload = {
            "auth_algo": header("paypal-auth-algo"),
            "cert_url": header("paypal-cert-url"),
            "transmission_id": header("paypal-transmission-id"),
            "transmission_sig": header("paypal-transmission-sig"),
            "transmission_time": header("paypal-transmission-time"),
            "webhook_id": webhook_id,
            "webhook_event": event,
        }
        _, body = self._request("POST", "/v1/notifications/verify-webhook-signature", json=payload)
        return str(body.get("verification_status") or "").upper() == "SUCCESS"

    # ----- Invoicing v2 (fallback for the Agent Toolkit) -----

    def create_invoice(self, payload: Dict[str, Any], *, request_id: Optional[str] = None) -> Dict[str, Any]:
        _, body = self._request("POST", "/v2/invoicing/invoices", json=payload, request_id=request_id or new_id())
        return body

    def send_invoice(self, invoice_id: str, *, note: str = "", request_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"send_to_recipient": True}
        if note:
            payload["note"] = note[:4000]
        _, body = self._request(
            "POST", f"/v2/invoicing/invoices/{invoice_id}/send", json=payload, request_id=request_id or new_id()
        )
        return body

    def list_invoices(self, *, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        _, body = self._request(
            "GET", f"/v2/invoicing/invoices?page={int(page)}&page_size={int(page_size)}&total_required=true"
        )
        return body

    def get_invoice(self, invoice_id: str) -> Dict[str, Any]:
        _, body = self._request("GET", f"/v2/invoicing/invoices/{invoice_id}")
        return body

    def send_invoice_reminder(self, invoice_id: str, *, note: str = "", request_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"send_to_recipient": True}
        if note:
            payload["note"] = note[:4000]
        _, body = self._request(
            "POST", f"/v2/invoicing/invoices/{invoice_id}/remind", json=payload, request_id=request_id or new_id()
        )
        return body

    def cancel_sent_invoice(self, invoice_id: str, *, note: str = "", request_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"send_to_recipient": True}
        if note:
            payload["note"] = note[:4000]
        _, body = self._request(
            "POST", f"/v2/invoicing/invoices/{invoice_id}/cancel", json=payload, request_id=request_id or new_id()
        )
        return body


# ---------- the process-wide client ----------

_client: Optional[PayPalClient] = None
_client_key: Optional[Tuple[str, str]] = None
_fake: Optional[Any] = None
_client_lock = threading.Lock()


def get_paypal() -> PayPalClient:
    """The client for the configured credentials, rebuilt when they change — or the tests' stand-in."""
    global _client, _client_key
    with _client_lock:
        if _fake is not None:
            return _fake
        key = (pro_config.paypal_client_id(), pro_config.paypal_client_secret())
        if _client is None or _client_key != key:
            _client = PayPalClient(key[0], key[1])
            _client_key = key
        return _client


def reset_paypal_for_tests(client: Optional[Any] = None) -> None:
    """Install a stand-in (any object with the client's methods), or go back to the real one."""
    global _client, _client_key, _fake
    with _client_lock:
        _fake = client
        _client = None
        _client_key = None


def first_authorization(order: Dict[str, Any]) -> Dict[str, Any]:
    """The authorization object out of an authorized order's reply."""
    try:
        return dict(order["purchase_units"][0]["payments"]["authorizations"][0])
    except (KeyError, IndexError, TypeError):
        return {}
