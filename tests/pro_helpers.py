"""
Stand-ins for the Pro tests: a fake PayPal with the client's methods and
enough memory to answer a GET truthfully, a fake Agent Toolkit bridge, a
scripted model for the agent loop, and the app fixture that wires them in.
No network anywhere.
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from server.services.paypal_client import PayPalError


class FakePayPal:
    """Sandbox PayPal in memory: orders → authorizations → captures/voids/refunds, plus invoices."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.auths: Dict[str, Dict[str, Any]] = {}
        self.captures: Dict[str, Dict[str, Any]] = {}
        self.refunds: Dict[str, Dict[str, Any]] = {}
        self.invoices: Dict[str, Dict[str, Any]] = {}
        self._counter = itertools.count(1)
        # Fault injection: how many capture calls fail (retryable) before one succeeds; a hard decline.
        self.capture_failures = 0
        self.capture_declined = False
        self.void_fails = False
        self.refund_fails = False
        self.webhook_ok = True
        self.request_ids: List[Optional[str]] = []

    def _next(self, prefix: str) -> str:
        return f"{prefix}{next(self._counter):03d}"

    def _log(self, method: str, request_id: Optional[str], **kwargs: Any) -> None:
        self.calls.append({"method": method, "request_id": request_id, **kwargs})
        self.request_ids.append(request_id)

    # Orders v2
    def create_order(self, *, amount_cents: int, custom_id: str, description: str, request_id: Optional[str] = None, currency: str = "USD") -> Dict[str, Any]:
        self._log("create_order", request_id, amount_cents=amount_cents, custom_id=custom_id)
        order_id = self._next("ORDER")
        self.orders[order_id] = {"id": order_id, "status": "CREATED", "custom_id": custom_id, "amount_cents": amount_cents, "auth": None}
        return {"id": order_id, "status": "CREATED"}

    def authorize_order(self, order_id: str, *, request_id: Optional[str] = None) -> Dict[str, Any]:
        self._log("authorize_order", request_id, order_id=order_id)
        order = self.orders.get(order_id)
        if order is None:
            raise PayPalError(404, '{"name":"RESOURCE_NOT_FOUND"}', name="RESOURCE_NOT_FOUND")
        if order["auth"]:
            raise PayPalError(422, '{"name":"UNPROCESSABLE_ENTITY","details":[{"issue":"ORDER_ALREADY_AUTHORIZED"}]}', name="UNPROCESSABLE_ENTITY")
        auth_id = self._next("AUTH")
        expires = (datetime.now(timezone.utc) + timedelta(days=29)).isoformat().replace("+00:00", "Z")
        self.auths[auth_id] = {"id": auth_id, "status": "CREATED", "order_id": order_id, "expiration_time": expires, "amount_cents": order["amount_cents"]}
        order["auth"] = auth_id
        order["status"] = "COMPLETED"
        return {"id": order_id, "status": "COMPLETED", "purchase_units": [{"payments": {"authorizations": [dict(self.auths[auth_id])]}}]}

    def get_order(self, order_id: str) -> Dict[str, Any]:
        self._log("get_order", None, order_id=order_id)
        order = self.orders.get(order_id)
        if order is None:
            raise PayPalError(404, "not found")
        payments: Dict[str, Any] = {}
        if order["auth"]:
            auth = self.auths[order["auth"]]
            payments["authorizations"] = [{"id": auth["id"], "status": auth["status"]}]
            payments["captures"] = [dict(c) for c in self.captures.values() if c["auth_id"] == auth["id"]]
            payments["refunds"] = [dict(r) for r in self.refunds.values() if r["auth_id"] == auth["id"]]
        return {"id": order_id, "status": order["status"], "purchase_units": [{"custom_id": order["custom_id"], "payments": payments}]}

    # Payments v2
    def capture_authorization(self, authorization_id: str, *, request_id: Optional[str] = None, final: bool = True) -> Dict[str, Any]:
        self._log("capture_authorization", request_id, authorization_id=authorization_id)
        auth = self.auths.get(authorization_id)
        if auth is None:
            raise PayPalError(404, "no such authorization")
        if self.capture_failures > 0:
            self.capture_failures -= 1
            raise PayPalError(503, '{"name":"SERVICE_UNAVAILABLE"}', name="SERVICE_UNAVAILABLE")
        if auth["status"] == "VOIDED":
            raise PayPalError(422, '{"name":"UNPROCESSABLE_ENTITY","details":[{"issue":"AUTHORIZATION_VOIDED"}]}')
        if auth["status"] == "CAPTURED":
            raise PayPalError(422, '{"name":"UNPROCESSABLE_ENTITY","details":[{"issue":"AUTHORIZATION_ALREADY_CAPTURED"}]}')
        capture_id = self._next("CAP")
        status = "DECLINED" if self.capture_declined else "COMPLETED"
        self.captures[capture_id] = {"id": capture_id, "status": status, "auth_id": authorization_id, "amount_cents": auth["amount_cents"]}
        if status == "COMPLETED":
            auth["status"] = "CAPTURED"
        return {"id": capture_id, "status": status}

    def void_authorization(self, authorization_id: str, *, request_id: Optional[str] = None) -> Dict[str, Any]:
        self._log("void_authorization", request_id, authorization_id=authorization_id)
        auth = self.auths.get(authorization_id)
        if auth is None:
            raise PayPalError(404, "no such authorization")
        if self.void_fails:
            raise PayPalError(503, "down")
        if auth["status"] == "CAPTURED":
            raise PayPalError(422, '{"name":"UNPROCESSABLE_ENTITY","details":[{"issue":"AUTHORIZATION_ALREADY_CAPTURED"}]}')
        auth["status"] = "VOIDED"
        return {"status": "VOIDED"}

    def get_authorization(self, authorization_id: str) -> Dict[str, Any]:
        auth = self.auths.get(authorization_id)
        if auth is None:
            raise PayPalError(404, "no such authorization")
        return dict(auth)

    def refund_capture(self, capture_id: str, *, request_id: Optional[str] = None, amount_cents: Optional[int] = None, currency: str = "USD", note: str = "") -> Dict[str, Any]:
        self._log("refund_capture", request_id, capture_id=capture_id, amount_cents=amount_cents)
        capture = self.captures.get(capture_id)
        if capture is None:
            raise PayPalError(404, "no such capture")
        if self.refund_fails:
            raise PayPalError(503, "down")
        if amount_cents is not None and amount_cents > capture["amount_cents"]:
            raise PayPalError(422, "REFUND_AMOUNT_EXCEEDED")
        refund_id = self._next("REF")
        self.refunds[refund_id] = {"id": refund_id, "status": "COMPLETED", "auth_id": capture["auth_id"], "amount_cents": amount_cents or capture["amount_cents"]}
        capture["status"] = "REFUNDED"
        return {"id": refund_id, "status": "COMPLETED"}

    # Webhooks
    def verify_webhook_signature(self, *, webhook_id: str, headers: Dict[str, str], event: Dict[str, Any]) -> bool:
        self._log("verify_webhook_signature", None, webhook_id=webhook_id)
        return self.webhook_ok

    # Invoicing v2 (the built-in fallback path)
    def create_invoice(self, payload: Dict[str, Any], *, request_id: Optional[str] = None) -> Dict[str, Any]:
        self._log("create_invoice", request_id, payload=payload)
        invoice_id = f"INV2-{self._next('')}"
        self.invoices[invoice_id] = {"id": invoice_id, "status": "DRAFT", "detail": payload.get("detail"), "items": payload.get("items")}
        return {"rel": "self", "href": f"https://api-m.sandbox.paypal.com/v2/invoicing/invoices/{invoice_id}", "method": "GET"}

    def send_invoice(self, invoice_id: str, *, note: str = "", request_id: Optional[str] = None) -> Dict[str, Any]:
        self._log("send_invoice", request_id, invoice_id=invoice_id)
        self.invoices[invoice_id]["status"] = "SENT"
        return {"href": f"https://www.sandbox.paypal.com/invoice/p/#{invoice_id}"}

    def list_invoices(self, *, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        return {"items": list(self.invoices.values())[:page_size], "total_items": len(self.invoices)}

    def get_invoice(self, invoice_id: str) -> Dict[str, Any]:
        if invoice_id not in self.invoices:
            raise PayPalError(404, "no such invoice")
        return dict(self.invoices[invoice_id])

    def send_invoice_reminder(self, invoice_id: str, *, note: str = "", request_id: Optional[str] = None) -> Dict[str, Any]:
        self._log("send_invoice_reminder", request_id, invoice_id=invoice_id)
        return {}

    def cancel_sent_invoice(self, invoice_id: str, *, note: str = "", request_id: Optional[str] = None) -> Dict[str, Any]:
        self._log("cancel_sent_invoice", request_id, invoice_id=invoice_id)
        self.invoices.get(invoice_id, {})["status"] = "CANCELLED"
        return {}


def paypal_http_error(status: int, body: Dict[str, Any]):
    """
    What the real toolkit raises on a PayPal 4xx/5xx: a requests.HTTPError
    carrying the response. The app must turn it into a sentence, never a 500.
    """
    import requests

    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return requests.exceptions.HTTPError(f"{status} Client Error", response=response)


class FakeToolkit:
    """
    The Agent Toolkit bridge with the toolkit's shape but a fake PayPal
    behind it. `fail(method, params)` may return an exception to raise, the
    way the real toolkit lets requests/pydantic errors out.
    """

    def __init__(self, paypal: FakePayPal, *, available: bool = True) -> None:
        from server.services.ops_agent import FALLBACK_DESCRIPTIONS, FALLBACK_SCHEMAS

        self.paypal = paypal
        self.available = available
        self.label = "PayPal Agent Toolkit (fake)" if available else "built-in Invoicing v2 client"
        self._schemas = FALLBACK_SCHEMAS
        self._descriptions = FALLBACK_DESCRIPTIONS
        self.runs: List[Dict[str, Any]] = []
        self.fail = None  # Optional[Callable[[str, dict], Optional[BaseException]]]

    def schema(self, method: str) -> Dict[str, Any]:
        return self._schemas.get(method, {"type": "object", "properties": {}})

    def description(self, method: str) -> str:
        return self._descriptions.get(method, method)

    def run(self, method: str, params: Dict[str, Any]) -> Any:
        from server.services.ops_agent import TOOLKIT_ALLOWED, AgentError
        from server.services.paypal_client import PayPalError

        self.runs.append({"method": method, "params": params})
        if method not in TOOLKIT_ALLOWED:
            raise AgentError(f"{method} is not available to this agent.", status=403)
        if self.fail is not None:
            exc = self.fail(method, params)
            if exc is not None:
                # The same translation the real bridge does for the real toolkit.
                import requests

                if isinstance(exc, requests.exceptions.RequestException):
                    raise PayPalError.from_requests_error(exc) from exc
                raise exc
        if method == "create_invoice":
            payload = {"detail": {"currency_code": params["currency_code"], "note": params.get("note")}, "items": params["items"]}
            return self.paypal.create_invoice(payload)
        if method == "send_invoice":
            return self.paypal.send_invoice(params["invoice_id"])
        if method == "list_invoices":
            return self.paypal.list_invoices()
        if method == "get_invoice":
            return self.paypal.get_invoice(params["invoice_id"])
        if method == "get_order_details":
            return self.paypal.get_order(params["order_id"])
        if method == "send_invoice_reminder":
            return self.paypal.send_invoice_reminder(params["invoice_id"])
        if method == "cancel_sent_invoice":
            return self.paypal.cancel_sent_invoice(params["invoice_id"])
        raise AgentError("unexpected")


class ScriptedModel:
    """
    A NebiusChatClient stand-in for the agent loop: `turns` is the list of
    assistant messages to return, in order. Records what it was asked.
    """

    def __init__(self, turns: List[Dict[str, Any]]) -> None:
        self.turns = list(turns)
        self.requests: List[Dict[str, Any]] = []
        self.meter = None

    async def create_tool_completion(self, messages, *, tools, model=None, max_tokens=800, tool_choice="auto"):
        self.requests.append({"messages": list(messages), "tools": tools, "model": model})
        if not self.turns:
            return {"role": "assistant", "content": "Done."}
        return self.turns.pop(0)


def tool_call(name: str, args: Dict[str, Any], call_id: str = "call_1") -> Dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def final(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": text}


@pytest.fixture
def paypal():
    return FakePayPal()


@pytest.fixture
def pro_app(tmp_path, monkeypatch, paypal):
    """
    The app with Pro on: fake sandbox keys, a SQLite file under tmp_path,
    the fake PayPal and toolkit installed, fresh budgets, the committed
    sample available, and no model key (drafts use the template).
    """
    from fastapi.testclient import TestClient

    from server.services import ops_agent, staff_auth
    from server.services.budgets import reset_budgets_for_tests
    from server.services.db import reset_db_for_tests
    from server.services.demo_guard import reset_guard
    from server.services.jobs import JobStore, reset_store
    from server.services.paypal_client import reset_paypal_for_tests

    monkeypatch.setenv("PAYPAL_CLIENT_ID", "sb-client-id")
    monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "sb-secret")
    monkeypatch.setenv("PAYPAL_ENV", "sandbox")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PRO_DB_PATH", str(tmp_path / "pro.sqlite"))
    monkeypatch.setenv("DEMO_SAMPLE_DIR", "demo_sample")
    monkeypatch.delenv("SAMPLE_CLIP_URL", raising=False)
    monkeypatch.delenv("SAMPLE_RESULT_URL", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setenv("COACH_PASSWORD", STAFF_PASSWORD)
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.delenv("STAFF_LOGIN_HOURLY_LIMIT", raising=False)
    monkeypatch.delenv("PRO_DEMO_MODE", raising=False)
    monkeypatch.delenv("DYNO", raising=False)
    for name in ("DEMO_ACCESS_CODE", "ACCESS_MODE", "DEMO_DISABLED", "DEMO_DAILY_LIMIT", "DEMO_IP_HOURLY_LIMIT", "MAX_CONCURRENT_JOBS", "TRUSTED_PROXIES"):
        monkeypatch.delenv(name, raising=False)
    reset_db_for_tests()
    reset_paypal_for_tests(paypal)
    ops_agent.reset_toolkit_for_tests(FakeToolkit(paypal))
    reset_budgets_for_tests()
    staff_auth.reset_login_attempts_for_tests()
    reset_store(JobStore(root=tmp_path / "jobs"))
    reset_guard()
    from server.main import app

    for name in ("DEMO_ACCESS_CODE", "ACCESS_MODE", "DEMO_DISABLED"):
        monkeypatch.delenv(name, raising=False)
    with TestClient(app) as client:
        # The coach and ops routes need the sign-in; the tests start signed
        # in and sign out explicitly where that is the point.
        staff_login(client)
        yield client
    reset_paypal_for_tests(None)
    ops_agent.reset_toolkit_for_tests(None)
    reset_budgets_for_tests()
    staff_auth.reset_login_attempts_for_tests()
    reset_db_for_tests()
    reset_store(JobStore(root=tmp_path / "jobs-after"))
    reset_guard()


STAFF_PASSWORD = "test-pass"


def staff_login(client, password: str = STAFF_PASSWORD):
    response = client.post("/api/auth/login", json={"password": password})
    assert response.status_code == 200, response.text
    return response


def staff_logout(client) -> None:
    assert client.post("/api/auth/logout").status_code == 200
    client.cookies.clear()


def buy_sample(client, *, name: str = "Dan", token: str = "tok-dan", note: str = "") -> Dict[str, Any]:
    """Tap "Get a coach's eyes" on the sample and approve it: a held review."""
    created = client.post(
        "/api/checkout/reviews",
        json={"source": "sample", "athlete_token": token, "athlete_name": name, "note": note},
    )
    assert created.status_code == 201, created.text
    review_id = created.json()["review_id"]
    approved = client.post(f"/api/checkout/reviews/{review_id}/approve")
    assert approved.status_code == 200, approved.text
    return approved.json()
