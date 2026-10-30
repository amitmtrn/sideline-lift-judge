"""
What every Pro route needs: the DB session helper, the error mapping, the
model client, the image URL shape, and the three gates (Pro configured; the
coach/ops sign-in; demo-only buttons).
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from fastapi import HTTPException, Request

from server.services import pro_config, staff_auth
from server.services.db import session_scope
from server.services.demo_guard import client_ip
from server.services.nebius_client import NebiusChatClient
from server.services.ops_agent import AgentError
from server.services.paypal_client import PayPalError
from server.services.reviews import ReviewError
from server.services.snapshots import SnapshotError

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROMPTS_DIR = BASE_DIR / "prompts"
T = TypeVar("T")


def image_url(snapshot_id: str, position: int) -> str:
    return f"/api/reviews/images/{snapshot_id}/{position}.jpg"


def require_pro() -> None:
    """503 with a plain sentence when PayPal is not configured, so the page can hide the flow."""
    if not pro_config.pro_enabled():
        raise HTTPException(status_code=503, detail="Coach review is not set up on this server (PayPal sandbox keys are missing).")


def require_staff(request: Request) -> str:
    """
    Every coach and ops route: 401 without the signed session cookie. The
    page then shows the sign-in. Server side, not just hidden in the UI.
    """
    role = staff_auth.role_of(request)
    if role is None:
        detail = "Sign in as the coach to use this." if staff_auth.configured() else "Coach/ops sign-in is not set up on this server (COACH_PASSWORD is unset)."
        raise HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Cookie"})
    return role


def require_demo_mode() -> None:
    """The demo-only buttons (Expire now, Reset): on in demo mode, and only for signed-in staff (checked by the route)."""
    if not pro_config.demo_mode():
        raise HTTPException(status_code=403, detail="Demo controls are off on this server.")


def http_error(exc: BaseException) -> HTTPException:
    """Every Pro error as one plain sentence with the right status; PayPal's own text never reaches the page."""
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, (ReviewError, SnapshotError, AgentError)):
        headers = None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            headers = {"Retry-After": str(retry_after)}
        return HTTPException(status_code=exc.status, detail=exc.detail, headers=headers)
    if isinstance(exc, PayPalError):
        logger.warning("PayPal error reached a route: %s", exc)
        return HTTPException(status_code=502, detail="PayPal did not answer as expected. Try again in a moment.")
    if isinstance(exc, pro_config.ProConfigError):
        return HTTPException(status_code=503, detail=str(exc))
    logger.exception("Pro route failed")
    return HTTPException(status_code=500, detail="Something went wrong on the server.")


async def in_thread(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a sync DB/PayPal function off the event loop, mapping its errors."""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except Exception as exc:
        raise http_error(exc) from exc


def with_session(fn: Callable[..., T]) -> Callable[..., T]:
    """`fn(session, *args)` run inside one session_scope."""

    def runner(*args: Any, **kwargs: Any) -> T:
        with session_scope() as session:
            return fn(session, *args, **kwargs)

    return runner


def model_client(meter: Optional[Callable[[], None]] = None) -> Optional[NebiusChatClient]:
    """The Token Factory client, or None when no key is set (the draft then uses its template)."""
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("NEBIUS_BASE_URL") or NebiusChatClient.DEFAULT_BASE_URL
    return NebiusChatClient(api_key=api_key, base_url=base_url, meter=meter)


def visitor_key(request: Request) -> str:
    return client_ip(request)


def athlete_token_of(request: Request, body: Optional[dict] = None) -> Optional[str]:
    """The page's anonymous athlete token: a header, or a field in the JSON body."""
    token = (request.headers.get("x-athlete-token") or "").strip()
    if not token and body:
        token = str(body.get("athlete_token") or "").strip()
    return token[:64] or None
