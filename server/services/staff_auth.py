"""
The coach and ops sign-in.

The coach queue, the editor, the ops agent and the demo buttons move money
(sandbox money, but money), and the URL is public until the judging ends. So
every coach and ops route is behind one password — COACH_PASSWORD in the
environment, never in the repo — and a signed session cookie:

    sp_staff = staff.<expires>.<nonce>.<hmac-sha256>

The signature key is derived from SESSION_SECRET, or from the password when
there is none, so changing either signs everyone out. The cookie is HttpOnly
and SameSite=Lax; it is sent Secure when the request came over HTTPS (Heroku's
router says so in X-Forwarded-Proto). No password is stored anywhere, and a
wrong guess costs one of STAFF_LOGIN_HOURLY_LIMIT tries per address per hour.

Athletes never sign in: their review is theirs by its unguessable id (128
random bits) and the anonymous token the page keeps.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from typing import Optional

from server.services.budgets import BudgetExceeded, DailyBudget

COOKIE = "sp_staff"
ROLE = "staff"
TTL_SECONDS = 30 * 24 * 3600
DEFAULT_LOGIN_HOURLY_LIMIT = 20


class LoginError(Exception):
    def __init__(self, detail: str, *, status: int, retry_after: Optional[int] = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status
        self.retry_after = retry_after


def password() -> str:
    return (os.getenv("COACH_PASSWORD") or "").strip()


def configured() -> bool:
    """Whether a coach/ops password is set. Without one the staff routes stay locked."""
    return bool(password())


def _key() -> bytes:
    seed = (os.getenv("SESSION_SECRET") or "").strip() or password()
    return hashlib.sha256(("sideline-pro-staff:" + seed).encode("utf-8")).digest()


def _sign(payload: str) -> str:
    return hmac.new(_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def issue(now: Optional[float] = None) -> str:
    """A fresh signed session token."""
    expires = int((now or time.time()) + TTL_SECONDS)
    payload = f"{ROLE}.{expires}.{secrets.token_hex(8)}"
    return f"{payload}.{_sign(payload)}"


def verify(token: Optional[str], now: Optional[float] = None) -> Optional[str]:
    """The role a token carries, or None when it is missing, forged, expired or no password is set."""
    if not token or not configured():
        return None
    parts = str(token).split(".")
    if len(parts) != 4:
        return None
    role, expires, nonce, signature = parts
    payload = f"{role}.{expires}.{nonce}"
    if not hmac.compare_digest(_sign(payload), signature):
        return None
    try:
        if int(expires) <= (now or time.time()):
            return None
    except ValueError:
        return None
    return role if role == ROLE else None


_attempts: Optional[DailyBudget] = None


def _login_budget() -> DailyBudget:
    global _attempts
    if _attempts is None:

        def hourly() -> int:
            raw = (os.getenv("STAFF_LOGIN_HOURLY_LIMIT") or "").strip()
            try:
                return int(raw) if raw else DEFAULT_LOGIN_HOURLY_LIMIT
            except ValueError:
                return DEFAULT_LOGIN_HOURLY_LIMIT

        _attempts = DailyBudget("sign-in", lambda: 0, hourly=hourly)
    return _attempts


def reset_login_attempts_for_tests() -> None:
    global _attempts
    _attempts = None


def login(given: str, *, visitor: str) -> str:
    """Check a password; a session token when it matches. Wrong guesses are counted per address."""
    if not configured():
        raise LoginError("Coach/ops sign-in is not set up on this server (COACH_PASSWORD is unset).", status=503)
    budget = _login_budget()
    try:
        budget.check(visitor)
    except BudgetExceeded as exc:
        raise LoginError("Too many sign-in attempts. Try again later.", status=429, retry_after=exc.retry_after)
    if not hmac.compare_digest((given or "").encode("utf-8"), password().encode("utf-8")):
        budget.charge(visitor)
        raise LoginError("Wrong password.", status=401)
    return issue()


def role_of(request) -> Optional[str]:
    """The signed-in role on a request, or None."""
    return verify(request.cookies.get(COOKIE))


def request_is_https(request) -> bool:
    forwarded = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    return forwarded == "https" or request.url.scheme == "https"


def set_cookie(response, token: str, *, secure: bool) -> None:
    response.set_cookie(
        key=COOKIE,
        value=token,
        max_age=TTL_SECONDS,
        path="/",
        secure=secure,
        httponly=True,
        samesite="lax",
    )


def clear_cookie(response) -> None:
    response.delete_cookie(key=COOKIE, path="/")
