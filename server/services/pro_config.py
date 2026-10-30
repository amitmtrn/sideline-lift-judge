"""
Configuration for Sideline Pro — the paid coach review on top of the judge.

Everything here is read live from the environment, like demo_guard, so a
config var on Heroku changes behaviour without a rebuild and the tests can
flip a value with monkeypatch:

    PAYPAL_ENV              sandbox (the only value this build accepts)
    PAYPAL_CLIENT_ID        the REST app's client id (public: the page loads
                            the JS SDK with it)
    PAYPAL_CLIENT_SECRET    the REST app's secret (server only)
    PAYPAL_WEBHOOK_ID       for webhook signature checks (P1; optional)
    DATABASE_URL            Heroku Postgres; a local SQLite file when unset
    PRO_DB_PATH             where that SQLite file lives (default .data/)
    REVIEW_PRICE_USD        15 unless set
    SLA_HOURS               48 unless set; never above 72 (PayPal's honor period)
    DRAFT_DAILY_LIMIT       coach-draft model calls per day (30 unless set)
    AGENT_DAILY_LIMIT       ops-agent model calls per day (60 unless set)
    AGENT_HOURLY_LIMIT      ops-agent requests per visitor per hour (10)
    PRO_DEMO_MODE           true (default): the demo-only buttons — Expire now,
                            Reset demo data — are on
    ADMIN_DEMO_KEY          optional; when set those buttons need X-Admin-Key
    PAYPAL_INVOICE_MAX_USD  200 unless set; AGENT_INVOICE_MAX_COUNT 5
    DEMO_ATHLETE_EMAILS     comma-separated sandbox personal emails for the
                            three seeded athletes (invoices go there)

Pro is "enabled" when a PayPal client id and secret are both set. Without
them the judge works exactly as before and the paid flow stays hidden.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SANDBOX_ENV = "sandbox"
SANDBOX_BASE_URL = "https://api-m.sandbox.paypal.com"
# PayPal guarantees a capture only inside the 3-day honor period; after that
# the authorization has to be reauthorized, which this build does not do.
MAX_SLA_HOURS = 72.0
DEFAULT_SLA_HOURS = 48.0
DEFAULT_PRICE_USD = 15.0
DEFAULT_DRAFT_DAILY_LIMIT = 30
DEFAULT_AGENT_DAILY_LIMIT = 60
DEFAULT_AGENT_HOURLY_LIMIT = 10
DEFAULT_INVOICE_MAX_USD = 200.0
DEFAULT_INVOICE_MAX_COUNT = 5
# A checkout the buyer never approved is forgotten after this long.
CHECKOUT_ABANDON_MINUTES = 30
# A proposed agent action must be confirmed within this long.
ACTION_CONFIRM_MINUTES = 10
DEFAULT_COACH_NAME = "Coach Maya"
DEFAULT_ATHLETES = ("Dan", "Noa", "Lior")


class ProConfigError(RuntimeError):
    """A Pro setting this build will not run with (a live PayPal environment, say)."""


def _flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _number(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def paypal_env() -> str:
    """PAYPAL_ENV: `sandbox` unless set. Anything else is refused by assert_sandbox()."""
    return (os.getenv("PAYPAL_ENV") or SANDBOX_ENV).strip().lower()


def assert_sandbox() -> None:
    """
    This hackathon build moves sandbox money only. A live environment is a
    configuration error, not a feature flag: refuse rather than charge anyone.
    """
    if paypal_env() != SANDBOX_ENV:
        raise ProConfigError(
            f"PAYPAL_ENV={paypal_env()!r}: this build is sandbox-only. Set PAYPAL_ENV=sandbox (or unset it)."
        )


def paypal_base_url() -> str:
    assert_sandbox()
    return SANDBOX_BASE_URL


def paypal_client_id() -> str:
    return (os.getenv("PAYPAL_CLIENT_ID") or "").strip()


def paypal_client_secret() -> str:
    return (os.getenv("PAYPAL_CLIENT_SECRET") or "").strip()


def paypal_webhook_id() -> str:
    return (os.getenv("PAYPAL_WEBHOOK_ID") or "").strip()


def pro_enabled() -> bool:
    """Whether the paid coach review is on: a PayPal client id and secret are both set."""
    return bool(paypal_client_id() and paypal_client_secret())


def database_url() -> str:
    """
    DATABASE_URL as SQLAlchemy wants it. Heroku hands out `postgres://`, which
    SQLAlchemy 2 no longer recognises, so it is rewritten to the psycopg
    driver. Unset: a local SQLite file (see PRO_DB_PATH).
    """
    raw = (os.getenv("DATABASE_URL") or "").strip()
    if not raw:
        return f"sqlite:///{sqlite_path()}"
    if raw.startswith("postgres://"):
        raw = "postgresql+psycopg://" + raw[len("postgres://") :]
    elif raw.startswith("postgresql://"):
        raw = "postgresql+psycopg://" + raw[len("postgresql://") :]
    return raw


def sqlite_path() -> Path:
    raw = (os.getenv("PRO_DB_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(".data") / "sideline-pro.sqlite"


def using_sqlite() -> bool:
    return database_url().startswith("sqlite:")


def price_usd() -> float:
    value = _number("REVIEW_PRICE_USD", DEFAULT_PRICE_USD)
    return value if value > 0 else DEFAULT_PRICE_USD


def price_cents() -> int:
    return int(round(price_usd() * 100))


def sla_hours() -> float:
    """SLA_HOURS, held to the 72 h capture-honor period whatever the variable says."""
    value = _number("SLA_HOURS", DEFAULT_SLA_HOURS)
    if value <= 0:
        value = DEFAULT_SLA_HOURS
    if value > MAX_SLA_HOURS:
        logger.warning("SLA_HOURS=%s is above PayPal's 72 h honor period; using %s", value, MAX_SLA_HOURS)
        return MAX_SLA_HOURS
    return value


def draft_daily_limit() -> int:
    return int(_number("DRAFT_DAILY_LIMIT", DEFAULT_DRAFT_DAILY_LIMIT))


def agent_daily_limit() -> int:
    return int(_number("AGENT_DAILY_LIMIT", DEFAULT_AGENT_DAILY_LIMIT))


def agent_hourly_limit() -> int:
    return int(_number("AGENT_HOURLY_LIMIT", DEFAULT_AGENT_HOURLY_LIMIT))


def invoice_max_usd() -> float:
    return _number("PAYPAL_INVOICE_MAX_USD", DEFAULT_INVOICE_MAX_USD)


def invoice_max_count() -> int:
    return int(_number("AGENT_INVOICE_MAX_COUNT", DEFAULT_INVOICE_MAX_COUNT))


def demo_mode() -> bool:
    """PRO_DEMO_MODE: the demo-only buttons (Expire now, Reset demo data). On unless set off."""
    return _flag("PRO_DEMO_MODE", True)


def admin_key() -> str:
    """ADMIN_DEMO_KEY: when set, the demo-only buttons need it as X-Admin-Key."""
    return (os.getenv("ADMIN_DEMO_KEY") or "").strip()


def coach_name() -> str:
    return (os.getenv("COACH_NAME") or DEFAULT_COACH_NAME).strip() or DEFAULT_COACH_NAME


def coach_email() -> str:
    """The sandbox business account's email, shown as the invoicer on invoices."""
    return (os.getenv("COACH_EMAIL") or "").strip()


def demo_athlete_emails() -> List[str]:
    """
    Sandbox personal-account emails for the seeded athletes, so an invoice the
    agent sends stays inside the sandbox. Fewer than three: the rest get
    placeholder addresses PayPal will accept but never deliver.
    """
    raw = os.getenv("DEMO_ATHLETE_EMAILS") or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


def public_config() -> dict:
    """What the page may know. The client id is public by design; nothing else is."""
    return {
        "pro_enabled": pro_enabled(),
        "paypal_env": paypal_env(),
        "paypal_client_id": paypal_client_id() if pro_enabled() else None,
        "price_usd": price_usd(),
        "sla_hours": sla_hours(),
        "demo_mode": demo_mode(),
        "demo_controls_locked": bool(admin_key()),
        "coach_name": coach_name(),
    }


def missing_pro_settings() -> List[str]:
    """What a Pro deploy is missing, for the startup log."""
    missing: List[str] = []
    if not paypal_client_id():
        missing.append("PAYPAL_CLIENT_ID")
    if not paypal_client_secret():
        missing.append("PAYPAL_CLIENT_SECRET")
    return missing


def describe() -> str:
    """One line for the startup log."""
    if not pro_enabled():
        return "Pro (coach review): off — set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET (sandbox) to enable"
    store = "Postgres" if not using_sqlite() else f"SQLite at {sqlite_path()} (set DATABASE_URL for a durable store)"
    return (
        f"Pro (coach review): on, PayPal {paypal_env()}, ${price_usd():.2f} per review, SLA {sla_hours():g} h, "
        f"store {store}, demo buttons {'on' if demo_mode() else 'off'}"
        f"{' (X-Admin-Key required)' if admin_key() else ''}"
    )


def optional_env(name: str) -> Optional[str]:
    value = (os.getenv(name) or "").strip()
    return value or None
