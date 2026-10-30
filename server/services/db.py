"""
The durable store behind the paid coach review.

Jobs live in memory for thirty minutes and a Heroku dyno's disk is wiped on
every restart, so anything money is attached to — the lift the athlete paid
to have reviewed, the PayPal ids, the coach's words — goes here instead.
Heroku Postgres in the deploy (DATABASE_URL); a SQLite file in a local
checkout and in the tests. SQLAlchemy keeps the two the same from the code's
point of view.

Tables (PRD §9): athlete, coach, lift_snapshot (+ its JPEG keyframes),
review_order, payment_event, agent_action, invoice_ref. A review's state
machine lives in services/reviews.py; this module only knows the shapes.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    event,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from server.services import pro_config

logger = logging.getLogger(__name__)


def new_id() -> str:
    """128 random bits as hex: review and action URLs are unguessable, never sequential."""
    return secrets.token_hex(16)


def utcnow() -> datetime:
    """Naive UTC, which is what every column stores (SQLite keeps no zone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(value: Optional[datetime]) -> Optional[str]:
    """A stored timestamp as an ISO-8601 string with an explicit Z, for the API."""
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads(raw: Optional[str], default: Any = None) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except ValueError:
        return default


class Base(DeclarativeBase):
    pass


class Athlete(Base):
    __tablename__ = "athlete"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    # The anonymous session the page keeps in localStorage; one athlete per device.
    session_token: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
    display_name: Mapped[str] = mapped_column(String(80), default="Athlete")
    paypal_payer_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    payer_email: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    # Seeded demo athletes are the ones the ops agent may invoice.
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Coach(Base):
    __tablename__ = "coach"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(80))
    email: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    paypal_merchant_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sla_hours: Mapped[float] = mapped_column(default=pro_config.DEFAULT_SLA_HOURS)
    price_cents: Mapped[int] = mapped_column(Integer, default=1500)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class LiftSnapshot(Base):
    __tablename__ = "lift_snapshot"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    source_kind: Mapped[str] = mapped_column(String(16))  # job | sample
    source_job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    lift_index: Mapped[int] = mapped_column(Integer, default=1)
    lift_type: Mapped[str] = mapped_column(String(16))
    # The whole lift payload the card was drawn from: judgment, cues,
    # evidence, measurement, warnings, model names.
    judgment_json: Mapped[str] = mapped_column(Text)
    measurement_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    angle_warning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    images: Mapped[List["SnapshotImage"]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan", order_by="SnapshotImage.position"
    )


class SnapshotImage(Base):
    __tablename__ = "snapshot_image"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("lift_snapshot.id"))
    position: Mapped[int] = mapped_column(Integer, default=0)
    label: Mapped[str] = mapped_column(String(80), default="")
    content_type: Mapped[str] = mapped_column(String(40), default="image/jpeg")
    data: Mapped[bytes] = mapped_column(LargeBinary)

    snapshot: Mapped[LiftSnapshot] = relationship(back_populates="images")


class ReviewOrder(Base):
    __tablename__ = "review_order"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athlete.id"))
    coach_id: Mapped[str] = mapped_column(ForeignKey("coach.id"))
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("lift_snapshot.id"))
    status: Mapped[str] = mapped_column(String(24), index=True)
    price_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    paypal_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    authorization_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    auth_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    capture_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    refund_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sla_due_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    ai_draft_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    draft_status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | ready | template
    coach_review_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    athlete_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    void_reason: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    capture_attempts: Mapped[int] = mapped_column(Integer, default=0)
    opened_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    athlete: Mapped[Athlete] = relationship()
    coach: Mapped[Coach] = relationship()
    snapshot: Mapped[LiftSnapshot] = relationship()


class PaymentEvent(Base):
    __tablename__ = "payment_event"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    review_id: Mapped[Optional[str]] = mapped_column(ForeignKey("review_order.id"), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(32))  # create | authorize | capture | void | refund | webhook | state
    paypal_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    http_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    payload_excerpt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AgentAction(Base):
    __tablename__ = "agent_action"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    actor_role: Mapped[str] = mapped_column(String(16), default="ops")
    prompt: Mapped[str] = mapped_column(Text, default="")
    tool: Mapped[str] = mapped_column(String(48))
    args_json: Mapped[str] = mapped_column(Text, default="{}")
    summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), index=True)  # PROPOSED | CONFIRMED | EXECUTED | FAILED | EXPIRED | REJECTED
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    paypal_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    confirm_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class InvoiceRef(Base):
    __tablename__ = "invoice_ref"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    paypal_invoice_id: Mapped[str] = mapped_column(String(64), index=True)
    athlete_id: Mapped[Optional[str]] = mapped_column(ForeignKey("athlete.id"), nullable=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), default="SENT")
    agent_action_id: Mapped[Optional[str]] = mapped_column(ForeignKey("agent_action.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------- engine and sessions ----------

_lock = threading.Lock()
_engine: Optional[Engine] = None
_engine_url: Optional[str] = None
_sessions: Optional[sessionmaker] = None


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite:"):
        path = url[len("sqlite:///") :]
        if path and path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False}, future=True)

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(connection, _record) -> None:  # pragma: no cover - trivial
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        return engine
    return create_engine(url, pool_pre_ping=True, future=True)


def get_engine() -> Engine:
    """The process-wide engine for DATABASE_URL, created on first use and when the URL changes."""
    global _engine, _engine_url, _sessions
    url = pro_config.database_url()
    with _lock:
        if _engine is None or _engine_url != url:
            if _engine is not None:
                _engine.dispose()
            _engine = _make_engine(url)
            _engine_url = url
            _sessions = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
            Base.metadata.create_all(_engine)
        return _engine


def init_db() -> None:
    """Create the tables (idempotent) and the seed rows the demo needs."""
    get_engine()
    with session_scope() as session:
        ensure_seed(session)


def reset_db_for_tests(url: Optional[str] = None) -> None:
    """Drop the engine so the next access rebuilds it — against `url` when given."""
    global _engine, _engine_url, _sessions
    import os

    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _engine_url = None
        _sessions = None
    if url is not None:
        os.environ["DATABASE_URL"] = url


@contextmanager
def session_scope() -> Iterator[Session]:
    """One unit of work: commit on success, roll back on any exception."""
    get_engine()
    assert _sessions is not None
    session = _sessions()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


# ---------- seed data ----------


def ensure_seed(session: Session) -> Coach:
    """
    The one coach and the three demo athletes the agent may invoice. Runs
    on every start and on "Reset demo data"; existing rows are kept, so
    reviews are never orphaned.
    """
    coach = session.scalar(select(Coach).order_by(Coach.created_at).limit(1))
    if coach is None:
        coach = Coach(
            name=pro_config.coach_name(),
            email=pro_config.coach_email() or None,
            sla_hours=pro_config.sla_hours(),
            price_cents=pro_config.price_cents(),
        )
        session.add(coach)
    else:
        coach.name = pro_config.coach_name()
        coach.sla_hours = pro_config.sla_hours()
        coach.price_cents = pro_config.price_cents()
        if pro_config.coach_email():
            coach.email = pro_config.coach_email()
    emails = pro_config.demo_athlete_emails()
    for position, name in enumerate(pro_config.DEFAULT_ATHLETES):
        existing = session.scalar(select(Athlete).where(Athlete.is_demo.is_(True), Athlete.display_name == name))
        email = emails[position] if position < len(emails) else f"sb-athlete{position + 1}@personal.example.com"
        if existing is None:
            session.add(Athlete(display_name=name, is_demo=True, payer_email=email))
        elif position < len(emails):
            existing.payer_email = email
    session.flush()
    return coach


def default_coach(session: Session) -> Coach:
    coach = session.scalar(select(Coach).order_by(Coach.created_at).limit(1))
    return coach if coach is not None else ensure_seed(session)


def athlete_for_session(session: Session, token: Optional[str], display_name: Optional[str] = None) -> Athlete:
    """
    The athlete behind a page session, created on first sight. A name typed
    at checkout updates the record (the agent finds reviews by name).
    """
    token = (token or "").strip()[:64] or None
    athlete: Optional[Athlete] = None
    if token:
        athlete = session.scalar(select(Athlete).where(Athlete.session_token == token))
    name = (display_name or "").strip()[:80]
    if athlete is None:
        athlete = Athlete(session_token=token, display_name=name or "Athlete")
        session.add(athlete)
        session.flush()
    elif name and athlete.display_name != name:
        athlete.display_name = name
    return athlete


def record_event(
    session: Session,
    *,
    review_id: Optional[str],
    kind: str,
    paypal_id: Optional[str] = None,
    request_id: Optional[str] = None,
    http_status: Optional[int] = None,
    payload: Any = None,
) -> PaymentEvent:
    """Every PayPal call and every state change leaves one row behind."""
    excerpt: Optional[str] = None
    if payload is not None:
        text = payload if isinstance(payload, str) else dumps(payload)
        excerpt = text[:2000]
    event_row = PaymentEvent(
        review_id=review_id,
        kind=kind,
        paypal_id=paypal_id,
        request_id=request_id,
        http_status=http_status,
        payload_excerpt=excerpt,
    )
    session.add(event_row)
    return event_row


def reset_demo_data(session: Session) -> Dict[str, int]:
    """
    "Reset demo data": forget every review, event, action and invoice ref,
    keep the coach, re-seed the athletes. Sandbox holds already placed are
    not touched here — the caller voids them first.
    """
    counts: Dict[str, int] = {}
    for model in (InvoiceRef, AgentAction, PaymentEvent, ReviewOrder, SnapshotImage, LiftSnapshot):
        rows = session.scalars(select(model)).all()
        counts[model.__tablename__] = len(rows)
        for row in rows:
            session.delete(row)
    session.flush()
    for athlete in session.scalars(select(Athlete)).all():
        if not athlete.is_demo:
            session.delete(athlete)
    session.flush()
    ensure_seed(session)
    return counts
