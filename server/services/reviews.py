"""
A paid coach review, from checkout to capture, void or refund.

    CHECKOUT_CREATED  the athlete tapped Pay; a PayPal order (AUTHORIZE) exists
    AUTHORIZED        the buyer approved; the money is held, not charged
    IN_REVIEW         the coach has opened it
    CAPTURED          the coach delivered; the hold was captured (= charged)
    VOIDED            the hold was released: sla_expired | coach_declined |
                      agent_cancel | capture_failed | expired_authorization
    REFUNDED          captured, then refunded by the coach or the agent
    ABANDONED         the buyer never approved within 30 minutes
    CAPTURING / VOIDING / REFUNDING
                      transient: one actor has claimed the transition and is
                      talking to PayPal. Nobody else may touch the row.

Every transition is a compare-and-set UPDATE on the status column, so the
coach pressing Deliver and the sweeper voiding an expired hold cannot both
win: whichever claims the transient state first goes ahead, the other sees
zero rows updated and stops. No capture is attempted once `sla_due_at` has
passed, whatever the row says. Every PayPal call and every state change
writes a payment_event row.

The functions take an open Session and commit at the points where a claim
must be visible to other workers before the PayPal call is made. The
routes run them in a worker thread.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from server.services import pro_config
from server.services.db import (
    Athlete,
    LiftSnapshot,
    ReviewOrder,
    SnapshotImage,
    default_coach,
    dumps,
    iso,
    loads,
    record_event,
    utcnow,
)
from server.services.paypal_client import PayPalError, first_authorization, get_paypal
from server.services.snapshots import lift_key, lift_label

logger = logging.getLogger(__name__)

CHECKOUT_CREATED = "CHECKOUT_CREATED"
AUTHORIZED = "AUTHORIZED"
IN_REVIEW = "IN_REVIEW"
CAPTURING = "CAPTURING"
VOIDING = "VOIDING"
REFUNDING = "REFUNDING"
CAPTURED = "CAPTURED"
CAPTURE_FAILED = "CAPTURE_FAILED"
VOIDED = "VOIDED"
REFUNDED = "REFUNDED"
ABANDONED = "ABANDONED"

HELD = (AUTHORIZED, IN_REVIEW)
TRANSIENT = (CAPTURING, VOIDING, REFUNDING)
FINAL = (CAPTURED, VOIDED, REFUNDED, ABANDONED)
# The athlete may read the coach's words only once the money moved.
VISIBLE_TO_ATHLETE = (CAPTURED, REFUNDED)
# A transient claim older than this is a crashed worker; the sweeper reconciles it.
STALE_CLAIM_MINUTES = 5

REASON_SLA = "sla_expired"
REASON_DECLINED = "coach_declined"
REASON_AGENT = "agent_cancel"
REASON_CAPTURE_FAILED = "capture_failed"
REASON_EXPIRED_AUTH = "expired_authorization"


class ReviewError(Exception):
    """A transition that cannot happen, in words the page can show."""

    def __init__(self, detail: str, *, status: int = 409) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status


def _paypal():
    return get_paypal()


def get_review(session: Session, review_id: str) -> ReviewOrder:
    review = session.get(ReviewOrder, review_id)
    if review is None:
        raise ReviewError("No such review.", status=404)
    return review


def _claim(session: Session, review_id: str, allowed: Tuple[str, ...], claim: str, *, before_sla: bool = False) -> bool:
    """
    Compare-and-set: move `review_id` from one of `allowed` to `claim`. True
    when this caller won the row. `before_sla` refuses once the deadline has
    passed (a capture must never happen late).
    """
    statement = update(ReviewOrder).where(ReviewOrder.id == review_id, ReviewOrder.status.in_(allowed))
    if before_sla:
        statement = statement.where(ReviewOrder.sla_due_at > utcnow())
    result = session.execute(statement.values(status=claim, updated_at=utcnow()))
    session.commit()
    return result.rowcount == 1


def _set_status(session: Session, review: ReviewOrder, status: str, **fields: Any) -> None:
    previous = review.status
    review.status = status
    for key, value in fields.items():
        setattr(review, key, value)
    record_event(session, review_id=review.id, kind="state", payload={"from": previous, "to": status, **{k: str(v) for k, v in fields.items() if k == "void_reason"}})


# ---------- checkout ----------


def start_checkout(
    session: Session,
    *,
    athlete: Athlete,
    snapshot: LiftSnapshot,
    note: Optional[str] = None,
) -> ReviewOrder:
    """
    A review order plus its PayPal order (intent AUTHORIZE, custom_id = the
    review id). Nothing is held yet: that happens when the buyer approves.
    """
    coach = default_coach(session)
    review = ReviewOrder(
        athlete_id=athlete.id,
        coach_id=coach.id,
        snapshot_id=snapshot.id,
        status=CHECKOUT_CREATED,
        price_cents=coach.price_cents or pro_config.price_cents(),
        athlete_note=(note or "").strip()[:500] or None,
    )
    session.add(review)
    session.flush()
    request_id = f"create-{review.id}"
    try:
        order = _paypal().create_order(
            amount_cents=review.price_cents,
            custom_id=review.id,
            description=f"Coach review – {lift_label(snapshot.lift_type)}",
            request_id=request_id,
        )
    except PayPalError as exc:
        record_event(session, review_id=review.id, kind="create", request_id=request_id, http_status=exc.status, payload=exc.body)
        session.commit()
        raise ReviewError("PayPal could not start the checkout. Try again in a moment.", status=502) from exc
    review.paypal_order_id = str(order.get("id") or "")
    if not review.paypal_order_id:
        raise ReviewError("PayPal did not return an order id.", status=502)
    record_event(
        session,
        review_id=review.id,
        kind="create",
        paypal_id=review.paypal_order_id,
        request_id=request_id,
        http_status=201,
        payload={"status": order.get("status")},
    )
    session.commit()
    return review


def approve(session: Session, review_id: str) -> ReviewOrder:
    """
    onApprove: authorize the order. The money is now held for the coach; the
    SLA clock starts. Idempotent — a double tap finds the row AUTHORIZED and
    returns it unchanged.
    """
    review = get_review(session, review_id)
    if review.status in HELD or review.status in (CAPTURING, CAPTURED):
        return review
    if review.status != CHECKOUT_CREATED:
        raise ReviewError(f"This review is {review.status.lower().replace('_', ' ')}; it cannot be approved now.")
    if not review.paypal_order_id:
        raise ReviewError("This review has no PayPal order.", status=500)
    request_id = f"auth-{review.id}"
    try:
        body = _paypal().authorize_order(review.paypal_order_id, request_id=request_id)
    except PayPalError as exc:
        record_event(session, review_id=review.id, kind="authorize", request_id=request_id, http_status=exc.status, payload=exc.body)
        session.commit()
        if exc.status == 422 and "ORDER_ALREADY_AUTHORIZED" in exc.body:
            return reconcile(session, review)
        raise ReviewError("PayPal did not authorize the payment. Nothing was charged; try again.", status=502) from exc
    auth = first_authorization(body)
    auth_status = str(auth.get("status") or "").upper()
    record_event(
        session,
        review_id=review.id,
        kind="authorize",
        paypal_id=auth.get("id"),
        request_id=request_id,
        http_status=201,
        payload={"order_status": body.get("status"), "authorization_status": auth_status},
    )
    if not auth.get("id") or auth_status not in ("CREATED", "PENDING", "CAPTURED"):
        session.commit()
        raise ReviewError("PayPal declined the authorization. Nothing was charged.", status=402)
    now = utcnow()
    sla_hours = review.coach.sla_hours if review.coach and review.coach.sla_hours else pro_config.sla_hours()
    sla_hours = min(float(sla_hours), pro_config.MAX_SLA_HOURS)
    _set_status(
        session,
        review,
        AUTHORIZED,
        authorization_id=str(auth["id"]),
        auth_expires_at=_parse_time(auth.get("expiration_time")),
        sla_due_at=now + timedelta(hours=sla_hours),
    )
    session.commit()
    return review


def _parse_time(raw: Any) -> Optional[datetime]:
    """PayPal's ISO-8601 timestamps (always with a zone) as the naive UTC the columns store."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


# ---------- the coach ----------


def open_review(session: Session, review_id: str) -> ReviewOrder:
    """The coach opened it: AUTHORIZED → IN_REVIEW, the first time only (for the review-time metric)."""
    review = get_review(session, review_id)
    if review.status == AUTHORIZED:
        _set_status(session, review, IN_REVIEW, opened_at=review.opened_at or utcnow())
        session.commit()
    return review


def deliver(session: Session, review_id: str, coach_review: Dict[str, Any]) -> ReviewOrder:
    """
    Deliver = capture. Claims the row first (so the sweeper cannot void it
    under the call), refuses once the SLA has passed, captures with one
    retry on a transient error, and on a hard failure releases the hold so
    the athlete is never charged for a review that did not go through.
    """
    review = get_review(session, review_id)
    if review.status == CAPTURED:
        return review
    if review.status not in HELD:
        raise ReviewError(_cannot(review, "delivered"))
    if review.sla_due_at is None or review.sla_due_at <= utcnow():
        raise ReviewError("The review window has passed; the hold was released and nothing can be charged now.")
    if not _claim(session, review_id, HELD, CAPTURING, before_sla=True):
        session.refresh(review)
        raise ReviewError(_cannot(review, "delivered"))
    session.refresh(review)
    review.coach_review_json = dumps(_clean_review(coach_review, review))
    session.commit()

    last_error: Optional[PayPalError] = None
    for attempt in (1, 2):
        request_id = f"capture-{review.id}-{attempt}"
        review.capture_attempts = (review.capture_attempts or 0) + 1
        try:
            body = _paypal().capture_authorization(review.authorization_id or "", request_id=request_id)
        except PayPalError as exc:
            last_error = exc
            record_event(session, review_id=review.id, kind="capture", request_id=request_id, http_status=exc.status, payload=exc.body)
            session.commit()
            if exc.retryable and attempt == 1:
                continue
            break
        status = str(body.get("status") or "").upper()
        record_event(
            session,
            review_id=review.id,
            kind="capture",
            paypal_id=body.get("id"),
            request_id=request_id,
            http_status=201,
            payload={"status": status},
        )
        if status in ("COMPLETED", "PENDING") and body.get("id"):
            _set_status(session, review, CAPTURED, capture_id=str(body["id"]), delivered_at=utcnow())
            session.commit()
            return review
        last_error = PayPalError(201, f"capture status {status or 'unknown'}", name="CAPTURE_DECLINED")
        break

    # Capture failed twice (or was declined): record it, then release the hold.
    _set_status(session, review, CAPTURE_FAILED)
    session.commit()
    _void_now(session, review, REASON_CAPTURE_FAILED, allowed=(CAPTURE_FAILED,))
    detail = "PayPal could not capture the payment, so the hold was released and the athlete was not charged."
    if last_error is not None:
        logger.warning("Capture failed for review %s: %s", review.id, last_error)
    raise ReviewError(detail, status=502)


def decline(session: Session, review_id: str) -> ReviewOrder:
    """The coach passes: release the hold straight away."""
    review = get_review(session, review_id)
    if review.status not in HELD:
        raise ReviewError(_cannot(review, "declined"))
    return _void_now(session, review, REASON_DECLINED)


def _cannot(review: ReviewOrder, verb: str) -> str:
    state = review.status.lower().replace("_", " ")
    return f"This review is already {state}; it cannot be {verb}."


def _clean_review(body: Dict[str, Any], review: ReviewOrder) -> Dict[str, Any]:
    """The coach's words, trimmed and typed; a kept-text ratio against the AI draft for the metric."""
    import difflib

    summary = str(body.get("summary") or "").strip()[:2000]
    cues_raw = body.get("cues") or []
    cues: List[str] = []
    for cue in cues_raw if isinstance(cues_raw, list) else []:
        text = cue.get("text") if isinstance(cue, dict) else cue
        text = str(text or "").strip()[:300]
        if text:
            cues.append(text)
    cues = cues[:3]
    drill = body.get("drill") or {}
    if isinstance(drill, str):
        drill = {"name": drill.strip()[:200], "sets_reps": ""}
    elif isinstance(drill, dict):
        drill = {"name": str(drill.get("name") or "").strip()[:200], "sets_reps": str(drill.get("sets_reps") or "").strip()[:100]}
    else:
        drill = {"name": "", "sets_reps": ""}
    score = body.get("score")
    try:
        score = int(score) if score is not None and str(score).strip() != "" else None
    except (TypeError, ValueError):
        score = None
    if score is not None and not 1 <= score <= 10:
        score = None
    band = str(body.get("band") or "").strip().lower() or None
    if band not in ("elite", "solid", "mixed", "poor", "fail", None):
        band = None
    if not summary and not cues:
        raise ReviewError("Write a summary or at least one cue before delivering.", status=400)
    draft = loads(review.ai_draft_json, {}) or {}
    draft_text = " ".join([str(draft.get("summary") or "")] + [str(c.get("text") or "") for c in draft.get("cues") or [] if isinstance(c, dict)])
    final_text = " ".join([summary] + cues)
    kept = round(difflib.SequenceMatcher(None, draft_text, final_text).ratio(), 2) if draft_text else None
    seconds = None
    if review.opened_at is not None:
        seconds = int((utcnow() - review.opened_at).total_seconds())
    return {
        "summary": summary,
        "cues": [{"text": cue} for cue in cues],
        "drill": drill,
        "score": score,
        "band": band,
        "draft_kept_ratio": kept,
        "review_seconds": seconds,
        "reviewed_by": review.coach.name if review.coach else pro_config.coach_name(),
        "ai_assisted": bool(draft),
    }


# ---------- releasing and refunding ----------


def _void_now(session: Session, review: ReviewOrder, reason: str, *, allowed: Tuple[str, ...] = HELD) -> ReviewOrder:
    """Claim, void at PayPal, mark VOIDED. A failed void puts the row back so the sweeper retries."""
    previous = review.status
    if not _claim(session, review.id, allowed, VOIDING):
        session.refresh(review)
        raise ReviewError(_cannot(review, "released"))
    session.refresh(review)
    request_id = f"void-{review.id}"
    try:
        body = _paypal().void_authorization(review.authorization_id or "", request_id=request_id)
    except PayPalError as exc:
        record_event(session, review_id=review.id, kind="void", request_id=request_id, http_status=exc.status, payload=exc.body)
        if exc.status == 422 and ("AUTHORIZATION_ALREADY_CAPTURED" in exc.body or "PREVIOUSLY_VOIDED" in exc.body or "AUTHORIZATION_VOIDED" in exc.body):
            session.commit()
            return reconcile(session, review)
        # Put it back where it was so the next sweep tries again.
        review.status = previous if previous in allowed else allowed[0]
        session.commit()
        raise ReviewError("PayPal could not release the hold right now; it will be retried.", status=502) from exc
    record_event(session, review_id=review.id, kind="void", paypal_id=review.authorization_id, request_id=request_id, http_status=204, payload={"status": body.get("status")})
    _set_status(session, review, VOIDED, void_reason=reason)
    session.commit()
    return review


def refund(session: Session, review_id: str, *, note: str = "") -> ReviewOrder:
    """A full refund of a captured review. The athlete keeps the words."""
    review = get_review(session, review_id)
    if review.status == REFUNDED:
        return review
    if review.status != CAPTURED or not review.capture_id:
        raise ReviewError(_cannot(review, "refunded"))
    if not _claim(session, review_id, (CAPTURED,), REFUNDING):
        session.refresh(review)
        raise ReviewError(_cannot(review, "refunded"))
    session.refresh(review)
    request_id = f"refund-{review.id}"
    try:
        body = _paypal().refund_capture(review.capture_id, request_id=request_id, amount_cents=review.price_cents, note=note)
    except PayPalError as exc:
        record_event(session, review_id=review.id, kind="refund", request_id=request_id, http_status=exc.status, payload=exc.body)
        review.status = CAPTURED
        session.commit()
        raise ReviewError("PayPal could not refund the payment right now.", status=502) from exc
    status = str(body.get("status") or "").upper()
    record_event(session, review_id=review.id, kind="refund", paypal_id=body.get("id"), request_id=request_id, http_status=201, payload={"status": status})
    if status not in ("COMPLETED", "PENDING"):
        review.status = CAPTURED
        session.commit()
        raise ReviewError(f"PayPal reported the refund as {status or 'unknown'}.", status=502)
    _set_status(session, review, REFUNDED, refund_id=str(body.get("id") or ""))
    session.commit()
    return review


def cancel_or_refund(session: Session, review_id: str, *, reason: str = REASON_AGENT, note: str = "") -> Tuple[str, ReviewOrder]:
    """
    The agent's one write on reviews: release the hold if it is still held,
    refund if it was captured. Returns which happened.
    """
    review = get_review(session, review_id)
    if review.status in HELD:
        return "voided", _void_now(session, review, reason)
    if review.status == CAPTURED:
        return "refunded", refund(session, review_id, note=note)
    raise ReviewError(_cannot(review, "released or refunded"))


def expire_now(session: Session, review_id: str) -> ReviewOrder:
    """Demo only: pull the deadline to now and let the sweeper do what it would in 48 h."""
    review = get_review(session, review_id)
    if review.status not in HELD:
        raise ReviewError(_cannot(review, "expired"))
    review.sla_due_at = utcnow() - timedelta(seconds=1)
    record_event(session, review_id=review.id, kind="state", payload={"demo": "expire_now"})
    session.commit()
    sweep(session)
    session.refresh(review)
    return review


# ---------- the sweeper ----------


def sweep(session: Session, now: Optional[datetime] = None) -> Dict[str, int]:
    """
    What time does on its own: abandon stale checkouts, release expired holds,
    reconcile claims a crashed worker left behind. Runs every minute in the
    background and again on every queue or status request, in case the dyno
    slept through a deadline.
    """
    now = now or utcnow()
    counts = {"abandoned": 0, "voided": 0, "reconciled": 0, "failed": 0}

    stale_checkouts = session.scalars(
        select(ReviewOrder).where(
            ReviewOrder.status == CHECKOUT_CREATED,
            ReviewOrder.created_at < now - timedelta(minutes=pro_config.CHECKOUT_ABANDON_MINUTES),
        )
    ).all()
    for review in stale_checkouts:
        if _claim(session, review.id, (CHECKOUT_CREATED,), ABANDONED):
            session.refresh(review)
            record_event(session, review_id=review.id, kind="state", payload={"from": CHECKOUT_CREATED, "to": ABANDONED})
            counts["abandoned"] += 1
    session.commit()

    expired = session.scalars(
        select(ReviewOrder).where(ReviewOrder.status.in_(HELD), ReviewOrder.sla_due_at <= now)
    ).all()
    for review in expired:
        try:
            _void_now(session, review, REASON_SLA)
            counts["voided"] += 1
        except ReviewError as exc:
            counts["failed"] += 1
            logger.warning("Sweep could not release review %s: %s", review.id, exc.detail)

    stuck = session.scalars(
        select(ReviewOrder).where(
            ReviewOrder.status.in_(TRANSIENT),
            ReviewOrder.updated_at < now - timedelta(minutes=STALE_CLAIM_MINUTES),
        )
    ).all()
    for review in stuck:
        try:
            reconcile(session, review)
            counts["reconciled"] += 1
        except ReviewError as exc:
            counts["failed"] += 1
            logger.warning("Sweep could not reconcile review %s: %s", review.id, exc.detail)
    return counts


def reconcile(session: Session, review: ReviewOrder) -> ReviewOrder:
    """Ask PayPal what actually happened to the order and set the row to match."""
    if not review.paypal_order_id:
        raise ReviewError("No PayPal order to reconcile against.", status=500)
    try:
        order = _paypal().get_order(review.paypal_order_id)
    except PayPalError as exc:
        raise ReviewError("PayPal could not be reached to reconcile.", status=502) from exc
    payments: Dict[str, Any] = {}
    try:
        payments = order["purchase_units"][0].get("payments") or {}
    except (KeyError, IndexError, TypeError):
        payments = {}
    captures = [c for c in payments.get("captures") or [] if str(c.get("status", "")).upper() in ("COMPLETED", "PENDING")]
    refunds = [r for r in payments.get("refunds") or [] if str(r.get("status", "")).upper() in ("COMPLETED", "PENDING")]
    auths = payments.get("authorizations") or []
    auth = auths[0] if auths else {}
    record_event(session, review_id=review.id, kind="reconcile", paypal_id=review.paypal_order_id, http_status=200, payload={"order_status": order.get("status"), "authorization_status": auth.get("status")})
    if refunds:
        _set_status(session, review, REFUNDED, refund_id=str(refunds[0].get("id") or ""), capture_id=review.capture_id or (str(captures[0]["id"]) if captures else None))
    elif captures:
        _set_status(session, review, CAPTURED, capture_id=str(captures[0].get("id") or ""), delivered_at=review.delivered_at or utcnow())
    elif str(auth.get("status", "")).upper() in ("VOIDED", "EXPIRED"):
        _set_status(session, review, VOIDED, void_reason=review.void_reason or REASON_EXPIRED_AUTH)
    elif str(auth.get("status", "")).upper() in ("CREATED", "PENDING", "PARTIALLY_CAPTURED"):
        if review.status not in HELD:
            _set_status(session, review, IN_REVIEW if review.opened_at else AUTHORIZED, authorization_id=review.authorization_id or str(auth.get("id") or ""))
    elif str(order.get("status", "")).upper() in ("CREATED", "APPROVED", "PAYER_ACTION_REQUIRED"):
        if review.status != CHECKOUT_CREATED:
            _set_status(session, review, CHECKOUT_CREATED)
    session.commit()
    return review


# ---------- what the page sees ----------


def price_text(review: ReviewOrder) -> str:
    return f"${review.price_cents / 100:.0f}" if review.price_cents % 100 == 0 else f"${review.price_cents / 100:.2f}"


def athlete_status(review: ReviewOrder) -> Dict[str, str]:
    """One label and one sentence for the athlete; the money's state is always in it."""
    price = price_text(review)
    coach = review.coach.name if review.coach else pro_config.coach_name()
    due = iso(review.sla_due_at) or ""
    status = review.status
    if status == CHECKOUT_CREATED:
        return {"label": "Awaiting PayPal approval", "detail": f"Finish the PayPal checkout to hold {price} for {coach}."}
    if status == AUTHORIZED:
        return {"label": f"Held · {price}", "detail": f"Held, not charged, until {coach} delivers. Past the deadline the hold is released by itself.", "due": due}
    if status == IN_REVIEW:
        return {"label": f"In review · {price} held", "detail": f"{coach} is on it. You are charged only when the review arrives before the deadline.", "due": due}
    if status in TRANSIENT:
        return {"label": "Updating…", "detail": "PayPal is confirming the payment status."}
    if status == CAPTURED:
        return {"label": f"Delivered · charged {price}", "detail": f"{coach} delivered. {price} was captured from the hold."}
    if status == REFUNDED:
        return {"label": f"Refunded {price}", "detail": f"{price} was refunded to you. The review stays yours."}
    if status == ABANDONED:
        return {"label": "Checkout not completed", "detail": "The PayPal approval was never finished. Nothing was held."}
    if status == CAPTURE_FAILED:
        return {"label": "Payment problem", "detail": "PayPal could not capture the payment. The hold is being released."}
    reason = review.void_reason or ""
    if reason == REASON_SLA:
        return {"label": "Released · not charged", "detail": f"Coach missed the deadline. Your {price} hold was released. You were not charged."}
    if reason == REASON_DECLINED:
        return {"label": "Released · not charged", "detail": f"{coach} declined this one. Your {price} hold was released. You were not charged."}
    if reason == REASON_CAPTURE_FAILED:
        return {"label": "Released · not charged", "detail": f"The payment could not be captured, so your {price} hold was released. You were not charged."}
    return {"label": "Released · not charged", "detail": f"Your {price} hold was released by the coach. You were not charged."}


def snapshot_payload(snapshot: LiftSnapshot, *, image_url) -> Dict[str, Any]:
    lift = loads(snapshot.judgment_json, {}) or {}
    cues = lift.get("cues") or []
    evidenced = lift.get("evidenced_cues") or []
    return {
        "id": snapshot.id,
        "lift": lift_key(snapshot.lift_type),
        "lift_label": lift_label(snapshot.lift_type),
        "score": lift.get("score"),
        "band": lift.get("band"),
        "confidence": lift.get("confidence"),
        "cues": [str(c) for c in cues],
        "evidenced_cues": [
            {"id": ec.get("id"), "text": ec.get("text"), "phase": ec.get("phase"), "why": ec.get("why"), "verified": ec.get("verified"), "vision_only": ec.get("vision_only")}
            for ec in evidenced
            if isinstance(ec, dict)
        ],
        "drift": (lift.get("drift") or {}).get("text") if isinstance(lift.get("drift"), dict) else lift.get("max_horizontal_drift"),
        "warnings": lift.get("warnings") or [],
        "angle_warning": snapshot.angle_warning,
        "notes": lift.get("notes"),
        "models": lift.get("models") or {},
        "measurement": loads(snapshot.measurement_json, None),
        "source": snapshot.source_kind,
        "keyframes": [
            {"url": image_url(snapshot.id, image.position), "label": image.label}
            for image in snapshot.images
        ],
    }


def review_payload(review: ReviewOrder, *, role: str, image_url) -> Dict[str, Any]:
    """
    The review as one role may see it. The athlete gets the coach's words
    only once the money moved and never the AI draft; the coach and ops get
    the lot, the AI draft clearly labelled as a draft.
    """
    status = athlete_status(review)
    body: Dict[str, Any] = {
        "id": review.id,
        "status": review.status,
        "status_label": status["label"],
        "status_detail": status["detail"],
        "void_reason": review.void_reason,
        "price_usd": review.price_cents / 100,
        "price_text": price_text(review),
        "currency": review.currency,
        "coach_name": review.coach.name if review.coach else pro_config.coach_name(),
        "created_at": iso(review.created_at),
        "updated_at": iso(review.updated_at),
        "sla_due_at": iso(review.sla_due_at),
        "delivered_at": iso(review.delivered_at),
        "seconds_to_deadline": None
        if review.sla_due_at is None
        else int((review.sla_due_at - utcnow()).total_seconds()),
        "paypal": {
            "order_id": review.paypal_order_id,
            "authorization_id": review.authorization_id,
            "capture_id": review.capture_id,
            "refund_id": review.refund_id,
        },
        "snapshot": snapshot_payload(review.snapshot, image_url=image_url) if review.snapshot else None,
        "athlete_note": review.athlete_note,
        "status_url": f"/review/{review.id}",
    }
    coach_review = loads(review.coach_review_json, None)
    if role == "athlete":
        body["coach_review"] = coach_review if review.status in VISIBLE_TO_ATHLETE else None
        body["review_visible"] = review.status in VISIBLE_TO_ATHLETE
        body["athlete_name"] = review.athlete.display_name if review.athlete else None
    else:
        body["athlete_name"] = review.athlete.display_name if review.athlete else None
        body["athlete_id"] = review.athlete_id
        body["ai_draft"] = loads(review.ai_draft_json, None)
        body["draft_status"] = review.draft_status
        body["coach_review"] = coach_review
        body["opened_at"] = iso(review.opened_at)
        body["capture_attempts"] = review.capture_attempts
    return body


def list_for_coach(session: Session, *, include_final: bool = True) -> List[ReviewOrder]:
    """
    The queue: held reviews first, nearest deadline first, then the rest
    newest first. Checkouts nobody approved (CHECKOUT_CREATED, and ABANDONED
    once the sweeper gave up on them) are not the coach's business.
    """
    rows = session.scalars(select(ReviewOrder).where(ReviewOrder.status.not_in((CHECKOUT_CREATED, ABANDONED)))).all()
    held = sorted([r for r in rows if r.status in HELD or r.status in TRANSIENT], key=lambda r: (r.sla_due_at or utcnow()))
    if not include_final:
        return held
    rest = sorted([r for r in rows if r.status not in HELD and r.status not in TRANSIENT], key=lambda r: r.updated_at, reverse=True)
    return held + rest


def find_reviews(
    session: Session,
    *,
    athlete: Optional[str] = None,
    status: Optional[str] = None,
    lift: Optional[str] = None,
    limit: int = 20,
) -> List[ReviewOrder]:
    """The agent's lookup: by athlete name (case-insensitive substring), status and lift."""
    rows = list_for_coach(session)
    wanted_status = (status or "").strip().upper()
    wanted_lift = lift_key(lift) if (lift or "").strip() else ""
    wanted_name = (athlete or "").strip().lower()
    out: List[ReviewOrder] = []
    for review in rows:
        if wanted_status and wanted_status != "ANY":
            if wanted_status == "HELD" and review.status not in HELD:
                continue
            if wanted_status != "HELD" and review.status != wanted_status:
                continue
        if wanted_lift and review.snapshot and lift_key(review.snapshot.lift_type) != wanted_lift:
            continue
        if wanted_name:
            name = (review.athlete.display_name if review.athlete else "").lower()
            if wanted_name not in name and name not in wanted_name:
                continue
        out.append(review)
        if len(out) >= limit:
            break
    return out


def ledger_summary(session: Session, *, days: int = 7) -> Dict[str, Any]:
    """Earnings from the local ledger (Transaction Search lags hours; this does not)."""
    since = utcnow() - timedelta(days=days)
    rows = session.scalars(select(ReviewOrder)).all()
    captured = [r for r in rows if r.status in (CAPTURED, REFUNDED) and r.delivered_at and r.delivered_at >= since]
    refunded = [r for r in captured if r.status == REFUNDED]
    held = [r for r in rows if r.status in HELD]
    expiring_today = [r for r in held if r.sla_due_at and r.sla_due_at <= utcnow() + timedelta(hours=24)]
    gross = sum(r.price_cents for r in captured)
    back = sum(r.price_cents for r in refunded)
    return {
        "period_days": days,
        "reviews_delivered": len(captured),
        "reviews_refunded": len(refunded),
        "gross_usd": gross / 100,
        "refunded_usd": back / 100,
        "net_usd": (gross - back) / 100,
        "holds_open": len(held),
        "holds_expiring_within_24h": len(expiring_today),
        "held_usd": sum(r.price_cents for r in held) / 100,
    }


def snapshot_image(session: Session, snapshot_id: str, position: int) -> Optional[SnapshotImage]:
    return session.scalar(
        select(SnapshotImage).where(SnapshotImage.snapshot_id == snapshot_id, SnapshotImage.position == position)
    )
