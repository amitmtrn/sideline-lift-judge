"""
PayPal webhooks (P1). The P0 loop relies on synchronous API replies plus a
GET reconcile; this endpoint keeps the rows honest when PayPal acts on its
own — an authorization that expired, a capture that completed late, an
invoice that was paid. Every event is verified with PayPal (which the
dashboard's simulator events cannot pass, so test with real sandbox
payments) and handled once: a second delivery of the same event id is a
no-op.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from server.routes.deps import in_thread, require_pro
from server.services import pro_config, reviews
from server.services.db import InvoiceRef, PaymentEvent, ReviewOrder, record_event, session_scope
from server.services.paypal_client import PayPalError, get_paypal

logger = logging.getLogger(__name__)
router = APIRouter()

HANDLED = (
    "CHECKOUT.ORDER.APPROVED",
    "PAYMENT.AUTHORIZATION.CREATED",
    "PAYMENT.AUTHORIZATION.VOIDED",
    "PAYMENT.CAPTURE.COMPLETED",
    "PAYMENT.CAPTURE.DECLINED",
    "PAYMENT.CAPTURE.REFUNDED",
    "INVOICING.INVOICE.PAID",
)


def _find_review(session, resource: Dict[str, Any]) -> Optional[ReviewOrder]:
    custom_id = str(resource.get("custom_id") or "")
    if custom_id:
        review = session.get(ReviewOrder, custom_id)
        if review is not None:
            return review
    resource_id = str(resource.get("id") or "")
    if resource_id:
        review = session.scalar(
            select(ReviewOrder).where(
                (ReviewOrder.authorization_id == resource_id)
                | (ReviewOrder.capture_id == resource_id)
                | (ReviewOrder.paypal_order_id == resource_id)
            )
        )
        if review is not None:
            return review
    for link in resource.get("links") or []:
        href = str(link.get("href") or "")
        if "/authorizations/" in href:
            auth_id = href.rsplit("/authorizations/", 1)[-1].split("/")[0]
            review = session.scalar(select(ReviewOrder).where(ReviewOrder.authorization_id == auth_id))
            if review is not None:
                return review
    return None


def handle_event(session, event: Dict[str, Any]) -> str:
    """Apply one verified event. Returns what was done, for the log."""
    event_id = str(event.get("id") or "")
    event_type = str(event.get("event_type") or "")
    if event_id:
        seen = session.scalar(select(PaymentEvent).where(PaymentEvent.kind == "webhook", PaymentEvent.paypal_id == event_id))
        if seen is not None:
            return "duplicate"
    resource = event.get("resource") or {}
    review = _find_review(session, resource) if isinstance(resource, dict) else None
    record_event(
        session,
        review_id=review.id if review else None,
        kind="webhook",
        paypal_id=event_id or None,
        payload={"event_type": event_type, "resource_id": resource.get("id") if isinstance(resource, dict) else None},
    )
    if event_type == "INVOICING.INVOICE.PAID":
        invoice = resource.get("invoice") if isinstance(resource.get("invoice"), dict) else resource
        invoice_id = str((invoice or {}).get("id") or "")
        if invoice_id:
            for ref in session.scalars(select(InvoiceRef).where(InvoiceRef.paypal_invoice_id == invoice_id)).all():
                ref.status = "PAID"
        return "invoice_paid"
    if review is None:
        return "unmatched"
    if event_type == "PAYMENT.AUTHORIZATION.VOIDED" and review.status in reviews.HELD:
        # PayPal ended the hold itself (its validity period passed) or an
        # out-of-band void; either way nothing can be captured now.
        reviews._set_status(session, review, reviews.VOIDED, void_reason=review.void_reason or reviews.REASON_EXPIRED_AUTH)
        return "voided"
    if event_type == "PAYMENT.CAPTURE.COMPLETED" and review.status != reviews.CAPTURED and review.status != reviews.REFUNDED:
        reviews._set_status(session, review, reviews.CAPTURED, capture_id=str(resource.get("id") or review.capture_id or ""), delivered_at=review.delivered_at or reviews.utcnow())
        return "captured"
    if event_type == "PAYMENT.CAPTURE.REFUNDED" and review.status != reviews.REFUNDED:
        reviews._set_status(session, review, reviews.REFUNDED, refund_id=str(resource.get("id") or review.refund_id or ""))
        return "refunded"
    if event_type == "PAYMENT.CAPTURE.DECLINED":
        logger.warning("Capture declined for review %s", review.id)
        return "capture_declined"
    return "noted"


@router.post("/paypal/webhook")
async def paypal_webhook(request: Request):
    require_pro()
    webhook_id = pro_config.paypal_webhook_id()
    if not webhook_id:
        raise HTTPException(status_code=503, detail="PAYPAL_WEBHOOK_ID is not set; webhooks are off.")
    try:
        event = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Not JSON.")
    if not isinstance(event, dict):
        raise HTTPException(status_code=400, detail="Not an event.")
    headers = {key: value for key, value in request.headers.items()}

    def verify() -> bool:
        try:
            return get_paypal().verify_webhook_signature(webhook_id=webhook_id, headers=headers, event=event)
        except PayPalError as exc:
            logger.warning("Webhook verification call failed: %s", exc)
            return False

    if not await in_thread(verify):
        raise HTTPException(status_code=400, detail="Webhook signature did not verify.")

    def work() -> str:
        with session_scope() as session:
            return handle_event(session, event)

    outcome = await in_thread(work)
    return JSONResponse(content={"ok": True, "handled": outcome})
