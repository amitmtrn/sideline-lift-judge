"""
The athlete's side: buy a coach review for a scored lift, approve it with
PayPal, and read its status afterwards.

    POST /api/checkout/reviews                 snapshot the lift, create the PayPal order
    POST /api/checkout/reviews/{id}/approve    onApprove: authorize (hold), start the SLA
    GET  /api/reviews/{id}                     the athlete's status page data
    GET  /api/reviews/mine                     this device's reviews
    GET  /api/reviews/images/{snap}/{n}.jpg    a keyframe from the snapshot
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select

from server.routes.deps import (
    PROMPTS_DIR,
    athlete_token_of,
    http_error,
    image_url,
    in_thread,
    model_client,
    require_pro,
)
from server.services import pro_config, reviews
from server.services.coach_draft import generate_draft, template_draft
from server.services.db import Athlete, ReviewOrder, athlete_for_session, dumps, loads, session_scope
from server.services.jobs import get_store
from server.services.sample import load_sample
from server.services.snapshots import SnapshotError, create_snapshot, keyframes_for_lift, pick_lift

logger = logging.getLogger(__name__)
router = APIRouter()

_draft_tasks: Dict[str, "asyncio.Task[None]"] = {}


# ---------- the AI coach draft, in the background ----------


async def draft_review(review_id: str) -> None:
    """One model call (or the template) and the result on the row. Never raises."""

    def load() -> Optional[Dict[str, Any]]:
        with session_scope() as session:
            review = session.get(ReviewOrder, review_id)
            if review is None or review.ai_draft_json:
                return None
            return {
                "lift": loads(review.snapshot.judgment_json, {}) if review.snapshot else {},
                "note": review.athlete_note,
                "coach": review.coach.name if review.coach else pro_config.coach_name(),
            }

    def store(draft: Dict[str, Any], source: str) -> None:
        with session_scope() as session:
            review = session.get(ReviewOrder, review_id)
            if review is not None and not review.ai_draft_json:
                review.ai_draft_json = dumps(draft)
                review.draft_status = "ready" if source == "model" else "template"

    try:
        data = await asyncio.to_thread(load)
        if data is None:
            return
        client = model_client()
        if client is None:
            draft, source = template_draft(data["lift"], coach_name=data["coach"]), "template"
        else:
            draft, source = await generate_draft(
                client=client, lift=data["lift"], note=data["note"], prompts_dir=str(PROMPTS_DIR), coach_name=data["coach"]
            )
        await asyncio.to_thread(store, draft, source)
    except Exception:
        logger.exception("Coach draft for review %s failed", review_id)
        try:
            await asyncio.to_thread(store, template_draft({}, coach_name=pro_config.coach_name()), "template")
        except Exception:
            pass
    finally:
        _draft_tasks.pop(review_id, None)


def spawn_draft(review_id: str) -> None:
    if review_id in _draft_tasks and not _draft_tasks[review_id].done():
        return
    _draft_tasks[review_id] = asyncio.create_task(draft_review(review_id))


async def wait_for_drafts() -> None:
    """Tests: let background drafts finish."""
    tasks = [task for task in _draft_tasks.values() if not task.done()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------- snapshot sources ----------


def _lift_and_stills(source: str, job_id: Optional[str], lift_index: Optional[int]):
    """The lift payload and its evidence stills from a live job or the cached sample."""
    if source == "sample":
        sample = load_sample()
        if sample is None:
            raise SnapshotError("No sample is set up on this server.", status=404)
        lift = pick_lift(sample.payload, lift_index)
        return "sample", None, lift, sample.stills
    if source == "job":
        job = get_store().get(job_id or "")
        if job is None or job.status != "done" or not job.payload:
            raise SnapshotError("That result has expired. Score the lift again, then request the review.", status=404)
        lift = pick_lift(job.payload, lift_index)
        return "job", job.id, lift, job.stills
    raise SnapshotError("source must be 'job' or 'sample'.")


# ---------- routes ----------


@router.post("/api/checkout/reviews")
async def create_review(request: Request):
    """
    Tap on "Get a coach's eyes": the lift is snapshotted into the database
    now (the job will not outlive the review), a review row is created and
    a PayPal order with intent AUTHORIZE is opened. The JS SDK's createOrder
    callback returns the order id from here.
    """
    require_pro()
    body = await _json(request)
    source = str(body.get("source") or "job").strip().lower()
    job_id = str(body.get("job_id") or "").strip() or None
    lift_index = body.get("lift_index")
    try:
        lift_index = int(lift_index) if lift_index is not None else None
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="lift_index must be a number.")
    token = athlete_token_of(request, body)
    name = str(body.get("athlete_name") or "").strip()[:80]
    note = str(body.get("note") or "").strip()[:500]

    def work() -> Dict[str, Any]:
        kind, source_job_id, lift, stills = _lift_and_stills(source, job_id, lift_index)
        keyframes = keyframes_for_lift(lift, stills)
        with session_scope() as session:
            athlete = athlete_for_session(session, token, name)
            snapshot = create_snapshot(session, source_kind=kind, source_job_id=source_job_id, lift=lift, keyframes=keyframes)
            review = reviews.start_checkout(session, athlete=athlete, snapshot=snapshot, note=note)
            return {
                "review_id": review.id,
                "paypal_order_id": review.paypal_order_id,
                "price_usd": review.price_cents / 100,
                "currency": review.currency,
                "status": review.status,
                "status_url": f"/review/{review.id}",
            }

    return JSONResponse(status_code=201, content=await in_thread(work))


@router.post("/api/checkout/reviews/{review_id}/approve")
async def approve_review(review_id: str):
    """onApprove: authorize the order. The hold is placed, the SLA starts, the draft is kicked off."""
    require_pro()

    def work() -> Dict[str, Any]:
        with session_scope() as session:
            review = reviews.approve(session, review_id)
            return reviews.review_payload(review, role="athlete", image_url=image_url)

    payload = await in_thread(work)
    if payload.get("status") in reviews.HELD:
        spawn_draft(review_id)
    return JSONResponse(content=payload)


@router.get("/api/reviews/mine")
async def my_reviews(request: Request):
    require_pro()
    token = athlete_token_of(request)
    if not token:
        return JSONResponse(content={"reviews": []})

    def work() -> List[Dict[str, Any]]:
        with session_scope() as session:
            reviews.sweep(session)
            athlete = session.scalar(select(Athlete).where(Athlete.session_token == token))
            if athlete is None:
                return []
            rows = session.scalars(
                select(ReviewOrder).where(ReviewOrder.athlete_id == athlete.id).order_by(ReviewOrder.created_at.desc()).limit(20)
            ).all()
            return [reviews.review_payload(row, role="athlete", image_url=image_url) for row in rows if row.status != reviews.ABANDONED]

    return JSONResponse(content={"reviews": await in_thread(work)})


@router.get("/api/reviews/{review_id}")
async def read_review(review_id: str):
    """The status page: the money's state in one line, the coach's words once captured."""
    require_pro()

    def work() -> Dict[str, Any]:
        with session_scope() as session:
            reviews.sweep(session)
            review = reviews.get_review(session, review_id)
            return reviews.review_payload(review, role="athlete", image_url=image_url)

    return JSONResponse(content=await in_thread(work), headers={"Cache-Control": "no-store"})


@router.get("/api/reviews/images/{snapshot_id}/{position}.jpg")
async def read_snapshot_image(snapshot_id: str, position: int):
    require_pro()

    def work():
        with session_scope() as session:
            image = reviews.snapshot_image(session, snapshot_id, position)
            return (image.content_type, bytes(image.data)) if image is not None else None

    found = await in_thread(work)
    if found is None:
        raise HTTPException(status_code=404, detail="No such keyframe.")
    content_type, data = found
    return Response(content=data, media_type=content_type, headers={"Cache-Control": "private, max-age=86400"})


async def _json(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Send a JSON body.")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Send a JSON object.")
    return body
