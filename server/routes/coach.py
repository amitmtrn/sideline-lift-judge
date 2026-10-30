"""
The coach's side: the queue sorted by deadline, one review with its AI
draft, Deliver (capture) and Decline (void). Every route needs the staff
sign-in (routes/auth.py); the queue never lists checkouts nobody paid for.

    GET  /api/coach/queue                 held first, nearest deadline first
    GET  /api/coach/reviews/{id}          opens it (AUTHORIZED → IN_REVIEW); draft included
    POST /api/coach/reviews/{id}/deliver  the coach's words → capture the hold
    POST /api/coach/reviews/{id}/decline  release the hold
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from server.routes.deps import image_url, in_thread, require_pro, require_staff
from server.services import reviews
from server.services.db import session_scope

router = APIRouter()


def _queue_row(review) -> Dict[str, Any]:
    body = reviews.review_payload(review, role="coach", image_url=image_url)
    snapshot = body.get("snapshot") or {}
    body["snapshot"] = {
        "lift": snapshot.get("lift"),
        "lift_label": snapshot.get("lift_label"),
        "score": snapshot.get("score"),
        "band": snapshot.get("band"),
        "keyframes": (snapshot.get("keyframes") or [])[:1],
        "angle_warning": snapshot.get("angle_warning"),
    }
    body.pop("ai_draft", None)
    body.pop("coach_review", None)
    body["has_draft"] = bool(review.ai_draft_json)
    return body


@router.get("/api/coach/queue")
async def coach_queue(request: Request):
    require_pro()
    require_staff(request)

    def work() -> List[Dict[str, Any]]:
        with session_scope() as session:
            reviews.sweep(session)
            return [_queue_row(row) for row in reviews.list_for_coach(session)]

    rows = await in_thread(work)
    return JSONResponse(content={"reviews": rows, "held": sum(1 for r in rows if r["status"] in reviews.HELD)}, headers={"Cache-Control": "no-store"})


@router.get("/api/coach/reviews/{review_id}")
async def coach_review(review_id: str, request: Request):
    require_pro()
    require_staff(request)

    def work() -> Dict[str, Any]:
        with session_scope() as session:
            reviews.sweep(session)
            review = reviews.open_review(session, review_id)
            return reviews.review_payload(review, role="coach", image_url=image_url)

    payload = await in_thread(work)
    if payload.get("ai_draft") is None and payload.get("status") in reviews.HELD:
        # The draft is made when the hold is placed; a restart in between
        # loses the background task, so ask again from here.
        from server.routes.checkout import spawn_draft

        spawn_draft(review_id)
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


@router.post("/api/coach/reviews/{review_id}/deliver")
async def deliver_review(review_id: str, request: Request):
    require_pro()
    require_staff(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Send the review as JSON.")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Send the review as a JSON object.")

    def work() -> Dict[str, Any]:
        with session_scope() as session:
            review = reviews.deliver(session, review_id, body)
            return reviews.review_payload(review, role="coach", image_url=image_url)

    return JSONResponse(content=await in_thread(work))


@router.post("/api/coach/reviews/{review_id}/decline")
async def decline_review(review_id: str, request: Request):
    require_pro()
    require_staff(request)

    def work() -> Dict[str, Any]:
        with session_scope() as session:
            review = reviews.decline(session, review_id)
            return reviews.review_payload(review, role="coach", image_url=image_url)

    return JSONResponse(content=await in_thread(work))
