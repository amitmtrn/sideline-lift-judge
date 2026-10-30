"""
The Ops tab: the plain-English agent, its confirm cards and audit log, and
the demo-only controls. Every route needs the staff sign-in (routes/auth.py).

    POST /api/ops/agent                      a request → reply + proposed action cards
    POST /api/ops/actions/{id}/confirm       the tap: re-validate, execute, record
    POST /api/ops/actions/{id}/reject
    GET  /api/ops/actions                    the audit log
    GET  /api/ops/overview                   held reviews, ledger, athletes
    POST /api/ops/reviews/{id}/expire        demo: pull the deadline to now → auto-void
    POST /api/ops/reset                      demo: void every hold, re-seed
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from server.routes.deps import (
    PROMPTS_DIR,
    http_error,
    image_url,
    in_thread,
    model_client,
    require_demo_mode,
    require_pro,
    require_staff,
    visitor_key,
)
from server.services import ops_agent, pro_config, reviews
from server.services.budgets import agent_budget
from server.services.db import reset_demo_data, session_scope

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/ops/agent")
async def ops_agent_request(request: Request):
    require_pro()
    require_staff(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Send a JSON body with a prompt.")
    prompt = str((body or {}).get("prompt") or "") if isinstance(body, dict) else ""
    client = model_client()
    if client is None:
        raise HTTPException(status_code=503, detail="The ops agent needs NEBIUS_API_KEY on the server.")
    try:
        reply = await ops_agent.run_agent(
            prompt,
            session_factory=session_scope,
            client=client,
            prompts_dir=str(PROMPTS_DIR),
            visitor=visitor_key(request),
        )
    except ops_agent.AgentError as exc:
        raise http_error(exc)
    except Exception as exc:
        logger.exception("Ops agent failed")
        raise HTTPException(status_code=502, detail="The agent's model did not answer. Try again in a moment.") from exc
    return JSONResponse(
        content={
            "reply": reply.text,
            "actions": reply.actions,
            "tools_called": reply.tools_called,
            "model": reply.model,
            "steps": reply.steps,
            "tool_source": ops_agent.toolkit().label,
        }
    )


@router.post("/api/ops/actions/{action_id}/confirm")
async def confirm_action(action_id: str, request: Request):
    require_pro()
    require_staff(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    token = str((body or {}).get("token") or "") if isinstance(body, dict) else ""

    def work() -> Dict[str, Any]:
        with session_scope() as session:
            action = ops_agent.confirm(session, action_id, token)
            return ops_agent.action_card(action, with_token=False)

    return JSONResponse(content=await in_thread(work))


@router.post("/api/ops/actions/{action_id}/reject")
async def reject_action(action_id: str, request: Request):
    require_pro()
    require_staff(request)

    def work() -> Dict[str, Any]:
        with session_scope() as session:
            return ops_agent.action_card(ops_agent.reject(session, action_id), with_token=False)

    return JSONResponse(content=await in_thread(work))


@router.get("/api/ops/actions")
async def list_actions(request: Request):
    require_pro()
    require_staff(request)

    def work() -> List[Dict[str, Any]]:
        with session_scope() as session:
            return ops_agent.list_actions(session)

    return JSONResponse(content={"actions": await in_thread(work)}, headers={"Cache-Control": "no-store"})


@router.get("/api/ops/overview")
async def overview(request: Request):
    require_pro()
    require_staff(request)

    def work() -> Dict[str, Any]:
        with session_scope() as session:
            reviews.sweep(session)
            held = [
                reviews.review_payload(row, role="ops", image_url=image_url)
                for row in reviews.list_for_coach(session, include_final=False)
            ]
            for row in held:
                row.pop("ai_draft", None)
                row.pop("coach_review", None)
                snapshot = row.get("snapshot") or {}
                row["snapshot"] = {
                    "lift": snapshot.get("lift"),
                    "lift_label": snapshot.get("lift_label"),
                    "score": snapshot.get("score"),
                    "band": snapshot.get("band"),
                }
            return {
                "held": held,
                "ledger": reviews.ledger_summary(session, days=7),
                "athletes": [
                    {"id": a.id, "name": a.display_name, "email": a.payer_email} for a in ops_agent.demo_athletes(session)
                ],
                "agent_calls_remaining_today": agent_budget().remaining_today(),
                "tool_source": ops_agent.toolkit().label,
                "model": ops_agent.agent_model(),
                "sla_hours": pro_config.sla_hours(),
                "examples": [
                    "Refund Dan's review, I can't make it this week.",
                    "Invoice my 3 athletes $60 each for November programming.",
                    "What did I earn this week?",
                    "Which holds expire today?",
                ],
            }

    return JSONResponse(content=await in_thread(work), headers={"Cache-Control": "no-store"})


@router.post("/api/ops/reviews/{review_id}/expire")
async def expire_review(review_id: str, request: Request):
    """Demo: what the deadline would do, now."""
    require_pro()
    require_staff(request)
    require_demo_mode()

    def work() -> Dict[str, Any]:
        with session_scope() as session:
            review = reviews.expire_now(session, review_id)
            return reviews.review_payload(review, role="ops", image_url=image_url)

    return JSONResponse(content=await in_thread(work))


@router.post("/api/ops/reset")
async def reset_demo(request: Request):
    """Demo: release every sandbox hold this app placed, forget the rows, re-seed the athletes and the coach."""
    require_pro()
    require_staff(request)
    require_demo_mode()

    def work() -> Dict[str, Any]:
        released = 0
        with session_scope() as session:
            for review in reviews.list_for_coach(session, include_final=False):
                try:
                    reviews.decline(session, review.id)
                    released += 1
                except reviews.ReviewError:
                    pass
            counts = reset_demo_data(session)
        return {"released_holds": released, "deleted": counts}

    return JSONResponse(content=await in_thread(work))
