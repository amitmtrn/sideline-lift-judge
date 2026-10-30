"""
The coach/ops sign-in.

    POST /api/auth/login    {password} → the session cookie (HttpOnly)
    POST /api/auth/logout   clears it
    GET  /api/auth/me       {staff: bool, configured: bool}

One password for the coach and ops roles, from COACH_PASSWORD; see
services/staff_auth.py. Athlete routes never need it.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from server.routes.deps import visitor_key
from server.services import staff_auth

router = APIRouter()


@router.post("/api/auth/login")
async def login(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    given = str((body or {}).get("password") or "") if isinstance(body, dict) else ""
    try:
        token = staff_auth.login(given, visitor=visitor_key(request))
    except staff_auth.LoginError as exc:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        raise HTTPException(status_code=exc.status, detail=exc.detail, headers=headers)
    response = JSONResponse(content={"staff": True, "role": staff_auth.ROLE})
    staff_auth.set_cookie(response, token, secure=staff_auth.request_is_https(request))
    return response


@router.post("/api/auth/logout")
async def logout():
    response = JSONResponse(content={"staff": False})
    staff_auth.clear_cookie(response)
    return response


@router.get("/api/auth/me")
async def me(request: Request):
    role = staff_auth.role_of(request)
    return JSONResponse(
        content={"staff": role is not None, "role": role, "configured": staff_auth.configured()},
        headers={"Cache-Control": "no-store"},
    )
