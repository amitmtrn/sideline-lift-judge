"""
What a user is told when scoring fails (spec §12: a graceful message, never a
stack trace). The raw exception goes to the server log; the phone gets one
plain sentence that says what to do next.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GENERIC = "Scoring failed on the server. Wait a moment and try the clip again."
BUSY = "The model service is busy right now. Wait a minute and try the clip again."
SLOW = "The model service took too long to answer. Try the clip again in a moment."
UNREACHABLE = "The model service could not be reached. Check the connection and try again."
NO_VERDICT = "The judge did not return a usable verdict for this clip. Try scoring it again."
MISCONFIGURED = "The scoring service is not configured correctly on the server."
TOO_LONG = "Scoring took too long and was stopped. Trim the clip to one lift and try again."

_STATUS_RE = re.compile(r"Nebius API error (\d{3})")


class UserFacingError(RuntimeError):
    """An error whose text was written for the user and may be shown as is."""


def user_message(exc: BaseException) -> str:
    """One sentence for the user. Logging the real error is the caller's job."""
    if isinstance(exc, UserFacingError):
        return str(exc)
    if isinstance(exc, asyncio.TimeoutError):
        return TOO_LONG
    if isinstance(exc, requests.exceptions.Timeout):
        return SLOW
    if isinstance(exc, requests.exceptions.ConnectionError):
        return UNREACHABLE
    status = _status_of(exc)
    if status is not None:
        if status in (401, 403):
            return MISCONFIGURED
        if status == 429 or status >= 500:
            return BUSY
        return GENERIC
    # Pydantic's ValidationError is a ValueError; so is our JSON extraction.
    if isinstance(exc, ValueError):
        return NO_VERDICT
    return GENERIC


def _status_of(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    match = _STATUS_RE.search(str(exc))
    return int(match.group(1)) if match else None


def detail_of(exc: BaseException) -> str:
    """The raw error, for logs and the CLI's --debug output. Never shown in the UI."""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text[:300]}" if text else type(exc).__name__
