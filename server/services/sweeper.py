"""
The background sweep: every minute, release expired holds and tidy stale
checkouts and claims (see reviews.sweep). Started by the app's lifespan when
Pro is on; the queue and status routes also sweep on demand, so a dyno that
slept through a deadline catches up on the next request.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

from server.services.db import session_scope
from server.services.ops_agent import expire_stale
from server.services.reviews import sweep

logger = logging.getLogger(__name__)

SWEEP_INTERVAL_SECONDS = 60.0


def run_sweep() -> Dict[str, int]:
    """One sweep, synchronously, in its own session."""
    with session_scope() as session:
        counts = sweep(session)
        counts["actions_expired"] = expire_stale(session)
    if any(counts.values()):
        logger.info("Sweep: %s", ", ".join(f"{key}={value}" for key, value in counts.items() if value))
    return counts


async def sweeper_loop(interval: float = SWEEP_INTERVAL_SECONDS) -> None:
    while True:
        try:
            await asyncio.to_thread(run_sweep)
        except asyncio.CancelledError:
            raise
        except Exception:  # a PayPal or DB hiccup must not stop the loop
            logger.exception("Sweep failed; trying again in %ss", interval)
        await asyncio.sleep(interval)


_task: Optional["asyncio.Task[None]"] = None


def start() -> None:
    global _task
    if _task is None or _task.done():
        _task = asyncio.create_task(sweeper_loop())


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None
