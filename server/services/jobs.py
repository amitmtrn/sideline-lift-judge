"""
Background scoring jobs with polling.

A phone on gym wifi cannot hold a request open for the length of two model
stages plus an ffmpeg render, so the upload returns a job id and the page polls
it (spec §13). Jobs live in this process only — one container, one store, which
is all the MVP deploys.

Privacy (§12): the uploaded clip is deleted as soon as scoring finishes, and the
whole job directory — annotated replays included — is removed once its TTL
passes or the process exits.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 1800  # 30 minutes: long enough to watch and share a replay
DEFAULT_MAX_JOBS = 200


def job_ttl_seconds() -> int:
    try:
        value = int(os.getenv("JOB_TTL_SECONDS") or DEFAULT_TTL_SECONDS)
    except ValueError:
        return DEFAULT_TTL_SECONDS
    return value if value > 0 else DEFAULT_TTL_SECONDS


@dataclass
class Job:
    id: str
    workdir: Path
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    message: str = "Starting…"
    progress: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source_path: Optional[Path] = None
    replays: Dict[int, Path] = field(default_factory=dict)
    task: Optional["asyncio.Task[None]"] = None

    def touch(self) -> None:
        self.updated_at = time.time()

    def set_progress(self, stage: str, message: str, fraction: float) -> None:
        self.stage = stage
        self.message = message
        # Never let a later stage report backwards; the UI only moves forward.
        self.progress = max(self.progress, min(1.0, max(0.0, fraction)))
        self.touch()

    def as_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "progress": round(self.progress, 3),
        }
        if self.error:
            body["error"] = self.error
        body.update(self.payload)
        return body

    def discard_source(self) -> None:
        """Drop the uploaded clip the moment it is no longer needed."""
        if self.source_path is not None:
            try:
                Path(self.source_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete upload for job %s: %s", self.id, exc)
            self.source_path = None


class JobStore:
    """In-process job registry. Expired jobs are purged on every access."""

    def __init__(self, root: Optional[Path] = None, *, max_jobs: int = DEFAULT_MAX_JOBS) -> None:
        self._root = Path(root) if root else Path(tempfile.gettempdir()) / "sideline-lift-judge"
        self._jobs: Dict[str, Job] = {}
        self._max_jobs = max_jobs

    @property
    def root(self) -> Path:
        return self._root

    def create(self) -> Job:
        self.purge_expired()
        job_id = uuid.uuid4().hex
        workdir = self._root / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, workdir=workdir)
        self._jobs[job_id] = job
        if len(self._jobs) > self._max_jobs:
            self._drop_oldest(len(self._jobs) - self._max_jobs)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        self.purge_expired()
        return self._jobs.get(job_id)

    def purge_expired(self, ttl: Optional[int] = None) -> int:
        ttl = ttl if ttl is not None else job_ttl_seconds()
        cutoff = time.time() - ttl
        expired = [job for job in self._jobs.values() if job.updated_at < cutoff and job.status in ("done", "error")]
        # A job still running past its TTL keeps its files; it is about to finish
        # or already failed, and deleting under a live ffmpeg would be worse.
        for job in expired:
            self._remove(job)
        return len(expired)

    def _drop_oldest(self, count: int) -> None:
        for job in sorted(self._jobs.values(), key=lambda item: item.updated_at)[:count]:
            if job.status in ("done", "error"):
                self._remove(job)

    def discard(self, job: Job) -> None:
        """
        Drop a job that never got going — a rejected upload or trim window —
        together with its directory. purge_expired() cannot do this: it only
        touches finished jobs, and a job refused before start() is still "queued".
        """
        if job.task is not None and not job.task.done():
            job.task.cancel()
        self._remove(job)

    def _remove(self, job: Job) -> None:
        self._jobs.pop(job.id, None)
        job.discard_source()
        shutil.rmtree(job.workdir, ignore_errors=True)

    def shutdown(self) -> None:
        """Cancel live jobs and remove every working directory."""
        for job in list(self._jobs.values()):
            if job.task is not None and not job.task.done():
                job.task.cancel()
            self._remove(job)
        shutil.rmtree(self._root, ignore_errors=True)

    def start(self, job: Job, runner: Callable[[Job], Awaitable[None]]) -> Job:
        """Run `runner` in the background; failures land on the job, not the loop."""

        async def guarded() -> None:
            job.status = "running"
            job.touch()
            try:
                await runner(job)
                if job.status == "running":
                    job.status = "done"
                    job.progress = 1.0
            except asyncio.CancelledError:
                job.status = "error"
                job.error = "Scoring was cancelled."
                raise
            except Exception as exc:
                from server.services.errors import user_message

                # The traceback goes to the log; the phone gets one plain sentence.
                logger.exception("Job %s failed", job.id)
                job.status = "error"
                job.stage = "error"
                job.message = "Scoring failed."
                job.error = user_message(exc)
            finally:
                # The clip is never needed after the run, pass or fail.
                job.discard_source()
                job.touch()

        job.task = asyncio.create_task(guarded())
        return job


_store: Optional[JobStore] = None


def get_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store


def reset_store(store: Optional[JobStore] = None) -> JobStore:
    """Replace the process-wide store. Used by tests and by app shutdown."""
    global _store
    if _store is not None:
        _store.shutdown()
    _store = store
    return get_store()
