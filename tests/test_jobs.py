"""Job store: polling state, TTL cleanup and the privacy promise (§12)."""
import asyncio
import time

import pytest

from server.services.jobs import DEFAULT_TTL_SECONDS, Job, JobStore, job_ttl_seconds


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store(tmp_path):
    store = JobStore(root=tmp_path / "jobs")
    yield store
    store.shutdown()


def test_create_makes_an_isolated_workdir(store):
    first = store.create()
    second = store.create()
    assert first.id != second.id
    assert first.workdir.is_dir() and second.workdir.is_dir()
    assert first.workdir != second.workdir
    assert store.get(first.id) is first


def test_unknown_job_is_none(store):
    assert store.get("nope") is None


def test_progress_never_runs_backwards(store):
    job = store.create()
    job.set_progress("judging", "Judging…", 0.6)
    job.set_progress("observing", "Reading…", 0.2)
    assert job.progress == 0.6
    assert job.stage == "observing"  # the label follows, the bar does not


def test_progress_is_clamped(store):
    job = store.create()
    job.set_progress("rendering", "…", 9.0)
    assert job.progress == 1.0


def test_as_dict_merges_the_payload(store):
    job = store.create()
    job.payload = {"lifts": [{"lift": "snatch"}]}
    job.status = "done"
    body = job.as_dict()
    assert body["id"] == job.id and body["status"] == "done"
    assert body["lifts"] == [{"lift": "snatch"}]
    assert "error" not in body


def test_discard_source_removes_the_upload(store):
    job = store.create()
    upload = job.workdir / "clip.mp4"
    upload.write_bytes(b"x" * 32)
    job.source_path = upload
    job.discard_source()
    assert not upload.exists()
    assert job.source_path is None
    job.discard_source()  # idempotent


def test_finished_jobs_expire_and_take_their_files(store):
    job = store.create()
    (job.workdir / "replay.mp4").write_bytes(b"x" * 32)
    job.status = "done"
    job.updated_at = time.time() - 10_000
    assert store.purge_expired(ttl=60) == 1
    assert store.get(job.id) is None
    assert not job.workdir.exists()


def test_running_jobs_are_never_purged(store):
    job = store.create()
    job.status = "running"
    job.updated_at = time.time() - 10_000
    assert store.purge_expired(ttl=60) == 0
    assert store.get(job.id) is job


def test_store_drops_the_oldest_when_over_capacity(tmp_path):
    store = JobStore(root=tmp_path / "jobs", max_jobs=2)
    try:
        first = store.create()
        first.status = "done"
        second = store.create()
        second.status = "done"
        third = store.create()
        assert store.get(first.id) is None
        assert store.get(third.id) is third
    finally:
        store.shutdown()


def test_start_runs_the_job_and_marks_it_done(store):
    job = store.create()
    seen = []

    async def runner(current: Job) -> None:
        current.set_progress("judging", "Judging…", 0.5)
        seen.append(current.id)

    async def drive():
        store.start(job, runner)
        await job.task

    run(drive())
    assert seen == [job.id]
    assert job.status == "done"
    assert job.progress == 1.0


def test_start_records_a_failure_instead_of_raising(store):
    job = store.create()
    upload = job.workdir / "clip.mp4"
    upload.write_bytes(b"x")
    job.source_path = upload

    async def runner(_: Job) -> None:
        raise RuntimeError("upstream exploded")

    async def drive():
        store.start(job, runner)
        await job.task

    run(drive())
    assert job.status == "error"
    # The phone gets a plain sentence, never the exception's text.
    assert job.error and "upstream exploded" not in job.error
    # The clip is deleted whether the run succeeded or failed.
    assert not upload.exists()


def test_shutdown_clears_everything(tmp_path):
    store = JobStore(root=tmp_path / "jobs")
    job = store.create()
    store.shutdown()
    assert not job.workdir.exists()
    assert store.get(job.id) is None


def test_ttl_from_env(monkeypatch):
    monkeypatch.delenv("JOB_TTL_SECONDS", raising=False)
    assert job_ttl_seconds() == DEFAULT_TTL_SECONDS
    monkeypatch.setenv("JOB_TTL_SECONDS", "120")
    assert job_ttl_seconds() == 120
    monkeypatch.setenv("JOB_TTL_SECONDS", "junk")
    assert job_ttl_seconds() == DEFAULT_TTL_SECONDS
