"""
The cached sample clip: a run scored once, ahead of time, and served back as
if it had just finished. It costs nothing per view, needs no access code and
is not counted against any limit, so a judge who arrives after the daily
budget is spent still sees the whole result screen, replay included.

The sample can come from either source, in this order:
  1. SAMPLE_RESULT_URL: a hosted sample.json fetched once per process and
     cached. SAMPLE_CLIP_URL, when set, is substituted into every lift's
     annotated_video_url so the replay plays from the same hosting.
  2. DEMO_SAMPLE_DIR: a local directory with `sample.json` plus optional
     replay MP4s next to it. `tools/make_demo_sample.py` builds that.

The remote path exists so a deploy with the model credential but no committed
sample video can still serve a replay (videos are not committed — spec §12).
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from server.schemas.judgment import LiftType
from server.services.nebius_client import NebiusChatClient
from server.services.pipeline import (
    DEFAULT_FRAMES,
    ProgressReporter,
    ScoreOutcome,
    ScoreRequest,
    clamp_frames,
    run_scoring,
)
from server.services.results import outcome_payload

logger = logging.getLogger(__name__)

SAMPLE_FILE = "sample.json"
SAMPLE_REPLAY_URL = "/api/sample/replay/{index}.mp4"
SAMPLE_MESSAGE = "Sample clip — scored earlier and cached. Nothing was sent to the model just now."


@dataclass
class DemoSample:
    payload: Dict[str, Any]
    replays: Dict[int, Path]


def sample_dir() -> Optional[Path]:
    raw = os.getenv("DEMO_SAMPLE_DIR")
    if raw is None or not raw.strip():
        return None
    return Path(raw.strip()).expanduser()


def _sample_result_url() -> str:
    return (os.getenv("SAMPLE_RESULT_URL") or "").strip()


def _sample_clip_url() -> str:
    return (os.getenv("SAMPLE_CLIP_URL") or "").strip()


def sample_available() -> bool:
    return load_sample() is not None


# Cache the remote JSON for the lifetime of the process keyed by URL, so a
# page view does not re-fetch — and so an unset env resets back to None.
_REMOTE_CACHE: Dict[str, Optional["DemoSample"]] = {}


def reset_remote_cache_for_tests() -> None:
    _REMOTE_CACHE.clear()


def _apply_clip_url(payload: Dict[str, Any]) -> None:
    """Point every lift's replay at SAMPLE_CLIP_URL when it is set."""
    url = _sample_clip_url()
    if not url:
        return
    for lift in payload.get("lifts") or []:
        if isinstance(lift, dict):
            lift["annotated_video_url"] = url


def _parse_sample_body(body: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(body, dict):
        return None
    payload = body.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("lifts"), list):
        return None
    return copy.deepcopy(payload)


def _load_remote_sample() -> Optional["DemoSample"]:
    url = _sample_result_url()
    if not url:
        return None
    if url in _REMOTE_CACHE:
        return _REMOTE_CACHE[url]
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            raw = response.read()
    except Exception as exc:
        logger.warning("Could not fetch SAMPLE_RESULT_URL: %s", exc)
        _REMOTE_CACHE[url] = None
        return None
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        logger.warning("SAMPLE_RESULT_URL did not return JSON: %s", exc)
        _REMOTE_CACHE[url] = None
        return None
    payload = _parse_sample_body(body)
    if payload is None:
        logger.warning("SAMPLE_RESULT_URL JSON has no lifts; ignoring")
        _REMOTE_CACHE[url] = None
        return None
    _apply_clip_url(payload)
    # The remote path drops local-replay URLs — the file endpoint has nothing
    # to serve. SAMPLE_CLIP_URL is the way to put a video back.
    if not _sample_clip_url():
        for lift in payload["lifts"]:
            if isinstance(lift, dict):
                lift["annotated_video_url"] = None
    payload["sample"] = True
    payload.setdefault("status", "done")
    payload.setdefault("message", SAMPLE_MESSAGE)
    sample = DemoSample(payload=payload, replays={})
    _REMOTE_CACHE[url] = sample
    return sample


def write_sample(outcome: ScoreOutcome, out_dir: Path, *, source_name: str = "") -> Path:
    """
    Serialise a finished run into `out_dir` next to the replays it rendered.
    A replay rendered elsewhere is copied in, so the directory stands alone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    replays: Dict[str, str] = {}
    for item in outcome.lifts:
        if not item.replay_path:
            continue
        path = Path(item.replay_path)
        if path.resolve().parent != out_dir.resolve():
            target = out_dir / path.name
            shutil.copyfile(path, target)
            path = target
        replays[str(item.index)] = path.name

    payload = outcome_payload(outcome, replay_url=lambda index: SAMPLE_REPLAY_URL.format(index=index))
    payload.update(
        {
            "status": "done",
            "stage": "done",
            "message": SAMPLE_MESSAGE,
            "progress": 1.0,
            "sample": True,
        }
    )
    body = {
        "version": 1,
        "source": source_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "replays": replays,
        "payload": payload,
    }
    target = out_dir / SAMPLE_FILE
    target.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_sample(directory: Optional[Path] = None) -> Optional[DemoSample]:
    """
    The sample from the remote URL (SAMPLE_RESULT_URL, cached per process) or
    `directory` (default DEMO_SAMPLE_DIR), or None when neither is reachable.
    A replay file that has gone missing drops its URL rather than the whole
    sample: the verdict still shows, just without the video.
    """
    if directory is None:
        remote = _load_remote_sample()
        if remote is not None:
            return remote
    directory = directory if directory is not None else sample_dir()
    if directory is None:
        return None
    path = Path(directory) / SAMPLE_FILE
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Sample at %s could not be read: %s", path, exc)
        return None
    payload = _parse_sample_body(body)
    if payload is None:
        logger.warning("Sample at %s has no lifts", path)
        return None

    replays: Dict[int, Path] = {}
    for key, name in (body.get("replays") or {}).items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        file = Path(directory) / str(name)
        if file.is_file():
            replays[index] = file

    for lift in payload["lifts"]:
        if isinstance(lift, dict) and lift.get("annotated_video_url") and lift.get("index") not in replays:
            lift["annotated_video_url"] = None
    _apply_clip_url(payload)
    payload["sample"] = True
    payload.setdefault("status", "done")
    payload.setdefault("message", SAMPLE_MESSAGE)
    return DemoSample(payload=payload, replays=replays)


async def make_sample(
    *,
    video_path: str,
    out_dir: Path,
    client: NebiusChatClient,
    prompts_dir: str,
    lift: Optional[LiftType] = None,
    mode: str = "single",
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
    frames: int = DEFAULT_FRAMES,
    progress: Optional[ProgressReporter] = None,
) -> ScoreOutcome:
    """Score `video_path` once, with replays, and write the sample into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    request = ScoreRequest(
        video_path=str(video_path),
        lift=lift,
        trim_start=trim_start,
        trim_end=trim_end,
        frames=clamp_frames(frames),
        mode="auto" if mode == "auto" else "single",
        render_replay=True,
        replay_dir=str(out_dir),
    )
    kwargs = {"progress": progress} if progress is not None else {}
    outcome = await run_scoring(request, client=client, prompts_dir=prompts_dir, **kwargs)
    if not outcome.scored:
        first_error = next((item.error for item in outcome.lifts if item.error), None)
        raise RuntimeError(first_error or "No snatch, clean or jerk could be scored in this clip.")
    write_sample(outcome, out_dir, source_name=Path(video_path).name)
    return outcome
