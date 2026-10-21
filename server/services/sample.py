"""
The cached sample clip: a run scored once, ahead of time, and served back as
if it had just finished. It costs nothing per view, needs no access code and
is not counted against any limit, so a judge who arrives after the daily
budget is spent still sees the whole result screen, replay included.

The sample lives in DEMO_SAMPLE_DIR as `sample.json` plus the replay MP4s the
run rendered. `tools/make_demo_sample.py` builds it. Videos are never
committed (spec §12), so the directory is a deployment artifact, not source.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import time
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


def sample_available() -> bool:
    return load_sample() is not None


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
    The sample from `directory` (default DEMO_SAMPLE_DIR), or None when there
    is none. A replay file that has gone missing drops its URL rather than
    the whole sample: the verdict still shows, just without the video.
    """
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
    payload = body.get("payload") if isinstance(body, dict) else None
    if not isinstance(payload, dict) or not isinstance(payload.get("lifts"), list):
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

    payload = copy.deepcopy(payload)
    for lift in payload["lifts"]:
        if isinstance(lift, dict) and lift.get("annotated_video_url") and lift.get("index") not in replays:
            lift["annotated_video_url"] = None
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
