"""
The cached sample clip: a run scored once, ahead of time, and served back as
if it had just finished. It costs nothing per view, needs no access code and
is not counted against any limit, so a judge who arrives after the daily
budget is spent still sees the whole result screen, replay included.

The sample bundle in `DEMO_SAMPLE_DIR` is just a `sample.json` and the
evidence stills next to it. The replay is **not** committed; the dyno
downloads `SAMPLE_CLIP_URL` on first access, caches it under
`SAMPLE_CACHE_DIR`, and re-renders the annotated replay from the
serialised bar-path points — same renderer the live job uses. This keeps
videos out of the repo and still shows judges the real overlay.

A hosted `SAMPLE_RESULT_URL` (JSON) is also honoured, for the case where
the whole bundle is served from elsewhere; the local path takes priority
so a committed sample always wins.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from server.schemas.judgment import LiftJudgment, LiftType
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
from server.utils.bar_track import BarPath, BarPoint
from server.utils.drive_download import ensure_cached_clip

logger = logging.getLogger(__name__)

SAMPLE_FILE = "sample.json"
SAMPLE_REPLAY_URL = "/api/sample/replay/{index}.mp4"
SAMPLE_STILL_URL = "/api/sample/evidence/{key}.jpg"
SAMPLE_MESSAGE = "Sample clip — scored earlier and cached. Nothing was sent to the model just now."
DEFAULT_CACHE_DIR = Path("/tmp/lift-judge-samples")


@dataclass
class DemoSample:
    payload: Dict[str, Any]
    replays: Dict[int, Path]
    # key "{lift_index}_{cue_id}" -> path to the committed JPG. Served by
    # /api/sample/evidence/{key}.jpg so the sample card renders the evidence
    # strip the same way a live card does.
    stills: Dict[str, Path] = field(default_factory=dict)


def sample_dir() -> Optional[Path]:
    raw = os.getenv("DEMO_SAMPLE_DIR")
    if raw is None or not raw.strip():
        return None
    return Path(raw.strip()).expanduser()


def _sample_result_url() -> str:
    return (os.getenv("SAMPLE_RESULT_URL") or "").strip()


def _sample_clip_url() -> str:
    return (os.getenv("SAMPLE_CLIP_URL") or "").strip()


def _sample_cache_dir() -> Path:
    raw = os.getenv("SAMPLE_CACHE_DIR")
    if raw and raw.strip():
        return Path(raw.strip()).expanduser()
    return DEFAULT_CACHE_DIR


def sample_available() -> bool:
    return load_sample() is not None


# Caches keyed per-URL/per-path, so a page view does not re-fetch or re-render,
# and so an unset env resets back to None rather than returning a stale sample.
_REMOTE_CACHE: Dict[str, Optional["DemoSample"]] = {}
_RENDER_LOCK = threading.Lock()


def reset_remote_cache_for_tests() -> None:
    """Clears both the remote-JSON cache and the lazy render guard."""
    _REMOTE_CACHE.clear()


# ---------- bar-path serialisation ----------
#
# The sample bundle has to carry the per-frame tracker points, not just the
# summary measurement, so the dyno can re-render the trace without re-running
# the tracker. The schema is intentionally plain: a list of [t, x, y] triples
# plus the few scalars `BarPath` needs to reconstruct.


def bar_path_to_dict(path: Optional[BarPath]) -> Optional[Dict[str, Any]]:
    if path is None or not path.points:
        return None
    return {
        "points": [[round(p.t, 4), round(p.x, 5), round(p.y, 5)] for p in path.points],
        "plate_radius": path.plate_radius,
        "coverage": path.coverage,
        "sampled": path.sampled,
        "frame_width": path.frame_width,
        "frame_height": path.frame_height,
        "reason": path.reason,
    }


def bar_path_from_dict(data: Any) -> Optional[BarPath]:
    if not isinstance(data, dict):
        return None
    raw_points = data.get("points") or []
    try:
        points = [
            BarPoint(t=float(row[0]), x=float(row[1]), y=float(row[2]))
            for row in raw_points
            if isinstance(row, (list, tuple)) and len(row) >= 3
        ]
    except (TypeError, ValueError):
        return None
    try:
        return BarPath(
            points=points,
            plate_radius=float(data.get("plate_radius", 0.0) or 0.0),
            coverage=float(data.get("coverage", 0.0) or 0.0),
            sampled=int(data.get("sampled", 0) or 0),
            frame_width=int(data.get("frame_width", 0) or 0),
            frame_height=int(data.get("frame_height", 0) or 0),
            reason=str(data.get("reason", "") or ""),
        )
    except (TypeError, ValueError):
        return None


# ---------- loading ----------


def _apply_clip_url_fallback(payload: Dict[str, Any]) -> None:
    """
    If SAMPLE_CLIP_URL is a direct video URL (not a Drive link we download), and
    no local replay rendered, use the URL as-is. We only reach this fallback
    when the lazy render failed; the Drive case normally hands back a local
    `/api/sample/replay/...` URL instead.
    """
    url = _sample_clip_url()
    if not url:
        return
    # Only forward URLs that a browser can play back directly. Drive download
    # links are handled by the lazy render; forwarding them loses Range support
    # and iOS stalls at the first seek.
    if "drive.google.com" in url or "drive.usercontent.google.com" in url:
        return
    for lift in payload.get("lifts") or []:
        if isinstance(lift, dict) and not lift.get("annotated_video_url"):
            lift["annotated_video_url"] = url


def _parse_sample_body(body: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(body, dict):
        return None
    payload = body.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("lifts"), list):
        return None
    return copy.deepcopy(payload)


def _finalise_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["sample"] = True
    payload.setdefault("status", "done")
    payload.setdefault("message", SAMPLE_MESSAGE)
    return payload


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
    for lift in payload["lifts"]:
        if isinstance(lift, dict):
            lift["annotated_video_url"] = None
    _apply_clip_url_fallback(payload)
    sample = DemoSample(payload=_finalise_payload(payload), replays={})
    _REMOTE_CACHE[url] = sample
    return sample


def _render_replay_for_sample(
    *,
    clip_path: Path,
    cache_dir: Path,
    index: int,
    judgment: LiftJudgment,
    bar_path: Optional[BarPath],
    start_sec: Optional[float],
    end_sec: Optional[float],
) -> Optional[Path]:
    """Call the production renderer with the stored bar_path, cache the MP4."""
    from server.services.replay import render_annotated_replay

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"sample_replay_{index}.mp4"
    if out.exists() and out.stat().st_size > 1024:
        return out
    with _RENDER_LOCK:
        if out.exists() and out.stat().st_size > 1024:
            return out
        written, _annotated = render_annotated_replay(
            str(clip_path),
            judgment,
            str(out),
            start_sec=start_sec,
            end_sec=end_sec,
            bar_path=bar_path,
        )
    if not written:
        return None
    return Path(written)


def _materialise_replays(
    *,
    body: Dict[str, Any],
    payload: Dict[str, Any],
    directory: Optional[Path],
) -> Dict[int, Path]:
    """
    Rebuild the per-lift replay files from (a) MP4s already next to the JSON
    (legacy), (b) a cached render we did earlier this process, (c) a lazy
    render from the downloaded clip and the serialised bar path. Returns the
    index -> local-path map the file endpoint serves from.
    """
    replays: Dict[int, Path] = {}

    # (a) MP4s next to the committed JSON — legacy sample bundles.
    if directory is not None:
        for key, name in (body.get("replays") or {}).items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            file = Path(directory) / str(name)
            if file.is_file():
                replays[index] = file

    # (b)/(c) Render from the downloaded clip whenever SAMPLE_CLIP_URL is set.
    clip_url = _sample_clip_url()
    if clip_url:
        cache_dir = _sample_cache_dir()
        clip_cache = cache_dir / "sample_clip.mp4"
        clip_local = ensure_cached_clip(clip_url, clip_cache)
        if clip_local is None:
            logger.warning("Sample clip download failed; serving verdict without replay")
            return replays
        lifts = payload.get("lifts") or []
        if not lifts:
            return replays
        bar_path_payload = payload.get("sample_bar_path")
        bar_path = bar_path_from_dict(bar_path_payload)
        first = lifts[0]
        if not isinstance(first, dict):
            return replays
        index = int(first.get("index") or 1)
        judgment_dict = {k: first.get(k) for k in LiftJudgment.model_fields if k in first}
        try:
            judgment = LiftJudgment(**judgment_dict)
        except Exception as exc:
            logger.warning("Sample JSON cannot be re-judged for replay rendering: %s", exc)
            return replays
        rendered = _render_replay_for_sample(
            clip_path=clip_local,
            cache_dir=cache_dir,
            index=index,
            judgment=judgment,
            bar_path=bar_path,
            start_sec=float(first.get("start_sec") or 0.0) if first.get("start_sec") is not None else None,
            end_sec=float(first.get("end_sec") or 0.0) if first.get("end_sec") is not None else None,
        )
        if rendered is not None:
            replays[index] = rendered
    return replays


def load_sample(directory: Optional[Path] = None) -> Optional[DemoSample]:
    """
    The sample from the local directory (default DEMO_SAMPLE_DIR), falling back
    to SAMPLE_RESULT_URL. A replay that could not be rendered drops its URL
    rather than the whole sample: the verdict still shows, just without video.
    """
    directory = directory if directory is not None else sample_dir()
    body: Dict[str, Any]
    if directory is not None:
        path = Path(directory) / SAMPLE_FILE
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            body = None  # type: ignore[assignment]
        except (OSError, ValueError) as exc:
            logger.warning("Sample at %s could not be read: %s", path, exc)
            return None
    else:
        body = None  # type: ignore[assignment]

    if body is None:
        return _load_remote_sample()

    payload = _parse_sample_body(body)
    if payload is None:
        logger.warning("Sample at %s has no lifts", directory)
        return None

    replays = _materialise_replays(body=body, payload=payload, directory=directory)
    stills = _materialise_stills(payload=payload, directory=directory)
    for lift in payload["lifts"]:
        if isinstance(lift, dict):
            index = lift.get("index")
            if lift.get("annotated_video_url") and index not in replays:
                lift["annotated_video_url"] = None
            elif index in replays and not lift.get("annotated_video_url"):
                lift["annotated_video_url"] = SAMPLE_REPLAY_URL.format(index=index)
    _apply_clip_url_fallback(payload)
    return DemoSample(payload=_finalise_payload(payload), replays=replays, stills=stills)


def _materialise_stills(
    *,
    payload: Dict[str, Any],
    directory: Optional[Path],
) -> Dict[str, Path]:
    """
    Match each evidenced cue to a committed JPG next to the sample JSON and
    inject the serving URL into `payload['lifts'][*]['evidenced_cues'][*].media`.
    The file naming convention mirrors what the pipeline writes
    (`still_{lift_index}_{cue_id}.jpg`), so an existing bundle built with
    `tools/make_demo_sample.py` lights up without having to rename anything.
    """
    stills: Dict[str, Path] = {}
    if directory is None:
        return stills
    for lift in payload.get("lifts") or []:
        if not isinstance(lift, dict):
            continue
        index = lift.get("index")
        if index is None:
            continue
        cues = lift.get("evidenced_cues") or []
        for ec in cues:
            if not isinstance(ec, dict):
                continue
            cue_id = ec.get("id")
            if not cue_id:
                continue
            jpg = Path(directory) / f"still_{index}_{cue_id}.jpg"
            if not jpg.is_file():
                continue
            key = f"{index}_{cue_id}"
            stills[key] = jpg
            media = ec.get("media") or {}
            if not isinstance(media, dict):
                media = {}
            # Default to a strip layout when the pipeline did not record one,
            # so an older bundle still renders as a strip (what the renderer
            # writes).
            media.setdefault("layout", "strip")
            media["still_url"] = SAMPLE_STILL_URL.format(key=key)
            ec["media"] = media
    return stills


# ---------- writing (used by the admin build-sample tool) ----------


def write_sample(outcome: ScoreOutcome, out_dir: Path, *, source_name: str = "") -> Path:
    """
    Serialise a finished run into `out_dir`. Replays go **next to** the JSON
    (`sample.json`); the committed bundle is JSON + evidence stills only. The
    MP4 the pipeline rendered stays on disk for local testing but is not part
    of the shippable bundle — videos are not committed to the repo (spec §12),
    and the live dyno re-renders at /api/sample time from the Drive clip and
    the serialised bar-path points added below.
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
    # Serialise the first lift's bar-path points so the dyno can re-render the
    # annotated replay from a downloaded Drive clip without re-running tracker
    # or model. Only the first lift is kept — the sample is a single lift.
    first = next((item for item in outcome.lifts if item.replay_path), None) or (
        outcome.lifts[0] if outcome.lifts else None
    )
    serialised_bar = bar_path_to_dict(getattr(first, "bar_path", None)) if first is not None else None
    if serialised_bar is not None:
        payload["sample_bar_path"] = serialised_bar

    body = {
        "version": 2,
        "source": source_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "replays": replays,
        "payload": payload,
    }
    target = out_dir / SAMPLE_FILE
    target.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


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
