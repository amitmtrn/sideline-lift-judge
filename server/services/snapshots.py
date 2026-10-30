"""
The lift, frozen at the moment the athlete pays.

A job and its evidence stills live for thirty minutes in memory and on a
dyno disk that is wiped on restart; a review is paid for and may take two
days. So at checkout the lift's verdict (the same JSON the card was drawn
from) and up to four of its evidence images are copied into the database,
the images re-encoded small enough to live in a row. The replay MP4 is not
kept: it is large, and the keyframes carry the evidence.
"""
from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
from sqlalchemy.orm import Session

from server.services.db import LiftSnapshot, SnapshotImage

logger = logging.getLogger(__name__)

MAX_KEYFRAMES = 4
MAX_LONG_SIDE = 1024
JPEG_QUALITY = 82
# A phrase in a warning that marks it as the camera-angle one.
_ANGLE_RE = re.compile(r"angle|oblique|off-axis|off axis|square to the side|not square", re.IGNORECASE)
# Fields the snapshot keeps from the lift payload. URLs that point at the
# dying job are dropped; everything the coach reads stays.
_DROPPED_KEYS = ("annotated_video_url", "observations")


class SnapshotError(Exception):
    def __init__(self, detail: str, *, status: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status


# The lift as a key ("snatch") and as words ("Snatch"). A live job's payload
# carries the LiftType enum, whose str() is "LiftType.snatch"; the sample's
# JSON carries the plain value. Both land here as the key, and the key is
# the only thing ever stored or compared.
LIFT_LABELS = {
    "snatch": "Snatch",
    "clean": "Clean",
    "jerk": "Jerk",
    "clean_jerk": "Clean & Jerk",
    "clean_and_jerk": "Clean & Jerk",
}


def lift_key(value: Any) -> str:
    raw = getattr(value, "value", value)
    text = str(raw or "").strip().lower()
    if text.startswith("lifttype."):
        text = text[len("lifttype.") :]
    text = text.replace("&", "and").replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "snatch"


def lift_label(value: Any) -> str:
    key = lift_key(value)
    return LIFT_LABELS.get(key, key.replace("_", " ").title())


def shrink_jpeg(data: bytes, *, max_long_side: int = MAX_LONG_SIDE, quality: int = JPEG_QUALITY) -> Optional[bytes]:
    """Re-encode an image small enough for a database row; None if it does not decode."""
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_long_side:
        scale = max_long_side / float(longest)
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return bytes(encoded.tobytes())


def angle_warning_of(lift: Dict[str, Any]) -> Optional[str]:
    for warning in lift.get("warnings") or []:
        if isinstance(warning, str) and _ANGLE_RE.search(warning):
            return warning
    return None


def pick_lift(payload: Dict[str, Any], lift_index: Optional[int]) -> Dict[str, Any]:
    """The scored lift with `lift_index` (or the first scored one) out of a result payload."""
    lifts = [item for item in payload.get("lifts") or [] if isinstance(item, dict)]
    if lift_index is not None:
        for item in lifts:
            if item.get("index") == lift_index:
                if not item.get("band"):
                    raise SnapshotError("That lift was not scored, so there is nothing for a coach to review.")
                return item
        raise SnapshotError("No lift with that index in this result.", status=404)
    for item in lifts:
        if item.get("band"):
            return item
    raise SnapshotError("This result holds no scored lift.", status=404)


def strip_lift(lift: Dict[str, Any]) -> Dict[str, Any]:
    """The lift payload without the job-bound URLs and the long vision transcript."""
    body = copy.deepcopy(lift)
    for key in _DROPPED_KEYS:
        body.pop(key, None)
    body.pop("replay_annotated", None)
    for cue in body.get("evidenced_cues") or []:
        if isinstance(cue, dict) and isinstance(cue.get("media"), dict):
            cue["media"] = {k: v for k, v in cue["media"].items() if k == "layout"}
    return body


def keyframes_for_lift(lift: Dict[str, Any], stills: Dict[str, Path]) -> List[Tuple[str, bytes]]:
    """
    The evidence images for one lift, in cue order, as (label, jpeg bytes).
    `stills` is the job's or the sample's "{index}_{cue_id}" → path map.
    """
    index = lift.get("index")
    out: List[Tuple[str, bytes]] = []
    for cue in lift.get("evidenced_cues") or []:
        if not isinstance(cue, dict):
            continue
        key = f"{index}_{cue.get('id')}"
        path = stills.get(key)
        if path is None or not Path(path).is_file():
            continue
        try:
            raw = Path(path).read_bytes()
        except OSError:
            continue
        small = shrink_jpeg(raw)
        if small is None:
            continue
        out.append((str(cue.get("text") or cue.get("phase") or "evidence")[:80], small))
        if len(out) >= MAX_KEYFRAMES:
            break
    return out


def create_snapshot(
    session: Session,
    *,
    source_kind: str,
    source_job_id: Optional[str],
    lift: Dict[str, Any],
    keyframes: Sequence[Tuple[str, bytes]],
) -> LiftSnapshot:
    from server.services.db import dumps

    body = strip_lift(lift)
    snapshot = LiftSnapshot(
        source_kind=source_kind,
        source_job_id=source_job_id,
        lift_index=int(lift.get("index") or 1),
        lift_type=lift_key(lift.get("lift")),
        judgment_json=dumps(body),
        measurement_json=dumps(lift.get("bar_path")) if lift.get("bar_path") is not None else None,
        angle_warning=angle_warning_of(lift),
    )
    for position, (label, data) in enumerate(list(keyframes)[:MAX_KEYFRAMES]):
        snapshot.images.append(SnapshotImage(position=position, label=label, data=data))
    session.add(snapshot)
    session.flush()
    return snapshot
