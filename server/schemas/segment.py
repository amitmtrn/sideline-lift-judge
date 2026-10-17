from __future__ import annotations

from collections import Counter
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from server.schemas.judgment import LiftJudgment, LiftType
from server.schemas.json_parse import parse_model_json

# Where the bar is, ordered low to high. Asking the vision model to name a lift
# makes it answer with the lift the scene reminds it of; asking where the bar is
# in each frame is a question it can actually answer from pixels, so the lift is
# derived here instead.
FLOOR, KNEES, SHOULDERS, OVERHEAD = 0, 1, 2, 3
NOT_VISIBLE = -1

_BAR_LEVELS = {
    "floor": FLOOR,
    "ground": FLOOR,
    "platform": FLOOR,
    "shin": FLOOR,
    "knees": KNEES,
    "knee": KNEES,
    "hang": KNEES,
    "hip": KNEES,
    "hips": KNEES,
    "thigh": KNEES,
    "thighs": KNEES,
    "waist": KNEES,
    "shoulders": SHOULDERS,
    "shoulder": SHOULDERS,
    "chest": SHOULDERS,
    "chin": SHOULDERS,
    "rack": SHOULDERS,
    "front rack": SHOULDERS,
    "overhead": OVERHEAD,
    "head": OVERHEAD,
    "above head": OVERHEAD,
    "locked out": OVERHEAD,
    "lockout": OVERHEAD,
}

# A level counts as actually reached only if the model reports it in at least
# this many frames; a single stray frame should not turn a clean into a jerk.
SUSTAINED_FRAMES = 2


def bar_level(name: object) -> int:
    """Map a free-text bar position onto a level, or NOT_VISIBLE."""
    if not isinstance(name, str):
        return NOT_VISIBLE
    text = name.strip().lower().replace("_", " ")
    if text in _BAR_LEVELS:
        return _BAR_LEVELS[text]
    for key, level in _BAR_LEVELS.items():
        if key in text:
            return level
    return NOT_VISIBLE


class SegmentTriage(BaseModel):
    """
    Stage 0b: what the vision model saw in a candidate window.

    The model reports where the bar sits in each frame; `lifts` is derived from
    that sequence. It is ordered and may hold two entries for a clean & jerk,
    which the spec scores as two separate judgments (clean, then jerk).
    """

    lifts: List[LiftType] = Field(default_factory=list)
    bar_by_frame: List[str] = Field(default_factory=list)
    athlete_lifting: Optional[bool] = Field(default=None)
    activity: str = Field(default="")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: Optional[str] = Field(default=None)

    @field_validator("bar_by_frame", mode="before")
    @classmethod
    def coerce_frames(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return []
        return [str(item) for item in value if item is not None]

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value: object) -> object:
        # Some replies answer in percent, some with a word.
        if isinstance(value, str):
            text = value.strip().rstrip("%")
            try:
                value = float(text)
            except ValueError:
                return None
        if isinstance(value, (int, float)) and value > 1:
            return min(1.0, float(value) / 100.0)
        return value

    @field_validator("lifts", mode="before")
    @classmethod
    def coerce_lifts(cls, value: object) -> object:
        """
        Vision models answer loosely: a bare string, "none", "clean and jerk",
        or a list with nulls. Normalize before the enum check rejects the lot.
        """
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return value

        out: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            name = item.strip().lower()
            if name in ("clean and jerk", "clean & jerk", "clean_and_jerk", "cleanandjerk"):
                out.extend(["clean", "jerk"])
                continue
            if name in ("", "none", "null", "no lift", "n/a", "other", "unknown"):
                continue
            out.append(name)
        # Drop anything outside the three scored lifts (deadlift, squat, press, ...)
        return [name for name in out if name in (e.value for e in LiftType)]

    @model_validator(mode="after")
    def derive_lifts_from_bar_path(self) -> "SegmentTriage":
        """
        The frame-by-frame bar path wins over any lift the model named itself,
        because naming is where it guesses. Falls back to whatever it named only
        when the path is unreadable.
        """
        if self.athlete_lifting is False:
            self.lifts = []
            return self
        derived = lifts_from_bar_path(self.bar_by_frame)
        if derived is not None:
            self.lifts = derived
        return self

    @property
    def is_lift(self) -> bool:
        return bool(self.lifts)


def lifts_from_bar_path(bar_by_frame: List[str]) -> Optional[List[LiftType]]:
    """
    Classify a window from the bar's height in each frame.

    Returns None when the path is too short or too occluded to decide, an empty
    list when nothing was lifted (the bar is being lowered, dropped or idle, or
    never rose past the hips), otherwise the lifts in the order they happen.
    """
    levels = [level for level in (bar_level(name) for name in bar_by_frame) if level != NOT_VISIBLE]
    if len(levels) < 3:
        return None

    counts = Counter(levels)
    start = _dominant(levels[: max(1, len(levels) // 3)])
    # One stray frame of a level does not mean the bar went there.
    sustained = [level for level, count in counts.items() if count >= SUSTAINED_FRAMES]
    high = max(sustained) if sustained else max(levels)

    if high <= start:
        # Nothing rose: a bar being lowered or dropped, or an idle window.
        return []
    if start <= KNEES:
        if high >= OVERHEAD:
            # Racked on the shoulders on the way up: a clean, then a jerk.
            return [LiftType.clean, LiftType.jerk] if _racked_on_the_way_up(levels) else [LiftType.snatch]
        if high >= SHOULDERS:
            return [LiftType.clean]
        return []  # never got above the hips: a pull or a deadlift
    if start == SHOULDERS and high >= OVERHEAD:
        return [LiftType.jerk]
    return []


def _dominant(levels: List[int]) -> int:
    """Most common level; ties go to the lower one (the bar starts low)."""
    counts = Counter(levels)
    best = max(counts.values())
    return min(level for level, count in counts.items() if count == best)


def _racked_on_the_way_up(levels: List[int]) -> bool:
    """True when the bar sat on the shoulders between leaving the floor and going overhead."""
    try:
        first_overhead = levels.index(OVERHEAD)
    except ValueError:
        return False
    before = levels[:first_overhead]
    return sum(1 for level in before if level == SHOULDERS) >= SUSTAINED_FRAMES


def parse_triage_json(text: str) -> SegmentTriage:
    return parse_model_json(text, SegmentTriage, what="segment triage JSON object")


class ExerciseReport(BaseModel):
    """One scored exercise inside a session clip."""

    index: int
    # The candidate window this came from. A clean & jerk splits into two
    # exercises that share one window, so the two numbering schemes differ.
    segment_index: int = 0
    start_sec: float
    end_sec: float
    lift: LiftType
    activity: str = Field(default="")
    triage_confidence: Optional[float] = Field(default=None)
    judgment: Optional[LiftJudgment] = Field(default=None)
    error: Optional[str] = Field(default=None)
    clip_path: Optional[str] = Field(default=None)
    frames_dir: Optional[str] = Field(default=None)
    observations: Optional[str] = Field(default=None)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


class SkippedSegment(BaseModel):
    """A candidate window the triage step rejected as not a scored lift."""

    index: int
    start_sec: float
    end_sec: float
    activity: str = Field(default="")
    reason: str = Field(default="")


class SessionReport(BaseModel):
    source: str
    duration_sec: float
    candidate_segments: int
    exercises: List[ExerciseReport] = Field(default_factory=list)
    skipped: List[SkippedSegment] = Field(default_factory=list)

    def ranked(self) -> List[ExerciseReport]:
        """
        Best first. Unscored exercises (errors, band "unknown") sort last, and
        ties keep clip order so the table stays stable between runs.
        """

        def key(item: ExerciseReport) -> tuple:
            score = item.judgment.score if item.judgment else None
            confidence = (item.judgment.confidence if item.judgment else None) or 0.0
            return (0 if score is not None else 1, -(score or 0), -confidence, item.start_sec)

        return sorted(self.exercises, key=key)
