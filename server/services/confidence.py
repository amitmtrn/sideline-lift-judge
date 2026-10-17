"""
FR-8: angle- and quality-aware confidence.

The judge only ever sees a text description of the clip, so it cannot know that
the clip was shot in portrait, that the window held two lifts, or that the bar
left the frame. Left to itself it returned 0.85 on a clip that mixed a clean and
a jerk (spec §10). This module derives those facts from the clip and from the
Stage-1 observations, caps the confidence accordingly, and produces the warnings
that sit next to the score in the UI.

Caps are ceilings, never floors: a judge that is already unsure stays unsure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

if TYPE_CHECKING:
    from server.schemas.judgment import LiftJudgment
    from server.utils.video_meta import VideoInfo

# Ceilings per problem, worst first. A mixed clip is the case the spec calls out
# by name, so it caps hardest.
CAP_MULTIPLE_LIFTS = 0.35
CAP_BAR_NOT_VISIBLE = 0.4
CAP_OFF_AXIS = 0.5
CAP_OCCLUSION = 0.55
CAP_PORTRAIT = 0.6
CAP_UNCLEAR_PHASES = 0.6
CAP_SHORT_CLIP = 0.6
CAP_FEW_FRAMES = 0.7

# A window this short cannot hold a whole lift at any sensible frame rate.
MIN_SENSIBLE_DURATION = 1.0
MIN_SENSIBLE_FRAMES = 6


def _phrases(*words: str) -> tuple:
    return words


# Matched against the Stage-1 observation text, lowercased. The vision model is
# asked to report angle and visibility problems, so this reads its own words.
OFF_AXIS_PHRASES = _phrases(
    "oblique",
    "off-axis",
    "off axis",
    "angled view",
    "from the front",
    "front view",
    "frontal view",
    "from behind",
    "rear view",
    "three-quarter",
    "diagonal view",
    "not square to the side",
    "not perpendicular",
    "camera is angled",
)
BAR_INVISIBLE_PHRASES = _phrases(
    "bar is not visible",
    "barbell is not visible",
    "bar not visible",
    "cannot see the bar",
    "can't see the bar",
    "bar is out of frame",
    "bar out of frame",
    "no barbell",
    "bar is hidden",
)
OCCLUSION_PHRASES = _phrases(
    "occlud",
    "obscur",
    "partially visible",
    "cut off",
    "out of frame",
    "blocked by",
    "dark",
    "low light",
    "poor lighting",
    "dimly lit",
    "blurry",
    "motion blur",
)
UNCLEAR_PHRASES = _phrases(
    "unclear",
    "hard to tell",
    "difficult to tell",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "ambiguous",
    "not clear",
    "uncertain",
    "phases blend",
    "phases are blended",
    "no distinct",
)
MULTIPLE_LIFT_PHRASES = _phrases(
    "two lifts",
    "multiple lifts",
    "second lift",
    "another lift",
    "clean and jerk",
    "clean & jerk",
    "more than one lift",
    "two attempts",
)


@dataclass
class ClipQuality:
    """Why we do or don't trust this clip, and how far confidence may go."""

    warnings: List[str] = field(default_factory=list)
    caps: List[float] = field(default_factory=list)

    @property
    def confidence_cap(self) -> Optional[float]:
        return min(self.caps) if self.caps else None

    def add(self, warning: str, cap: Optional[float] = None) -> None:
        if warning and warning not in self.warnings:
            self.warnings.append(warning)
        if cap is not None:
            self.caps.append(cap)

    def __bool__(self) -> bool:
        return bool(self.warnings or self.caps)


def _mentions(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def assess_clip_quality(
    *,
    observations: str = "",
    info: Optional["VideoInfo"] = None,
    frame_count: int = 0,
    duration_sec: float = 0.0,
    lifts_in_window: int = 1,
    was_split: bool = False,
    trimmed: bool = False,
) -> ClipQuality:
    """
    Turn clip facts and Stage-1 observations into caps plus plain-language
    warnings. `lifts_in_window` above 1 means triage found more than one lift in
    the window we judged; `was_split` means we separated them and scored each on
    its own, which is the good path and only earns a note.
    """
    quality = ClipQuality()
    text = (observations or "").lower()

    if lifts_in_window > 1 and not was_split:
        quality.add(
            "This clip looks like it holds more than one lift. Trim it to a single lift — "
            "score a clean & jerk as two clips.",
            CAP_MULTIPLE_LIFTS,
        )
    elif _mentions(text, MULTIPLE_LIFT_PHRASES) and not was_split:
        quality.add(
            "The frames may cover more than one lift, so the phases could be blended. "
            "Trim to one lift for a cleaner score.",
            CAP_MULTIPLE_LIFTS,
        )

    if was_split:
        quality.add("This lift was separated out of a longer clip and scored on its own.")

    if _mentions(text, BAR_INVISIBLE_PHRASES):
        quality.add("The bar was not visible in part of the clip.", CAP_BAR_NOT_VISIBLE)
    if _mentions(text, OFF_AXIS_PHRASES):
        quality.add(
            "The camera does not look square to the side, so horizontal drift is hard to measure. "
            "Film from square to the side, 3–5 m away, at hip height.",
            CAP_OFF_AXIS,
        )
    if _mentions(text, OCCLUSION_PHRASES):
        quality.add("Parts of the lift were occluded, dark or blurred.", CAP_OCCLUSION)
    if _mentions(text, UNCLEAR_PHRASES):
        quality.add("The phases of the lift were not clearly separable in these frames.", CAP_UNCLEAR_PHASES)

    if info is not None and info.probed and info.is_portrait:
        quality.add(
            "Filmed in portrait. Landscape keeps the whole bar in frame and scores more reliably.",
            CAP_PORTRAIT,
        )

    effective_duration = duration_sec or (info.duration_sec if info else 0.0)
    if 0 < effective_duration < MIN_SENSIBLE_DURATION:
        quality.add("The clip is too short to hold a full lift.", CAP_SHORT_CLIP)
    if 0 < frame_count < MIN_SENSIBLE_FRAMES:
        quality.add("Only a few frames could be read from this clip.", CAP_FEW_FRAMES)

    return quality


def apply_quality(judgment: "LiftJudgment", quality: ClipQuality) -> "LiftJudgment":
    """
    Return a copy of `judgment` with the cap applied, the warnings attached, and
    the same caveats folded into `notes` so the CLI and raw JSON carry them too.
    """
    updates: dict = {}
    cap = quality.confidence_cap
    if cap is not None:
        current = judgment.confidence
        updates["confidence"] = cap if current is None else min(current, cap)
    if quality.warnings:
        updates["warnings"] = list(quality.warnings)
        updates["notes"] = merge_notes(judgment.notes, quality.warnings)
    return judgment.model_copy(update=updates) if updates else judgment


def merge_notes(notes: Optional[str], warnings: Sequence[str]) -> str:
    """Append warnings to the judge's notes without repeating what it already said."""
    existing = (notes or "").strip()
    extra = [warning for warning in warnings if warning.lower() not in existing.lower()]
    if not extra:
        return existing
    joined = " ".join(_ensure_sentence(warning) for warning in extra)
    return f"{existing} {joined}".strip() if existing else joined


def _ensure_sentence(text: str) -> str:
    text = text.strip()
    if text and not re.search(r"[.!?]$", text):
        return text + "."
    return text
