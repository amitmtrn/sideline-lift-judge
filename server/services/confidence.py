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
    from server.services.measurement import BarPathMeasurement
    from server.utils.video_meta import VideoInfo

# Ceilings per problem, worst first. A mixed clip is the case the spec calls out
# by name, so it caps hardest.
CAP_MULTIPLE_LIFTS = 0.35
CAP_BAR_NOT_VISIBLE = 0.4
CAP_OFF_AXIS = 0.5
CAP_VERDICT_DISAGREES = 0.5
CAP_OCCLUSION = 0.55
CAP_UNCLEAR_PHASES = 0.6
CAP_SHORT_CLIP = 0.6
CAP_FRAME_EDGE = 0.6
CAP_SMALL_PLATE = 0.65
CAP_NO_TRACE = 0.65
CAP_FEW_FRAMES = 0.7
CAP_SPARSE_TRACE = 0.7

# A window this short cannot hold a whole lift at any sensible frame rate.
MIN_SENSIBLE_DURATION = 1.0
MIN_SENSIBLE_FRAMES = 6
# A trace that saw the plate in fewer sampled frames than this is drawn, but
# the numbers read off it are rough.
MIN_TRACE_COVERAGE = 0.75
# The re-check of the verdict against the tracker (in plate diameters): a path
# called close by the judge but measured this wide, or called poor but measured
# this narrow, earns a warning and a cap. Only when the tracker saw nearly the
# whole lift, and never on a clip the observations say is filmed off-axis,
# where the measured drift is inflated by parallax.
DISAGREE_WIDE_DIAMETERS = 0.5
DISAGREE_NARROW_DIAMETERS = 0.1
DISAGREE_MIN_COVERAGE = 0.9
GOOD_SCORE = 7
POOR_SCORE = 4


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
    measurement: Optional["BarPathMeasurement"] = None,
) -> ClipQuality:
    """
    Turn clip facts and Stage-1 observations into caps plus plain-language
    warnings. `lifts_in_window` above 1 means triage found more than one lift in
    the window we judged; `was_split` means we separated them and scored each on
    its own, which is the good path and only earns a note. `measurement` is what
    the plate tracker made of the window: no trace, or a sparse one, means the
    drift rests on the sampled frames alone.
    """
    quality = ClipQuality()
    text = (observations or "").lower()

    if measurement is not None:
        if not measurement.tracked:
            quality.add(
                "The bar could not be followed through this clip, so the drift is read from the "
                "sampled frames alone.",
                CAP_NO_TRACE,
            )
        else:
            if measurement.coverage < MIN_TRACE_COVERAGE:
                quality.add(
                    f"The bar was followed in only {measurement.coverage:.0%} of the sampled frames, "
                    "so the measured drift is rough.",
                    CAP_SPARSE_TRACE,
                )
            if measurement.near_frame_edge:
                quality.add(
                    "Keep the whole bar and plate in frame for the full lift — the bar left "
                    "the frame here, so the drift is measured from an incomplete trail.",
                    CAP_FRAME_EDGE,
                )
            if measurement.plate_too_small:
                quality.add(
                    "The plate is small on screen — stand closer or zoom in a little so the "
                    "plate is bigger than a thumbnail in frame, and the drift reads cleanly.",
                    CAP_SMALL_PLATE,
                )
            # Plate ellipse aspect gives us a measured oblique signal that
            # does not depend on the vision model noticing it in words.
            # Calibrated against the hand-labelled clips: a perfectly side-on
            # plate measures ~0.998, a visibly tilted camera drops to ~0.91,
            # and anything a coach would call off-axis sits well under 0.9.
            # The threshold leaves a buffer for a slight tilt that still
            # earns the warning — the drift reads inflated at even 20° off.
            axis_ratio = measurement.plate_axis_ratio
            if axis_ratio is not None and axis_ratio < 0.93:
                quality.add(
                    "Film square from the side; this angle may overstate forward drift.",
                    CAP_OFF_AXIS,
                )

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

    # Portrait itself is not a quality problem — most lifters film this way,
    # and a well-framed portrait clip can track the plate at 100%. We only
    # cap confidence when a *measured* quality signal fires: tracker coverage
    # too low, plate wandering out of frame, or the plate too small to fit
    # reliably. Those live on `measurement` above.

    effective_duration = duration_sec or (info.duration_sec if info else 0.0)
    if 0 < effective_duration < MIN_SENSIBLE_DURATION:
        quality.add("The clip is too short to hold a full lift.", CAP_SHORT_CLIP)
    if 0 < frame_count < MIN_SENSIBLE_FRAMES:
        quality.add("Only a few frames could be read from this clip.", CAP_FEW_FRAMES)

    return quality


def cross_check(
    judgment: "LiftJudgment", measurement: Optional["BarPathMeasurement"], quality: ClipQuality
) -> ClipQuality:
    """
    The re-check after the judge replies: does its verdict square with what the
    tracker measured? A judge that calls the path close when the plate was
    measured half a diameter off its line (or poor when it barely moved) has
    leaned on the prose over the number, and the user should know.
    """
    if measurement is None or not measurement.tracked or judgment.score is None:
        return quality
    if measurement.coverage < DISAGREE_MIN_COVERAGE or measurement.max_drift_diameters is None:
        return quality
    if any(warning.startswith("The camera does not look square") for warning in quality.warnings):
        return quality
    drift = measurement.max_drift_diameters
    if judgment.score >= GOOD_SCORE and drift >= DISAGREE_WIDE_DIAMETERS:
        quality.add(
            "The judge called this path close, but the tracker measured the bar "
            f"{drift:.1f} plate diameters off its starting line. Treat the score with care.",
            CAP_VERDICT_DISAGREES,
        )
    elif judgment.score <= POOR_SCORE and drift <= DISAGREE_NARROW_DIAMETERS:
        quality.add(
            "The judge called this path poor, but the tracker measured almost no sideways travel. "
            "Treat the score with care.",
            CAP_VERDICT_DISAGREES,
        )
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
