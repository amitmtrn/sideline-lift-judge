"""
What the plate tracker measured, as numbers for the API and as words for the
judge.

The judge (Nemotron) never sees pixels. Until now it only read the vision
model's prose, so its drift call could contradict the trace drawn on the replay.
This turns the tracked path into a second, independent piece of evidence —
how far the bar strayed, where in the rise, how much of the lift was seen — in
plate diameters, the one length whose real size is known in every clip.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from server.utils.bar_track import BarPath, PLATE_DIAMETER_M

# Rule of thumb for a clip filmed square to the side, in plate diameters of a
# 450 mm bumper: under CLOSE (~7 cm) the path is close to the body; above LARGE
# (~16 cm) the bar is well away from it. Offered to the judge as a yardstick,
# not a rule — the rubric still decides.
CLOSE_DRIFT_DIAMETERS = 0.15
LARGE_DRIFT_DIAMETERS = 0.35


@dataclass
class BarPathMeasurement:
    tracked: bool
    coverage: float = 0.0
    points: int = 0
    max_drift_diameters: Optional[float] = None
    max_drift_cm: Optional[float] = None
    drift_at_rise_fraction: Optional[float] = None  # 0 = where the bar started, 1 = the top
    sideways_range_diameters: Optional[float] = None
    rise_diameters: Optional[float] = None
    # Signals the confidence module reads to decide quality — ground truth,
    # not opinions. `near_frame_edge` fires when any tracked point was within
    # about a plate radius of a frame edge during the lift (the bar left view,
    # so the measurement is not trustworthy there). `plate_too_small` fires
    # when the plate is too small on screen for the tracker's rim-finding to
    # stay inside a few pixels. `plate_axis_ratio` is the plate's measured
    # minor/major axis — 1.0 side-on, dropping as the camera tilts; the
    # confidence module caps on it and the judge is told about it so it does
    # not claim side-on when the tracker measured otherwise.
    near_frame_edge: bool = False
    plate_too_small: bool = False
    plate_axis_ratio: Optional[float] = None
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "tracked": self.tracked,
            "coverage": round(self.coverage, 2),
            "points": self.points,
        }
        if self.tracked:
            body.update(
                {
                    "max_drift_plate_diameters": _rounded(self.max_drift_diameters),
                    "max_drift_cm_estimate": None if self.max_drift_cm is None else round(self.max_drift_cm),
                    "drift_at_rise_fraction": _rounded(self.drift_at_rise_fraction),
                    "sideways_range_plate_diameters": _rounded(self.sideways_range_diameters),
                    "rise_plate_diameters": _rounded(self.rise_diameters, 1),
                }
            )
        elif self.reason:
            body["reason"] = self.reason
        return body


def _rounded(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(value, digits)


def measure(path: Optional[BarPath]) -> BarPathMeasurement:
    """Numbers from a tracked path; an honest 'not tracked' otherwise."""
    if path is None:
        return BarPathMeasurement(tracked=False, reason="the bar was not tracked")
    if not path.tracked or not path.plate_radius or not path.frame_width:
        return BarPathMeasurement(
            tracked=False,
            coverage=path.coverage,
            points=len(path.points),
            reason=path.reason or "the bar was not tracked",
        )
    plate_px = 2.0 * path.plate_radius * path.frame_width
    drift_norm, worst = path.max_drift()
    xs = [p.x for p in path.points]
    ys = [p.y for p in path.points]
    lowest, highest = max(ys), min(ys)
    fraction = None
    if worst is not None and lowest > highest:
        fraction = (lowest - worst.y) / (lowest - highest)
    drift_diameters = drift_norm * path.frame_width / plate_px
    # Plate near frame edges: any point within ~1 plate radius of any edge
    # of the frame during the lift. Expressed in the normalised coordinates
    # the tracker uses (x and y both in 0..1).
    plate_margin_x = path.plate_radius
    plate_margin_y = (path.plate_radius * path.frame_width) / max(1, path.frame_height)
    near_edge = any(
        p.x <= plate_margin_x
        or p.x >= 1.0 - plate_margin_x
        or p.y <= plate_margin_y
        or p.y >= 1.0 - plate_margin_y
        for p in path.points
    )
    # Plate too small: diameter under ~4% of the shorter frame dimension —
    # a plate so small that the ellipse / Hough fit is loose by several pixels
    # on every frame, which the drift measurement compounds.
    shorter_px = min(path.frame_width, path.frame_height) or path.frame_width
    plate_too_small = plate_px < 0.04 * shorter_px
    return BarPathMeasurement(
        tracked=True,
        coverage=path.coverage,
        points=len(path.points),
        max_drift_diameters=drift_diameters,
        max_drift_cm=drift_diameters * PLATE_DIAMETER_M * 100.0,
        drift_at_rise_fraction=fraction,
        sideways_range_diameters=(max(xs) - min(xs)) * path.frame_width / plate_px,
        rise_diameters=(lowest - highest) * path.frame_height / plate_px,
        near_frame_edge=near_edge,
        plate_too_small=plate_too_small,
        plate_axis_ratio=path.plate_axis_ratio,
    )


# (upper bound of the rise fraction, the phase as a name, the phase as a place)
_JERK_PHASES = (
    (0.35, "the dip and the start of the drive", "in the dip and the start of the drive"),
    (0.8, "the drive", "in the drive"),
    (None, "the lockout", "at the lockout"),
)
_PULL_PHASES = (
    (0.25, "the first pull, just off the floor", "in the first pull, just off the floor"),
    (0.5, "around the knee, in the transition", "around the knee, in the transition"),
    (0.75, "the second pull, around contact", "in the second pull, around contact"),
    (None, "the turnover and catch", "at the turnover and catch"),
)


def where_in_the_rise(fraction: Optional[float], lift: str, *, as_place: bool = False) -> str:
    """
    A coach's name for a height along the rise, by lift. `as_place` returns
    it with its preposition, to end a sentence with ("worst … in the drive").
    """
    if fraction is None:
        return "" if as_place else "an unknown point of the rise"
    phases = _JERK_PHASES if lift == "jerk" else _PULL_PHASES
    for bound, name, place in phases:
        if bound is None or fraction < bound:
            return place if as_place else name
    return phases[-1][2] if as_place else phases[-1][1]


# Read off the judge's own drift text. The tracker measures size and height,
# not direction, so the one thing taken from the words is forward or back.
_FORWARD_RE = re.compile(r"\b(forward|forwards|ahead of|in front of|out in front|away from the (?:body|shins|legs|thighs|lifter|athlete))\b")
_BACKWARD_RE = re.compile(r"\b(backward|backwards|back toward|back towards|swings back|drifts back|loops back|behind the)\b")


def drift_direction(text: Optional[str]) -> Optional[str]:
    """`forward`, `backward`, or None when the text says neither or both."""
    lowered = (text or "").lower()
    forward = bool(_FORWARD_RE.search(lowered))
    backward = bool(_BACKWARD_RE.search(lowered))
    if forward == backward:
        return None
    return "forward" if forward else "backward"


def drift_summary(
    measurement: Optional[BarPathMeasurement], lift: str, judge_text: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    The result card's one line about the drift.

    When the bar was tracked it is written from the measurement — the same
    number the judge was given and the replay's orange ring is drawn from — so
    the words on the card cannot contradict the figure under them. The judge's
    text is consulted for one thing the tracker cannot know: the direction.
    Without a trace the line is the judge's own words, labelled as such.
    """
    judge_text = (judge_text or "").strip()
    if measurement is None or not measurement.tracked or measurement.max_drift_diameters is None:
        if not judge_text:
            return None
        return {"text": judge_text, "source": "judge", "direction": drift_direction(judge_text)}

    diameters = measurement.max_drift_diameters
    cm = measurement.max_drift_cm if measurement.max_drift_cm is not None else diameters * PLATE_DIAMETER_M * 100.0
    cm_text = f"about {max(1, round(cm))} cm ({diameters:.2f} plate widths)"
    direction = drift_direction(judge_text)
    if diameters < CLOSE_DRIFT_DIAMETERS:
        head = "Close path" + (f", a little {direction}" if direction else "")
        body = f"the bar stayed within {cm_text} of its start line"
    else:
        head = ("Noticeable drift" if diameters < LARGE_DRIFT_DIAMETERS else "Large drift") + (
            f" {direction}" if direction else ""
        )
        body = f"{cm_text} off the start line"
    where = where_in_the_rise(measurement.drift_at_rise_fraction, lift, as_place=True)
    text = f"{head}: {body}" + (f", furthest {where}" if where else "") + "."
    return {"text": text, "source": "measurement", "direction": direction}


def describe(measurement: BarPathMeasurement, lift: str) -> str:
    """The measurement as a short evidence block for the judge's prompt."""
    if not measurement.tracked:
        why = measurement.reason or "it could not be followed"
        return (
            "Bar-path measurement from the local plate tracker (no model involved): "
            f"the bar was NOT tracked in this clip ({why}). Judge from the vision observations "
            "alone and keep confidence at or below 0.7."
        )
    drift = measurement.max_drift_diameters or 0.0
    cm = measurement.max_drift_cm or 0.0
    lines = [
        "Bar-path measurement from the local plate tracker (no model involved). It measures size and",
        "height, not direction: the vision observations say whether the bar went forward or back.",
        f"- The plate was followed in {measurement.coverage:.0%} of the sampled frames ({measurement.points} points).",
        f"- Maximum horizontal departure from the line the bar started over: {drift:.2f} plate diameters,",
        f"  roughly {cm:.0f} cm if the plate is a 450 mm bumper.",
        f"- It happened at about {100 * (measurement.drift_at_rise_fraction or 0):.0f}% of the bar's rise:"
        f" {where_in_the_rise(measurement.drift_at_rise_fraction, lift)}.",
        f"- Total sideways range over the lift: {measurement.sideways_range_diameters or 0:.2f} plate diameters.",
        f"- The bar rose {measurement.rise_diameters or 0:.1f} plate diameters.",
        f"Rule of thumb for a clip filmed square to the side: under {CLOSE_DRIFT_DIAMETERS:.2f} diameters"
        f" (~{CLOSE_DRIFT_DIAMETERS * PLATE_DIAMETER_M * 100:.0f} cm) is a close path;"
        f" {CLOSE_DRIFT_DIAMETERS:.2f}-{LARGE_DRIFT_DIAMETERS:.2f} is a noticeable drift;"
        f" above {LARGE_DRIFT_DIAMETERS:.2f} (~{LARGE_DRIFT_DIAMETERS * PLATE_DIAMETER_M * 100:.0f} cm+) is a large one.",
        "From an oblique camera the measured drift is inflated by the athlete moving toward or away",
        "from the camera, so discount it when the observations say the view is not square to the side.",
    ]
    # Oblique signal from the plate's own shape — a side-on plate projects to
    # a near-circle. Tell the judge the measured ratio when it indicates tilt,
    # so it does not reply "no indication of oblique camera inflation" on an
    # obviously oblique clip.
    axis_ratio = measurement.plate_axis_ratio
    if axis_ratio is not None and axis_ratio < 0.93:
        lines.append(
            f"- The plate was measured at a {axis_ratio:.2f} minor/major axis ratio: the camera "
            "is NOT square to the side, so the drift above is inflated. Treat the direction as "
            "indicative, not the exact centimetres."
        )
    return "\n".join(lines)
