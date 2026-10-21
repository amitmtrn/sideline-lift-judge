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
    return BarPathMeasurement(
        tracked=True,
        coverage=path.coverage,
        points=len(path.points),
        max_drift_diameters=drift_diameters,
        max_drift_cm=drift_diameters * PLATE_DIAMETER_M * 100.0,
        drift_at_rise_fraction=fraction,
        sideways_range_diameters=(max(xs) - min(xs)) * path.frame_width / plate_px,
        rise_diameters=(lowest - highest) * path.frame_height / plate_px,
    )


def where_in_the_rise(fraction: Optional[float], lift: str) -> str:
    """A coach's name for a height along the rise, by lift."""
    if fraction is None:
        return "an unknown point of the rise"
    if lift == "jerk":
        if fraction < 0.35:
            return "the dip and the start of the drive"
        if fraction < 0.8:
            return "the drive"
        return "the lockout"
    if fraction < 0.25:
        return "the first pull, just off the floor"
    if fraction < 0.5:
        return "around the knee, in the transition"
    if fraction < 0.75:
        return "the second pull, around contact"
    return "the turnover and catch"


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
    return "\n".join(lines)
