"""
Evidence builder: maps each model cue to tracker data, phases, and time windows.
Implements the grounding check (PRD §6): drops cues where the tracker data
contradicts the claim, tags vision-only issues honestly, and logs every drop.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from server.schemas.judgment import CueEvidence, CueMedia, EvidencedCue
from server.services.measurement import BarPathMeasurement
from server.utils.bar_track import BarPath, PLATE_DIAMETER_M

logger = logging.getLogger(__name__)

# Issue codes the tracker can measure (horizontal drift).
# A cue with one of these codes is dropped when drift is below the threshold.
TRACKER_MEASURABLE = frozenset({"bar_forward_drift", "bar_loops_away", "catch_forward", "catch_behind", "jerk_dip_forward"})

# Issues the tracker cannot measure — always shown, tagged vision_only=True.
VISION_ONLY_ISSUES = frozenset({"early_arm_bend", "hips_rise_first", "incomplete_extension", "slow_turnover", "unstable_catch", "press_out"})

# Minimum drift (cm) to show a tracker-measurable cue; below this, drop it.
# Approx. 4 cm for bar path cues, 3 cm for jerk dip.
DRIFT_THRESHOLD_CM = {
    "bar_forward_drift": 4.0,
    "bar_loops_away": 5.0,
    "catch_forward": 4.0,
    "catch_behind": 4.0,
    "jerk_dip_forward": 3.0,
}

# Rubric target description per issue code, shown as evidence.target.
RUBRIC_TARGETS = {
    "bar_forward_drift": "≤ 4 cm",
    "bar_loops_away": "≤ 5 cm loop",
    "early_arm_bend": "arms straight through extension",
    "hips_rise_first": "hips and shoulders rise together",
    "incomplete_extension": "full hip/knee/ankle extension before pull-under",
    "slow_turnover": "fast turnover immediately after extension",
    "catch_forward": "≤ 4 cm from start line at catch",
    "catch_behind": "≤ 4 cm from start line at catch",
    "unstable_catch": "stable overhead lockout",
    "jerk_dip_forward": "≤ 3 cm during dip",
    "press_out": "bar locked out overhead without pressing",
}

# Cue text substrings → issue code. Order matters: first match wins.
_CUE_ISSUE_MAP: List[Tuple[str, str]] = [
    ("early arm", "early_arm_bend"),
    ("no early arms", "early_arm_bend"),
    ("early arms", "early_arm_bend"),
    ("finish the legs", "incomplete_extension"),
    ("triple extension", "incomplete_extension"),
    ("hips rise first", "hips_rise_first"),
    ("push the floor", "hips_rise_first"),
    ("slow turnover", "slow_turnover"),
    ("keep speed through transition", "slow_turnover"),
    ("elbows high and fast", "slow_turnover"),
    ("dip straight down", "jerk_dip_forward"),
    ("no forward knee", "jerk_dip_forward"),
    ("press out", "press_out"),
    ("don't press out", "press_out"),
    ("lockout with bar stacked", "press_out"),
    ("aggressive pull-under", "catch_forward"),
    ("catch locked overhead", "catch_forward"),
    ("don't chase a forward bar", "catch_forward"),
    ("bar loops", "bar_loops_away"),
    ("loops away", "bar_loops_away"),
    # Generic bar-close cues all map to bar_forward_drift
    ("keep the bar close", "bar_forward_drift"),
    ("bar close", "bar_forward_drift"),
    ("force up", "bar_forward_drift"),
    ("don't waste sideways", "bar_forward_drift"),
    ("long close pull", "bar_forward_drift"),
    ("drive up vertical", "bar_forward_drift"),
]


def cue_to_issue(cue_text: str) -> str:
    """Map a cue string to an issue code."""
    lower = cue_text.lower()
    for fragment, issue in _CUE_ISSUE_MAP:
        if fragment in lower:
            return issue
    return "bar_forward_drift"


def issue_to_phase(issue: str, measurement: BarPathMeasurement, lift: str) -> str:
    """Return the most relevant phase name for this issue."""
    if issue == "jerk_dip_forward":
        return "jerk_dip"
    if issue in ("catch_forward", "catch_behind", "unstable_catch"):
        return "catch"
    if issue == "press_out":
        return "catch"
    if issue == "slow_turnover":
        return "turnover"
    if issue == "early_arm_bend":
        return "second_pull"
    if issue in ("hips_rise_first", "incomplete_extension"):
        return "second_pull"
    # Bar path issues: map drift fraction to phase
    frac = measurement.drift_at_rise_fraction
    if lift == "jerk":
        if frac is None or frac < 0.35:
            return "jerk_dip"
        if frac < 0.8:
            return "jerk_drive"
        return "catch"
    if frac is None or frac < 0.25:
        return "first_pull"
    if frac < 0.5:
        return "transition"
    if frac < 0.75:
        return "second_pull"
    return "turnover"


def _key_point_in_window(bar_path: BarPath, t_start: float, t_end: float):
    """The tracked point with the largest drift within [t_start, t_end]."""
    if not bar_path.tracked or not bar_path.points:
        return None
    ref = bar_path.reference_x
    candidates = [p for p in bar_path.points if t_start <= p.t <= t_end]
    if not candidates:
        # Fall back to global max-drift point
        candidates = bar_path.points
    return max(candidates, key=lambda p: abs(p.x - ref))


def _drift_cm_in_window(bar_path: BarPath, t_start: float, t_end: float) -> Optional[float]:
    """Max drift in cm within the time window, or None if scale is unknown."""
    if not bar_path.tracked or not bar_path.plate_radius or not bar_path.frame_width:
        return None
    kp = _key_point_in_window(bar_path, t_start, t_end)
    if kp is None:
        return None
    drift_norm = abs(kp.x - bar_path.reference_x)
    plate_px = 2.0 * bar_path.plate_radius * bar_path.frame_width
    drift_px = drift_norm * bar_path.frame_width
    if plate_px <= 0:
        return None
    return (drift_px / plate_px) * PLATE_DIAMETER_M * 100.0


def _time_window(
    bar_path: Optional[BarPath],
    measurement: BarPathMeasurement,
    window_start: float,
    window_end: float,
) -> Tuple[float, float]:
    """Compute the cue's time window: 0.5 s each side of max-drift point."""
    if bar_path is not None and bar_path.tracked and bar_path.points:
        ref = bar_path.reference_x
        worst = max(bar_path.points, key=lambda p: abs(p.x - ref))
        key_t = worst.t
    else:
        frac = measurement.drift_at_rise_fraction or 0.5
        key_t = window_start + frac * (window_end - window_start)
    t_start = max(window_start, key_t - 0.5)
    t_end = min(window_end, key_t + 0.5)
    return t_start, t_end


def _plural(lift: str) -> str:
    """Snatches, cleans, jerks — not 'cleanes'. Keeps the sentence readable."""
    if lift.endswith("ch") or lift.endswith("sh") or lift.endswith("s"):
        return f"{lift}es"
    return f"{lift}s"


def _number(target: str) -> str:
    """
    "≤ 4 cm" reads as a rubric limit on an evidence card, but inside the
    sentence "Good cleans stay within about ≤ 4 cm here" the inequality
    sign duplicates the "within about" and reads oddly. Strip the comparison
    for the sentence; the card still carries the full target string.
    """
    stripped = target.lstrip("≤<= ").strip()
    return stripped or target


def _why_sentence(
    issue: str,
    lift: str,
    phase: str,
    t_start: float,
    t_end: float,
    value_cm: Optional[float],
) -> str:
    """One plain English sentence explaining the cue's evidence."""
    phase_label = phase.replace("_", " ")
    time_label = f"{t_start:.1f}–{t_end:.1f} s"
    target = RUBRIC_TARGETS.get(issue, "")

    if issue == "bar_forward_drift":
        if value_cm is not None:
            return (
                f"Measured {value_cm:.0f} cm forward drift at {time_label}. "
                f"Good {_plural(lift)} stay within about {_number(target)} here."
            )
        return f"Bar appeared to drift forward during the {phase_label}. Keep it close."

    if issue == "bar_loops_away":
        if value_cm is not None:
            return (
                f"Bar looped {value_cm:.0f} cm away from the body at {time_label}. "
                f"Aim for {target}."
            )
        return f"Bar looped away from the body during the {phase_label}."

    if issue == "early_arm_bend":
        return (
            f"Arms appear to pull the bar away before full extension ({phase_label}, {time_label}). "
            "Keep arms long until the hips and legs have finished their drive."
        )

    if issue == "hips_rise_first":
        return (
            f"Hips appear to rise ahead of shoulders in the {phase_label} ({time_label}). "
            "Push the floor hard so hips and shoulders rise together."
        )

    if issue == "incomplete_extension":
        return (
            f"Full triple extension (hip, knee, ankle) not reached before the pull-under "
            f"({phase_label}, {time_label}). Finish the legs before the arms take over."
        )

    if issue == "slow_turnover":
        return (
            f"Turnover into the catch appears slow after extension ({time_label}). "
            "Faster pull-under puts you under a still-rising bar."
        )

    if issue == "catch_forward":
        if value_cm is not None:
            return (
                f"Catch appears {value_cm:.0f} cm forward of the start line at {time_label}. "
                f"Aim for {target}."
            )
        return f"Catch position appears forward of mid-foot ({time_label})."

    if issue == "catch_behind":
        return f"Catch appears behind mid-foot at {time_label}. Chase the bar back to mid-foot."

    if issue == "unstable_catch":
        return f"Catch appears unstable at {time_label}. Prioritise getting under a close bar."

    if issue == "jerk_dip_forward":
        if value_cm is not None:
            return (
                f"Dip sends bar ~{value_cm:.0f} cm forward of mid-foot at {time_label}. "
                f"Aim for {target}."
            )
        return f"Dip appears to send the bar forward of mid-foot at {time_label}."

    if issue == "press_out":
        return f"Bar pressed out rather than locked out overhead at {time_label} — potential press-out call."

    return f"Issue detected in the {phase_label} at {time_label}."


def build_evidence_cues(
    cues: List[str],
    measurement: BarPathMeasurement,
    bar_path: Optional[BarPath],
    lift: str,
    window_start_sec: float,
    window_end_sec: float,
) -> List[EvidencedCue]:
    """
    Map each model cue text to an EvidencedCue with tracker data.

    Drops cues whose claimed drift is contradicted by the tracker (G4).
    Vision-only issues (not measurable by the tracker) are always kept,
    tagged with vision_only=True.

    Returns the surviving cues in order.
    """
    result: List[EvidencedCue] = []
    dropped = 0

    for n, cue_text in enumerate(cues, start=1):
        cue_id = f"c{n}"
        issue = cue_to_issue(cue_text)
        phase = issue_to_phase(issue, measurement, lift)
        t_start, t_end = _time_window(bar_path, measurement, window_start_sec, window_end_sec)
        is_vision_only = issue in VISION_ONLY_ISSUES

        if is_vision_only:
            # Cannot measure; always show with source="vision", no arrow
            evidence = CueEvidence(
                t_start_s=t_start,
                t_end_s=t_end,
                metric=None,
                value=None,
                target=RUBRIC_TARGETS.get(issue, ""),
                source="vision",
            )
            why = _why_sentence(issue, lift, phase, t_start, t_end, None)
            result.append(EvidencedCue(
                id=cue_id,
                text=cue_text,
                phase=phase,
                issue=issue,
                evidence=evidence,
                why=why,
                verified=False,
                vision_only=True,
            ))
            continue

        # Tracker-measurable issue
        value_cm: Optional[float] = None
        verified = False

        if measurement.tracked and bar_path is not None:
            # Check tracker coverage in window
            if bar_path.points:
                window_pts = [p for p in bar_path.points if t_start <= p.t <= t_end]
                window_duration = t_end - t_start
                fps_estimate = len(bar_path.points) / max(0.1, window_end_sec - window_start_sec)
                expected = max(1, fps_estimate * window_duration)
                coverage_in_window = len(window_pts) / expected if expected > 0 else 0.0
            else:
                coverage_in_window = 0.0

            value_cm = _drift_cm_in_window(bar_path, t_start, t_end)
            threshold = DRIFT_THRESHOLD_CM.get(issue, 4.0)

            if value_cm is not None and value_cm < threshold:
                # Tracker contradicts the cue — drop it
                logger.info(
                    "Dropped cue %r (issue %s): tracker measured %.1f cm < threshold %.1f cm",
                    cue_text, issue, value_cm, threshold,
                )
                dropped += 1
                continue

            verified = value_cm is not None and coverage_in_window >= 0.6
            if coverage_in_window < 0.6:
                logger.debug("Low tracking coverage in cue window: %.0f%%", coverage_in_window * 100)

        source = "tracker" if measurement.tracked else "vision"
        evidence = CueEvidence(
            t_start_s=t_start,
            t_end_s=t_end,
            metric="horizontal_drift_cm",
            # Round to a whole centimetre so the number on the card, the
            # label on the replay strip and the why sentence all agree.
            value=float(round(value_cm)) if value_cm is not None else None,
            target=RUBRIC_TARGETS.get(issue, ""),
            source=source,
        )
        why = _why_sentence(issue, lift, phase, t_start, t_end, value_cm)
        result.append(EvidencedCue(
            id=cue_id,
            text=cue_text,
            phase=phase,
            issue=issue,
            evidence=evidence,
            why=why,
            verified=verified,
            vision_only=False,
        ))

    if dropped:
        logger.info("Evidence builder: dropped %d/%d cues (tracker contradicted claim)", dropped, len(cues))

    return result
