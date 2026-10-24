"""
The JSON shape of a finished run, shared by the jobs API, the synchronous
endpoint and the cached sample clip, so a verdict looks the same however it
was produced.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from server.services.measurement import drift_summary
from server.services.pipeline import ScoreOutcome, ScoredLift

ReplayUrl = Callable[[int], str]


def lift_payload(item: ScoredLift, *, replay_url: Optional[ReplayUrl] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "index": item.index,
        "lift": item.lift.value,
        "start_sec": item.start_sec,
        "end_sec": item.end_sec,
        "was_split": item.was_split,
        "bar_path_tracked": item.bar_path_tracked,
    }
    if item.error:
        body["error"] = item.error
    if item.measurement is not None:
        # What the tracker measured — the same numbers the judge was given and
        # the replay's trace is drawn from.
        body["bar_path"] = item.measurement.as_dict()
    if item.judgment is not None:
        body.update(item.judgment.model_dump())
        if item.replay_path and replay_url is not None:
            body["annotated_video_url"] = replay_url(item.index)
        else:
            body["annotated_video_url"] = None
        # The card's drift line: from the measurement when there is one, so it
        # cannot contradict the number; the judge's own words stay in
        # `max_horizontal_drift` for the record.
        drift = drift_summary(item.measurement, item.lift.value, item.judgment.max_horizontal_drift)
        if drift:
            body["drift"] = drift
    if item.models:
        # Which model saw the frames and which one judged — the fallback, when
        # the primary failed — so the card never implies a model it did not use.
        body["models"] = dict(item.models)
    if item.observations:
        body["observations"] = item.observations
    return body


def outcome_payload(outcome: ScoreOutcome, *, replay_url: Optional[ReplayUrl] = None) -> Dict[str, Any]:
    return {
        "lifts": [lift_payload(item, replay_url=replay_url) for item in outcome.lifts],
        "split_detected": outcome.split_detected,
        "candidate_windows": outcome.candidate_windows,
        "skipped": outcome.skipped,
        "duration_sec": round(outcome.duration_sec, 2),
        "start_sec": round(outcome.start_sec, 2),
        "end_sec": round(outcome.end_sec, 2),
    }
