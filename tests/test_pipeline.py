"""Scoring pipeline: trim validation, the clean & jerk split, and failure paths."""
import asyncio
from unittest.mock import patch

import pytest

from server.schemas.judgment import LiftType
from server.services.nebius_client import NebiusChatClient
from server.services.pipeline import (
    DEFAULT_FRAMES,
    MAX_FRAMES,
    MIN_FRAMES,
    ScoreRequest,
    TrimError,
    clamp_frames,
    resolve_trim,
    run_scoring,
)
from tests.conftest import judge_reply, observation_reply, triage_reply

# A bar that leaves the floor, is racked, then goes overhead: a clean & jerk.
CLEAN_AND_JERK_PATH = ["floor", "knees", "shoulders", "shoulders", "overhead", "overhead"]
SNATCH_PATH = ["floor", "knees", "overhead", "overhead", "overhead", "overhead"]
NO_LIFT_PATH = ["not_visible", "floor", "floor", "floor", "not_visible", "floor"]


def run(coro):
    return asyncio.run(coro)


def client() -> NebiusChatClient:
    return NebiusChatClient(api_key="x", base_url="https://example.test/v1")


def score(video, responses, **kwargs):
    request = ScoreRequest(video_path=video, **kwargs)
    with patch("server.services.nebius_client.requests.post", side_effect=responses) as post:
        outcome = run(run_scoring(request, client=client(), prompts_dir="prompts"))
    return outcome, post


# ---------- trim ----------


def test_resolve_trim_defaults_to_the_whole_clip():
    assert resolve_trim(duration_sec=12.0, trim_start=None, trim_end=None) == (0.0, 12.0, False)


def test_resolve_trim_accepts_a_window_inside_the_clip():
    assert resolve_trim(duration_sec=12.0, trim_start=2.0, trim_end=5.5) == (2.0, 5.5, True)


def test_resolve_trim_clamps_an_end_past_the_clip():
    start, end, trimmed = resolve_trim(duration_sec=12.0, trim_start=2.0, trim_end=99.0)
    assert (start, end, trimmed) == (2.0, 12.0, True)


@pytest.mark.parametrize(
    "start,end",
    [(-1.0, 4.0), (5.0, 5.0), (6.0, 3.0), (20.0, 25.0)],
)
def test_resolve_trim_rejects_an_impossible_window(start, end):
    with pytest.raises(TrimError):
        resolve_trim(duration_sec=12.0, trim_start=start, trim_end=end)


def test_resolve_trim_without_a_known_duration_trusts_the_request():
    assert resolve_trim(duration_sec=0.0, trim_start=1.0, trim_end=4.0) == (1.0, 4.0, True)


@pytest.mark.parametrize(
    "value,expected",
    [(None, DEFAULT_FRAMES), (1, MIN_FRAMES), (99, MAX_FRAMES), (12, 12), ("nonsense", DEFAULT_FRAMES)],
)
def test_clamp_frames(value, expected):
    assert clamp_frames(value) == expected


# ---------- scoring ----------


def test_single_clip_scores_the_selected_lift(lift_video):
    outcome, post = score(
        lift_video,
        [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
        lift=LiftType.snatch,
        render_replay=False,
    )
    assert len(outcome.scored) == 1
    scored = outcome.scored[0]
    assert scored.lift is LiftType.snatch
    assert scored.judgment.score == 7
    assert scored.was_split is False
    # triage + observations + judge
    assert post.call_count == 3


def test_clean_and_jerk_is_split_into_two_scored_lifts(lift_video):
    outcome, post = score(
        lift_video,
        [
            triage_reply(CLEAN_AND_JERK_PATH),
            observation_reply(),
            judge_reply("clean", 6, "mixed", 0.7),
            observation_reply(),
            judge_reply("jerk", 5, "mixed", 0.7),
        ],
        lift=LiftType.clean,
        render_replay=False,
    )
    assert [item.lift for item in outcome.scored] == [LiftType.clean, LiftType.jerk]
    assert outcome.split_detected is True
    assert all(item.was_split for item in outcome.scored)
    # The two halves cover different parts of the clip.
    first, second = outcome.scored
    assert first.end_sec <= second.start_sec + 1e-6
    assert first.start_sec < first.end_sec and second.start_sec < second.end_sec
    assert post.call_count == 5


def test_split_lifts_are_noted_rather_than_penalised(lift_video):
    outcome, _ = score(
        lift_video,
        [
            triage_reply(CLEAN_AND_JERK_PATH),
            observation_reply(),
            judge_reply("clean", 6, "mixed", 0.7),
            observation_reply(),
            judge_reply("jerk", 5, "mixed", 0.7),
        ],
        lift=LiftType.clean,
        render_replay=False,
    )
    for item in outcome.scored:
        # The split itself never caps. On this tiny synthetic clip the tracker
        # may not follow the block through a half, and that cap (0.65) is the
        # only thing allowed to lower the judge's 0.7.
        assert item.judgment.confidence == (0.7 if item.bar_path_tracked else 0.65)
        assert any("scored on its own" in warning for warning in item.judgment.warnings)
        assert not any("more than one lift" in warning for warning in item.judgment.warnings)


def test_triage_that_finds_nothing_falls_back_to_the_chosen_lift(lift_video):
    outcome, _ = score(
        lift_video,
        [triage_reply(NO_LIFT_PATH), observation_reply(), judge_reply("jerk", 4, "poor", 0.6)],
        lift=LiftType.jerk,
        render_replay=False,
    )
    assert len(outcome.scored) == 1
    assert outcome.scored[0].lift is LiftType.jerk


def test_detect_split_can_be_turned_off(lift_video):
    outcome, post = score(
        lift_video,
        [observation_reply(), judge_reply("snatch", 8, "solid", 0.8)],
        lift=LiftType.snatch,
        render_replay=False,
        detect_split=False,
    )
    assert len(outcome.scored) == 1
    assert post.call_count == 2  # no triage call


def test_auto_mode_skips_windows_without_a_lift(lift_video):
    outcome, _ = score(
        lift_video,
        [triage_reply(NO_LIFT_PATH, activity="walking up to the bar")],
        lift=None,
        mode="auto",
        render_replay=False,
    )
    assert outcome.scored == []
    assert outcome.skipped and "walking up to the bar" in outcome.skipped[0]


def test_a_failing_judge_marks_only_that_lift(lift_video):
    boom = RuntimeError("Nebius API error 503")
    outcome, _ = score(
        lift_video,
        [triage_reply(SNATCH_PATH), observation_reply(), boom, boom],
        lift=LiftType.snatch,
        render_replay=False,
    )
    assert outcome.scored == []
    assert len(outcome.lifts) == 1
    # The user gets one plain sentence; the raw error is kept for logs and --debug.
    failed = outcome.lifts[0]
    assert "busy" in failed.error and "503" not in failed.error
    assert "Nebius API error 503" in failed.error_detail


def test_partial_judgment_when_judge_fails_but_tracker_tracked():
    """
    Judge went down mid-run: the lift still has drift, a path and a replay,
    so the card shows a verdict row with "unknown" band and a plain-English
    note instead of nothing. Nothing but the judge's answer is lost.
    """
    from server.schemas.judgment import LiftJudgment
    from server.services.measurement import BarPathMeasurement
    from server.services.pipeline import _partial_judgment

    measurement = BarPathMeasurement(
        tracked=True, coverage=0.98, points=120,
        max_drift_diameters=0.3, max_drift_cm=14.0,
        drift_at_rise_fraction=0.6, sideways_range_diameters=0.4, rise_diameters=3.8,
    )
    judgment = _partial_judgment(RuntimeError("The model service took too long to answer"),
                                 measurement, LiftType.clean)
    assert isinstance(judgment, LiftJudgment)
    assert judgment.score is None
    assert judgment.band == "unknown"
    assert judgment.cues == []
    assert "judge did not answer" in (judgment.notes or "")
    assert "14 cm" in (judgment.max_horizontal_drift or "")
    assert judgment.warnings and "unavailable" in judgment.warnings[0].lower()


def test_the_judge_is_given_the_tracker_measurement(lift_video):
    from server.services.pipeline import ScoreRequest

    responses = [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)]
    request = ScoreRequest(video_path=lift_video, lift=LiftType.snatch, render_replay=False)
    with patch("server.services.nebius_client.requests.post", side_effect=responses) as post:
        outcome = run(run_scoring(request, client=client(), prompts_dir="prompts"))
    scored = outcome.scored[0]
    assert scored.measurement is not None
    judge_call = post.call_args_list[-1].kwargs["json"]
    evidence = "\n".join(str(message["content"]) for message in judge_call["messages"])
    assert "Bar-path measurement from the local plate tracker" in evidence
    if scored.measurement.tracked:
        assert "plate diameters" in evidence
    else:
        assert "NOT tracked" in evidence


def test_trim_window_limits_what_is_judged(lift_video):
    outcome, _ = score(
        lift_video,
        [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
        lift=LiftType.snatch,
        trim_start=1.0,
        trim_end=3.0,
        render_replay=False,
    )
    scored = outcome.scored[0]
    assert scored.start_sec >= 1.0 - 1e-6
    assert scored.end_sec <= 3.0 + 1e-6


def test_observations_are_kept_only_when_asked(lift_video):
    outcome, _ = score(
        lift_video,
        [triage_reply(SNATCH_PATH), observation_reply("Bar close throughout."), judge_reply()],
        lift=LiftType.snatch,
        render_replay=False,
        keep_observations=True,
    )
    assert outcome.scored[0].observations == "Bar close throughout."


# ---------- splitting a window triage calls a clean & jerk ----------


def _window(start, end, index=1):
    from server.utils.segmentation import CandidateSegment

    return CandidateSegment(index=index, start_sec=start, end_sec=end)


def _tracked(points):
    from server.utils.bar_track import BarPoint, LiftWindow

    return LiftWindow(
        start_sec=points[0][0], end_sec=points[-1][0], rise=0.4, peak_sec=points[-1][0], radius=40,
        points=[BarPoint(t=t, x=0.5, y=y) for t, y in points], frame_width=640, frame_height=1138,
    )


def _rise(t0, t1, y0, y1, n=20):
    return [(t0 + (t1 - t0) * k / (n - 1), y0 + (y1 - y0) * k / (n - 1)) for k in range(n)]


def _hold(t0, t1, y, n=10):
    return [(t0 + (t1 - t0) * k / (n - 1), y) for k in range(n)]


def test_two_rises_split_between_them():
    from server.services.pipeline import _split_by_rises

    tracked = _tracked(_rise(0, 1.5, 0.85, 0.5) + _hold(1.5, 4.0, 0.5) + _rise(4.0, 5.0, 0.5, 0.2) + _hold(5.0, 6.0, 0.2))
    parts, lifts = _split_by_rises(_window(0.0, 6.0), tracked, [LiftType.clean, LiftType.jerk])
    assert [l.value for l in lifts] == ["clean", "jerk"]
    assert len(parts) == 2
    cut = parts[0].end_sec
    assert 1.5 < cut < 4.0  # between the clean's top and the jerk's drive
    assert parts[1].start_sec == cut


def test_one_rise_is_left_to_the_valley_split():
    from server.services.pipeline import _split_by_rises

    tracked = _tracked(_rise(0, 1.5, 0.85, 0.4) + _hold(1.5, 4.0, 0.4))
    parts, lifts = _split_by_rises(_window(0.0, 4.0), tracked, [LiftType.clean, LiftType.jerk])
    assert parts == []
    assert len(lifts) == 2  # caller decides, with the motion profile


def test_a_second_rise_too_short_to_be_a_lift_does_not_split():
    from server.services.pipeline import MIN_SPLIT_PART, _split_by_rises

    # Second rise right at the end: the "jerk" part would be 0.6 s long, so the
    # window stays whole and is sized as one lift by its total travel.
    tracked = _tracked(_rise(0, 1.5, 0.85, 0.5) + _hold(1.5, 3.0, 0.5) + _rise(3.0, 3.4, 0.5, 0.2, n=6))
    parts, lifts = _split_by_rises(_window(0.0, 3.6), tracked, [LiftType.clean, LiftType.jerk])
    assert parts == [] and len(lifts) == 2
    assert MIN_SPLIT_PART > 0.6

