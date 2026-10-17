"""Scoring pipeline: trim validation, the clean & jerk split, and failure paths."""
import asyncio
from unittest.mock import patch

import pytest

from app.schemas.judgment import LiftType
from app.services.nebius_client import NebiusChatClient
from app.services.pipeline import (
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
    with patch("app.services.nebius_client.requests.post", side_effect=responses) as post:
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
        assert item.judgment.confidence == 0.7
        assert any("scored on its own" in warning for warning in item.judgment.warnings)


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
    assert "Nebius API error 503" in outcome.lifts[0].error


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
