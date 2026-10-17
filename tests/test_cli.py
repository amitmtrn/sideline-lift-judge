"""FR-13: the CLI runs the same pipeline the web app runs."""
import json
from unittest.mock import patch

import pytest

import score_clip
from app.schemas.judgment import LiftJudgment, LiftType
from app.services.pipeline import DEFAULT_FRAMES, ScoreOutcome, ScoredLift
from tests.conftest import judge_reply, observation_reply, triage_reply

SNATCH_PATH = ["floor", "knees", "overhead", "overhead", "overhead", "overhead"]
CLEAN_AND_JERK_PATH = ["floor", "knees", "shoulders", "shoulders", "overhead", "overhead"]


def verdict(lift="snatch", score=7) -> LiftJudgment:
    return LiftJudgment(
        lift=lift,
        score=score,
        band="solid",
        max_horizontal_drift="small forward loop",
        cues=["keep the bar close to the body"],
        confidence=0.7,
    )


def test_parse_args_defaults():
    args = score_clip.parse_args(["clip.mp4", "--lift", "snatch"])
    assert args.lift == "snatch"
    assert args.frames == DEFAULT_FRAMES
    assert args.auto is False and args.no_split is False
    assert args.trim_start is None and args.trim_end is None


def test_parse_args_accepts_the_old_underscore_flag():
    # The original CLI documented --prompts_dir; keep it working.
    args = score_clip.parse_args(["clip.mp4", "--lift", "clean", "--prompts_dir", "p/"])
    assert args.prompts_dir == "p/"


def test_parse_args_trim_and_auto():
    args = score_clip.parse_args(["clip.mp4", "--auto", "--trim-start", "1.5", "--trim-end", "9"])
    assert args.auto is True and args.trim_start == 1.5 and args.trim_end == 9.0


def test_single_lift_payload_is_the_bare_verdict():
    outcome = ScoreOutcome(
        lifts=[ScoredLift(index=1, lift=LiftType.snatch, start_sec=0.0, end_sec=4.0, judgment=verdict())]
    )
    payload = score_clip.build_payload(outcome)
    assert payload["lift"] == "snatch" and payload["score"] == 7
    assert payload["start_sec"] == 0.0 and payload["was_split"] is False


def test_two_lifts_payload_keeps_them_separate():
    outcome = ScoreOutcome(
        lifts=[
            ScoredLift(index=1, lift=LiftType.clean, start_sec=0.0, end_sec=3.0, judgment=verdict("clean", 6), was_split=True),
            ScoredLift(index=2, lift=LiftType.jerk, start_sec=3.0, end_sec=6.0, judgment=verdict("jerk", 5), was_split=True),
        ],
        split_detected=True,
    )
    payload = score_clip.build_payload(outcome)
    assert [item["lift"] for item in payload["lifts"]] == ["clean", "jerk"]
    assert payload["split_detected"] is True


def test_payload_carries_a_failed_lift_as_an_error():
    outcome = ScoreOutcome(lifts=[ScoredLift(index=1, lift=LiftType.jerk, start_sec=0, end_sec=2, error="boom")])
    assert score_clip.build_payload(outcome)["error"] == "boom"


def test_missing_file_exits_without_calling_anything(capsys):
    assert score_clip.main(["nope.mp4", "--lift", "snatch"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_requires_a_lift_or_auto(lift_video, capsys, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    assert score_clip.main([lift_video]) == 2
    assert "--lift" in capsys.readouterr().err


def test_missing_api_key_is_reported_plainly(lift_video, capsys, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    # .env in the working tree would otherwise supply a key.
    monkeypatch.setattr(score_clip, "load_dotenv", lambda *a, **k: None)
    assert score_clip.main([lift_video, "--lift", "snatch"]) == 2
    assert "NEBIUS_API_KEY" in capsys.readouterr().err


def test_end_to_end_prints_one_verdict(lift_video, capsys, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    monkeypatch.setattr(score_clip, "load_dotenv", lambda *a, **k: None)
    with patch(
        "app.services.nebius_client.requests.post",
        side_effect=[triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 8, "solid", 0.8)],
    ):
        assert score_clip.main([lift_video, "--lift", "snatch"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["lift"] == "snatch" and body["score"] == 8
    assert "warnings" in body  # the confidence rule ran, same as on the web


def test_end_to_end_splits_a_clean_and_jerk(lift_video, capsys, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    monkeypatch.setattr(score_clip, "load_dotenv", lambda *a, **k: None)
    with patch(
        "app.services.nebius_client.requests.post",
        side_effect=[
            triage_reply(CLEAN_AND_JERK_PATH),
            observation_reply(),
            judge_reply("clean", 6, "mixed", 0.7),
            observation_reply(),
            judge_reply("jerk", 5, "mixed", 0.7),
        ],
    ):
        assert score_clip.main([lift_video, "--lift", "clean"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["split_detected"] is True
    assert [item["lift"] for item in body["lifts"]] == ["clean", "jerk"]


def test_no_split_skips_the_triage_call(lift_video, capsys, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    monkeypatch.setattr(score_clip, "load_dotenv", lambda *a, **k: None)
    with patch(
        "app.services.nebius_client.requests.post",
        side_effect=[observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
    ) as post:
        assert score_clip.main([lift_video, "--lift", "snatch", "--no-split"]) == 0
    assert post.call_count == 2
