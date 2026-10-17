import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from server.schemas.judgment import LiftType
from server.schemas.segment import ExerciseReport, SegmentTriage, SessionReport, parse_triage_json
from server.utils.segmentation import CandidateSegment, MotionProfile
from tests.test_segmentation import _write_session_video

import score_session


# --- triage parsing -------------------------------------------------------


def test_triage_messages_label_every_frame():
    """
    Unlabelled frames made the model name the lift the scene reminded it of;
    numbered frames make it answer per frame.
    """
    from server.services.nebius_client import assemble_triage_messages

    messages = assemble_triage_messages(images=["data:image/jpeg;base64,x", "data:image/jpeg;base64,y"], prompts_dir="prompts")
    content = messages[1]["content"]
    assert [part["text"] for part in content if part["type"] == "text"][1:] == ["Frame 1:", "Frame 2:"]
    assert [part["type"] for part in content] == ["text", "text", "image_url", "text", "image_url"]


def test_parse_triage_json_plain():
    triage = parse_triage_json('{"lifts":["snatch"],"activity":"snatch attempt","confidence":0.8}')
    assert triage.lifts == [LiftType.snatch]
    assert triage.is_lift


def test_parse_triage_json_from_prose_and_fences():
    text = 'Looking at the frames:\n```json\n{"lifts": ["clean", "jerk"], "activity": "clean and jerk"}\n```'
    triage = parse_triage_json(text)
    assert triage.lifts == [LiftType.clean, LiftType.jerk]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"lifts":"snatch","activity":"x"}', [LiftType.snatch]),
        ('{"lifts":["clean and jerk"],"activity":"x"}', [LiftType.clean, LiftType.jerk]),
        ('{"lifts":["none"],"activity":"walking"}', []),
        ('{"lifts":null,"activity":"resting"}', []),
        ('{"lifts":[],"activity":"loading plates"}', []),
        ('{"lifts":["deadlift"],"activity":"deadlift"}', []),
        ('{"lifts":["SNATCH"],"activity":"x"}', [LiftType.snatch]),
    ],
)
def test_triage_tolerates_loose_vision_replies(raw, expected):
    """Vision models answer loosely; a shrug must not crash a session run."""
    assert parse_triage_json(raw).lifts == expected


def test_triage_coerces_percent_confidence():
    assert parse_triage_json('{"lifts":[],"confidence":85}').confidence == 0.85
    assert parse_triage_json('{"lifts":[],"confidence":"0.4"}').confidence == 0.4
    assert parse_triage_json('{"lifts":[],"confidence":"high"}').confidence is None


# --- deriving the lift from the bar path ---------------------------------


@pytest.mark.parametrize(
    "path, expected",
    [
        # Real MiniCPM-V readings of samples/clip.mp4 at 768px, window by window.
        (["floor", "floor", "floor", "shoulders", "shoulders", "shoulders", "shoulders", "overhead"], ["clean"]),
        (["shoulders"] * 4 + ["overhead"] * 4, ["jerk"]),
        (["overhead", "overhead", "overhead", "floor", "floor", "floor", "floor", "floor"], []),
        # Textbook shapes.
        (["floor", "floor", "knees", "overhead", "overhead", "overhead"], ["snatch"]),
        (["floor", "knees", "shoulders", "shoulders", "overhead", "overhead"], ["clean", "jerk"]),
        (["floor", "floor", "knees", "knees", "floor", "floor"], []),  # deadlift / aborted pull
        (["floor"] * 6, []),  # bar idle on the platform
        (["shoulders", "shoulders", "shoulders", "floor", "floor"], []),  # dumping from the rack
    ],
)
def test_lifts_from_bar_path(path, expected):
    from server.schemas.segment import lifts_from_bar_path

    assert lifts_from_bar_path(path) == [LiftType(name) for name in expected]


def test_single_stray_frame_does_not_promote_a_clean_to_a_jerk():
    from server.schemas.segment import lifts_from_bar_path

    # One frame misread as overhead at the end of a clean: the rest of the
    # window says the bar stayed on the shoulders.
    assert lifts_from_bar_path(["floor", "floor", "shoulders", "shoulders", "shoulders", "overhead"]) == [
        LiftType.clean
    ]


def test_bar_path_overrides_a_lift_the_model_named():
    triage = parse_triage_json(
        '{"lifts":["clean","jerk"],"bar_by_frame":["shoulders","shoulders","shoulders","overhead","overhead"],'
        '"athlete_lifting":true,"activity":"clean and jerk"}'
    )
    assert triage.lifts == [LiftType.jerk]


def test_athlete_not_lifting_overrides_everything():
    triage = parse_triage_json(
        '{"bar_by_frame":["floor","knees","shoulders","overhead"],"athlete_lifting":false,"activity":"coach demo"}'
    )
    assert triage.lifts == []
    assert not triage.is_lift


def test_unreadable_bar_path_falls_back_to_the_named_lift():
    triage = parse_triage_json('{"lifts":["snatch"],"bar_by_frame":["not_visible","not_visible"],"activity":"dark"}')
    assert triage.lifts == [LiftType.snatch]


def test_bar_level_synonyms():
    from server.schemas.segment import FLOOR, KNEES, NOT_VISIBLE, OVERHEAD, SHOULDERS, bar_level

    assert bar_level("Floor") == FLOOR
    assert bar_level("on the ground") == FLOOR
    assert bar_level("hips") == KNEES
    assert bar_level("front rack") == SHOULDERS
    assert bar_level("locked out overhead") == OVERHEAD
    assert bar_level("not_visible") == NOT_VISIBLE
    assert bar_level(None) == NOT_VISIBLE


# --- clean & jerk expansion ----------------------------------------------


def _two_hump_profile() -> MotionProfile:
    times = [i * 0.1 for i in range(101)]
    energy = [1.0 if (1.0 < t < 4.0 or 6.0 < t < 9.0) else 0.05 for t in times]
    return MotionProfile(times=times, energy=energy, duration_sec=10.0, fps=30.0, sample_fps=10.0)


def test_expand_for_lifts_single_lift():
    segment = CandidateSegment(index=1, start_sec=0.0, end_sec=10.0)
    pairs = score_session.expand_for_lifts(segment, SegmentTriage(lifts=[LiftType.snatch]), _two_hump_profile())
    assert pairs == [(segment, LiftType.snatch)]


def test_expand_for_lifts_skips_non_lift():
    segment = CandidateSegment(index=1, start_sec=0.0, end_sec=10.0)
    assert score_session.expand_for_lifts(segment, SegmentTriage(lifts=[]), _two_hump_profile()) == []


def test_expand_for_lifts_splits_clean_and_jerk():
    segment = CandidateSegment(index=1, start_sec=0.0, end_sec=10.0)
    triage = SegmentTriage(lifts=[LiftType.clean, LiftType.jerk])
    pairs = score_session.expand_for_lifts(segment, triage, _two_hump_profile())
    assert [lift for _, lift in pairs] == [LiftType.clean, LiftType.jerk]
    clean_window, jerk_window = pairs[0][0], pairs[1][0]
    assert clean_window.start_sec == 0.0
    assert clean_window.end_sec == jerk_window.start_sec
    assert jerk_window.end_sec == 10.0
    # The split lands in the quiet stretch between the two bursts of motion.
    assert 4.0 <= clean_window.end_sec <= 6.0


def test_expand_for_lifts_falls_back_to_whole_window_when_unsplittable():
    segment = CandidateSegment(index=1, start_sec=0.0, end_sec=1.0)
    triage = SegmentTriage(lifts=[LiftType.clean, LiftType.jerk])
    pairs = score_session.expand_for_lifts(segment, triage, MotionProfile())
    assert [lift for _, lift in pairs] == [LiftType.clean, LiftType.jerk]
    assert pairs[0][0] is segment and pairs[1][0] is segment


# --- ranking --------------------------------------------------------------


def _exercise(index: int, score, *, confidence=0.7, band="solid", error=None) -> ExerciseReport:
    judgment = None
    if score is not None or band == "unknown":
        judgment = {
            "lift": "snatch",
            "score": score,
            "band": band,
            "max_horizontal_drift": "x",
            "cues": [] if band == "unknown" else ["keep the bar close to the body"],
            "confidence": confidence,
        }
    return ExerciseReport(
        index=index,
        start_sec=float(index),
        end_sec=float(index) + 3.0,
        lift=LiftType.snatch,
        judgment=judgment,
        error=error,
    )


def test_ranked_orders_best_first_and_sinks_unscored():
    report = SessionReport(
        source="s.mp4",
        duration_sec=60.0,
        candidate_segments=5,
        exercises=[
            _exercise(1, 6),
            _exercise(2, None, band="unknown"),
            _exercise(3, 9),
            _exercise(4, None, error="boom"),
            _exercise(5, 9, confidence=0.9),
        ],
    )
    assert [item.index for item in report.ranked()] == [5, 3, 1, 2, 4]


# --- end to end, every model call mocked ---------------------------------


class FakeNebius:
    """
    Answers by prompt content rather than call order, so the assertions hold
    regardless of how the event loop interleaves the calls.
    """

    def __init__(self, triage_replies: List[str], scores: List[int]) -> None:
        self.triage_replies = list(triage_replies)
        self.scores = list(scores)
        self.calls: List[str] = []

    def __call__(self, url: str, headers=None, json: Dict[str, Any] = None, timeout=None):  # noqa: A002
        blob = _messages_text(json["messages"])
        if "bar_by_frame" in blob:
            self.calls.append("triage")
            return _response(self.triage_replies.pop(0) if self.triage_replies else '{"lifts":[],"activity":"?"}')
        if "factual observer" in blob:
            self.calls.append("observe")
            return _response("- setup: bar over mid-foot\n- catch: stacked overhead")
        self.calls.append("judge")
        lift = "snatch"
        for name in ("snatch", "clean", "jerk"):
            if f'lift = "{name}"' in blob:
                lift = name
        score = self.scores.pop(0) if self.scores else 7
        return _response(
            json_dumps_judgment(lift=lift, score=score)
        )


SNATCH_PATH = json.dumps(
    {
        "bar_by_frame": ["floor", "floor", "knees", "overhead", "overhead", "overhead"],
        "athlete_lifting": True,
        "activity": "snatch attempt",
        "confidence": 0.9,
    }
)


def _messages_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(item.get("text", "") for item in content if isinstance(item, dict))
    return "\n".join(parts)


def _response(content: str) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


def json_dumps_judgment(*, lift: str, score: int) -> str:
    return json.dumps(
        {
            "lift": lift,
            "score": score,
            "band": "solid" if score >= 7 else "mixed",
            "max_horizontal_drift": "small forward loop at the knee",
            "cues": ["keep the bar close to the body"],
            "confidence": 0.7,
        }
    )


@pytest.fixture
def session_clip():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "session.mp4")
        _write_session_video(path, busy_windows=[(2.0, 5.0), (11.0, 14.0), (20.0, 23.0)], duration=26.0)
        yield path, td


def test_session_run_splits_triages_scores_and_ranks(session_clip, monkeypatch, capsys):
    video_path, td = session_clip
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    out_dir = os.path.join(td, "out")
    fake = FakeNebius(
        triage_replies=[
            SNATCH_PATH,
            '{"bar_by_frame":["floor","floor","floor","floor"],"athlete_lifting":false,'
            '"activity":"athlete loading plates","confidence":0.6}',
            SNATCH_PATH,
        ],
        scores=[9, 5],
    )
    with patch("server.services.nebius_client.requests.post", side_effect=fake):
        code = score_session.main(
            [video_path, "--out-dir", out_dir, "--concurrency", "1", "--no-clips", "--frames", "8"]
        )
    assert code == 0

    report = json.loads(open(os.path.join(out_dir, "report.json"), encoding="utf-8").read())
    assert report["candidate_segments"] == 3
    assert len(report["exercises"]) == 2
    assert len(report["skipped"]) == 1
    assert report["skipped"][0]["activity"] == "athlete loading plates"
    assert {item["judgment"]["score"] for item in report["exercises"]} == {9, 5}
    # One triage call per window, then observe + judge per scored exercise.
    assert fake.calls.count("triage") == 3
    assert fake.calls.count("observe") == 2
    assert fake.calls.count("judge") == 2

    # Frames were saved for inspection; windows carry real timestamps.
    for item in report["exercises"]:
        assert item["frames_dir"] and os.path.isdir(item["frames_dir"])
        assert os.listdir(item["frames_dir"])
        assert item["end_sec"] > item["start_sec"]

    table = capsys.readouterr().out
    assert "rank" in table and "ex#" in table
    scores_in_order = [int(line.split()[4]) for line in table.splitlines() if "snatch" in line]
    assert scores_in_order == sorted(scores_in_order, reverse=True)


def test_session_forced_lift_skips_triage(session_clip, monkeypatch):
    video_path, td = session_clip
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    fake = FakeNebius(triage_replies=[], scores=[8, 8, 8])
    with patch("server.services.nebius_client.requests.post", side_effect=fake):
        code = score_session.main(
            [video_path, "--lift", "clean", "--concurrency", "1", "--json", os.path.join(td, "r.json")]
        )
    assert code == 0
    assert fake.calls.count("triage") == 0
    report = json.loads(open(os.path.join(td, "r.json"), encoding="utf-8").read())
    assert len(report["exercises"]) == 3
    assert {item["lift"] for item in report["exercises"]} == {"clean"}


def test_session_split_only_makes_no_model_calls(session_clip, capsys):
    video_path, _ = session_clip
    with patch("server.services.nebius_client.requests.post", side_effect=AssertionError("no API call expected")):
        code = score_session.main([video_path, "--split-only"])
    assert code == 0
    out = capsys.readouterr().out
    assert len([line for line in out.splitlines() if line.strip() and "start" not in line]) == 3


def test_session_reports_judge_failure_per_exercise(session_clip, monkeypatch):
    video_path, td = session_clip
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_JUDGE_FALLBACK_MODEL", "")

    def flaky(url, headers=None, json=None, timeout=None):  # noqa: A002
        blob = _messages_text(json["messages"])
        if "bar_by_frame" in blob:
            return _response(SNATCH_PATH)
        if "factual observer" in blob:
            return _response("- bar drifts forward")
        return _response("I need to think about this more")  # never valid JSON

    with patch("server.services.nebius_client.requests.post", side_effect=flaky):
        code = score_session.main([video_path, "--concurrency", "1", "--json", os.path.join(td, "r.json")])
    # Nothing scored, but the run completes and records why.
    assert code == 6
    report = json.loads(open(os.path.join(td, "r.json"), encoding="utf-8").read())
    assert len(report["exercises"]) == 3
    assert all(item["judgment"] is None and item["error"] for item in report["exercises"])


def test_session_judge_fallback_model_is_used(session_clip, monkeypatch):
    video_path, td = session_clip
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_JUDGE_MODEL", "primary-judge")
    monkeypatch.setenv("NEBIUS_JUDGE_FALLBACK_MODEL", "backup-judge")
    seen_models: List[str] = []

    def by_model(url, headers=None, json=None, timeout=None):  # noqa: A002
        blob = _messages_text(json["messages"])
        if "bar_by_frame" in blob:
            return _response(SNATCH_PATH)
        if "factual observer" in blob:
            return _response("- bar drifts forward")
        seen_models.append(json["model"])
        if json["model"] == "primary-judge":
            return _response("still thinking")
        return _response(json_dumps_judgment(lift="snatch", score=6))

    with patch("server.services.nebius_client.requests.post", side_effect=by_model):
        code = score_session.main(
            [video_path, "--concurrency", "1", "--max-segments", "1", "--json", os.path.join(td, "r.json")]
        )
    assert code == 0
    assert "backup-judge" in seen_models
    report = json.loads(open(os.path.join(td, "r.json"), encoding="utf-8").read())
    assert report["exercises"][0]["judgment"]["score"] == 6


@pytest.mark.skipif(not score_session.has_ffmpeg(), reason="ffmpeg not installed")
def test_session_cuts_clip_files(session_clip, monkeypatch):
    video_path, td = session_clip
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    out_dir = os.path.join(td, "cuts")
    fake = FakeNebius(triage_replies=[SNATCH_PATH], scores=[7])
    with patch("server.services.nebius_client.requests.post", side_effect=fake):
        code = score_session.main(
            [video_path, "--out-dir", out_dir, "--max-segments", "1", "--concurrency", "1", "--no-frames"]
        )
    assert code == 0
    report = json.loads(open(os.path.join(out_dir, "report.json"), encoding="utf-8").read())
    clip_path = report["exercises"][0]["clip_path"]
    assert clip_path and os.path.getsize(clip_path) > 1024
