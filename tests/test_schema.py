import json

import pytest
from pydantic import ValidationError

from app.schemas.judgment import LiftJudgment, parse_and_validate_json


def test_parse_and_validate_json_ok():
    payload = {
        "lift": "snatch",
        "score": 8,
        "band": "solid",
        "max_horizontal_drift": "small forward loop past mid-foot at turnover",
        "cues": ["keep the bar close to the body"],
        "confidence": 0.76,
        "notes": "Good extension, minor drift corrected.",
    }
    text = json.dumps(payload)
    j = parse_and_validate_json(text)
    assert isinstance(j, LiftJudgment)
    assert j.lift == "snatch"
    assert j.band == "solid"


def test_parse_and_validate_json_with_fences():
    text = "```json\n" + json.dumps(
        {
            "lift": "clean",
            "score": 6,
            "band": "mixed",
            "max_horizontal_drift": "noticeable away from shins early",
            "cues": ["finish the legs/hips (triple extension) before the arms take over"],
            "confidence": 0.55,
        }
    ) + "\n```"
    j = parse_and_validate_json(text)
    assert j.lift == "clean"
    assert j.band == "mixed"


def test_validate_cues_count_enforced():
    bad = json.dumps(
        {
            "lift": "jerk",
            "score": 7,
            "band": "solid",
            "max_horizontal_drift": "—",
            "cues": [],
            "confidence": 0.8,
        }
    )
    with pytest.raises((ValidationError, ValueError)):
        parse_and_validate_json(bad)


def test_band_unknown_allows_empty_cues():
    ok = json.dumps(
        {
            "lift": "snatch",
            "score": None,
            "band": "unknown",
            "max_horizontal_drift": "",
            "cues": [],
            "confidence": None,
            "notes": "bar not visible",
        }
    )
    j = parse_and_validate_json(ok)
    assert j.band == "unknown"
    assert j.cues == []


def test_parse_skips_reasoning_prose_with_braces():
    # Reasoning models sometimes narrate the schema (braces included) before answering.
    text = (
        "Let me think. The schema is { \"lift\": ..., \"score\": ... } and the bar "
        'drifts forward {not an object}, so:\n'
        '{"lift":"clean","score":6,"band":"mixed","max_horizontal_drift":"forward loop in transition",'
        '"cues":["keep the bar close to the body"],"confidence":0.6}\n'
        "That is my answer."
    )
    j = parse_and_validate_json(text)
    assert j.lift.value == "clean"
    assert j.score == 6


def test_parse_raises_value_error_on_truncated_reasoning():
    # Judge truncated mid-thought (finish_reason=length) -> no JSON at all.
    with pytest.raises(ValueError):
        parse_and_validate_json("The user wants me to analyze the observations. Key points about bar path")

