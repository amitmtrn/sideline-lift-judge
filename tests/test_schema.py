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
        "cues_he": ["מוט צמוד לגוף"],
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
            "cues_he": ["דחוף את הרצפה חזק ומהר"],
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
            "cues_he": [],
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
            "cues_he": [],
            "confidence": None,
            "notes": "bar not visible",
        }
    )
    j = parse_and_validate_json(ok)
    assert j.band == "unknown"
    assert j.cues == []

