"""The AI coach draft: valid JSON is kept, anything else falls back to the template, and the budget holds."""
from __future__ import annotations

import json

import pytest

from server.services import coach_draft
from server.services.budgets import reset_budgets_for_tests

LIFT = {
    "index": 1,
    "lift": "snatch",
    "score": 6,
    "band": "mixed",
    "confidence": 0.5,
    "cues": ["Keep the bar closer off the floor", "Finish the pull before you dive under"],
    "warnings": ["Camera looks off-axis: drift may be overstated."],
    "drift": {"text": "about 12 cm forward in the second pull", "source": "measurement"},
    "evidenced_cues": [{"id": "c1", "text": "Keep the bar closer off the floor", "phase": "first_pull", "why": "12 cm forward at 2.0–2.9 s"}],
    "bar_path": {"tracked": True, "max_drift_cm_estimate": 12, "points": [[0, 0, 0]]},
}

GOOD = json.dumps(
    {
        "summary": "Strong pull. The bar drifted forward off the floor; keep it over mid-foot.",
        "cues": [{"text": "Push the knees back as the bar leaves the floor", "evidence_ref": "c1"}],
        "drill": {"name": "Pause snatch pull to the knee", "sets_reps": "4x3 at 70%"},
        "confidence_note": "The camera was a little off-axis, so the drift number is a guide.",
        "suggested_score": 6,
        "suggested_band": "mixed",
    }
)


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.meter = None

    async def create_chat_completion(self, messages, model=None, max_tokens=800, extra_body=None):
        self.calls.append({"messages": messages, "model": model})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def _budgets(monkeypatch):
    monkeypatch.delenv("DRAFT_DAILY_LIMIT", raising=False)
    reset_budgets_for_tests()
    yield
    reset_budgets_for_tests()


async def _draft(client, **kwargs):
    return await coach_draft.generate_draft(client=client, lift=LIFT, note=kwargs.pop("note", "my left knee caves"), prompts_dir="prompts", model="m", coach_name="Coach Maya", **kwargs)


def test_a_valid_reply_becomes_the_draft():
    import asyncio

    client = FakeClient([GOOD])
    draft, source = asyncio.run(_draft(client))
    assert source == "model" and draft["source"] == "model"
    assert draft["cues"][0]["evidence_ref"] == "c1" and draft["suggested_band"] == "mixed"
    sent = client.calls[0]["messages"]
    assert sent[0]["role"] == "system" and "JSON only" in sent[0]["content"]
    user = sent[1]["content"]
    assert '"athlete_note": "my left knee caves"' in user and '"points"' not in user, "the note is quoted data; the raw points are not sent"


def test_prose_around_the_json_is_tolerated():
    import asyncio

    draft, source = asyncio.run(_draft(FakeClient(["Here you go:\n" + GOOD + "\nHope that helps"])))
    assert source == "model" and draft["summary"].startswith("Strong pull")


def test_invalid_json_falls_back_to_the_template():
    import asyncio

    draft, source = asyncio.run(_draft(FakeClient(['{"summary": "no cues here"}'])))
    assert source == "template" and draft["source"] == "template"
    assert draft["cues"][0]["text"] == "Keep the bar closer off the floor" and draft["cues"][0]["evidence_ref"] == "c1"
    assert "6/10 (mixed)" in draft["summary"] and "12 cm forward" in draft["summary"]
    assert draft["confidence_note"].startswith("Read with care: Camera looks off-axis")
    assert draft["suggested_score"] == 6 and draft["suggested_band"] == "mixed"


def test_a_model_error_falls_back_too():
    import asyncio

    draft, source = asyncio.run(_draft(FakeClient([RuntimeError("Nebius API error 503: busy")])))
    assert source == "template"


def test_the_draft_budget_stops_the_call(monkeypatch):
    import asyncio

    monkeypatch.setenv("DRAFT_DAILY_LIMIT", "1")
    reset_budgets_for_tests()
    client = FakeClient([GOOD, GOOD])
    _, first = asyncio.run(_draft(client))
    coach_draft.draft_budget().charge()  # the one call of the day
    _, second = asyncio.run(_draft(client))
    assert first == "model" and second == "template"
    assert len(client.calls) == 1


def test_the_template_needs_no_model_at_all():
    draft = coach_draft.template_draft({"lift": "jerk", "score": 8, "band": "solid", "cues": ["Drive straight up"]})
    assert draft["cues"] == [{"text": "Drive straight up", "evidence_ref": "c1"}]
    assert draft["summary"].startswith("Nice jerk.") and draft["suggested_score"] == 8
    empty = coach_draft.template_draft({})
    assert empty["cues"] and empty["summary"]
