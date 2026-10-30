"""
The AI coach draft: the review the coach starts from instead of a blank page.

One Nemotron call per paid review, made once when the hold is placed and
stored on the review — never regenerated on a page load. The input is the
judgment JSON the athlete already saw (no new vision call), the measurement,
the clip warnings and the athlete's note, passed as quoted data. The output
is validated against a fixed schema; anything else, a model error, or a
spent DRAFT_DAILY_LIMIT falls back to a deterministic template built from
the judge's own cues, so the coach always has something to edit and the
sample's payment loop never depends on the model.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from server.schemas.json_parse import parse_model_json
from server.services.budgets import BudgetExceeded, draft_budget
from server.services.nebius_client import NO_THINKING_EXTRA_BODY, NebiusChatClient, judge_model

logger = logging.getLogger(__name__)

PROMPT_FILE = "coach_draft.txt"
DRAFT_MAX_TOKENS = 700
BANDS = ("elite", "solid", "mixed", "poor", "fail")


class DraftCue(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    evidence_ref: str = Field(default="", max_length=40)


class DraftDrill(BaseModel):
    name: str = Field(default="", max_length=200)
    sets_reps: str = Field(default="", max_length=100)


class CoachDraft(BaseModel):
    summary: str = Field(min_length=1, max_length=1200)
    cues: List[DraftCue] = Field(min_length=1, max_length=3)
    drill: DraftDrill = Field(default_factory=DraftDrill)
    confidence_note: str = Field(default="", max_length=600)
    suggested_score: Optional[int] = Field(default=None, ge=1, le=10)
    suggested_band: Optional[str] = Field(default=None)

    def as_dict(self, *, source: str) -> Dict[str, Any]:
        body = self.model_dump()
        band = (body.get("suggested_band") or "").strip().lower()
        body["suggested_band"] = band if band in BANDS else None
        body["source"] = source
        return body


def parse_draft(text: str) -> CoachDraft:
    return parse_model_json(text, CoachDraft, what="coach draft JSON object")


def _lift_summary_for_prompt(lift: Dict[str, Any], note: Optional[str]) -> str:
    """The verdict as the draft model sees it: the fields that matter, nothing job-bound."""
    keep = {
        key: lift.get(key)
        for key in ("lift", "score", "band", "confidence", "max_horizontal_drift", "cues", "notes", "warnings", "drift")
        if lift.get(key) is not None
    }
    cues = []
    for cue in lift.get("evidenced_cues") or []:
        if isinstance(cue, dict):
            cues.append({k: cue.get(k) for k in ("id", "text", "phase", "issue", "why", "verified", "vision_only", "evidence") if cue.get(k) is not None})
    if cues:
        keep["evidenced_cues"] = cues
    bar_path = lift.get("bar_path")
    if isinstance(bar_path, dict):
        keep["measurement"] = {k: v for k, v in bar_path.items() if k != "points"}
    payload = {"verdict": keep}
    if note:
        payload["athlete_note"] = note[:500]
    return json.dumps(payload, ensure_ascii=False, indent=1)


def template_draft(lift: Dict[str, Any], *, coach_name: str = "your coach") -> Dict[str, Any]:
    """A draft with no model in it: the judge's cues, in a coach's framing."""
    lift_name = str(lift.get("lift") or "lift")
    score = lift.get("score")
    band = str(lift.get("band") or "").lower()
    cues_src: List[Dict[str, str]] = []
    for cue in lift.get("evidenced_cues") or []:
        if isinstance(cue, dict) and cue.get("text"):
            cues_src.append({"text": str(cue["text"]), "evidence_ref": str(cue.get("id") or cue.get("phase") or "")})
    if not cues_src:
        for position, cue in enumerate(lift.get("cues") or [], start=1):
            if isinstance(cue, str) and cue.strip():
                cues_src.append({"text": cue.strip(), "evidence_ref": f"c{position}"})
    if not cues_src:
        cues_src.append({"text": "Keep the bar close through the whole pull.", "evidence_ref": ""})
    cues_src = cues_src[:3]
    drift = lift.get("drift")
    drift_text = drift.get("text") if isinstance(drift, dict) else lift.get("max_horizontal_drift")
    opening = f"Nice {lift_name}." if band in ("elite", "solid") else f"Good effort on the {lift_name}."
    score_text = f" The bar path came out at {score}/10 ({band})." if score is not None and band else ""
    drift_line = f" {drift_text.rstrip('.')}." if isinstance(drift_text, str) and drift_text.strip() else ""
    main = cues_src[0]["text"].rstrip(".")
    summary = f"{opening}{score_text}{drift_line} The one thing to work on: {main.lower() if main[:1].isupper() and not main[:2].isupper() else main}."
    warnings = [w for w in lift.get("warnings") or [] if isinstance(w, str)]
    confidence = lift.get("confidence")
    if warnings:
        note = "Read with care: " + warnings[0].rstrip(".") + "."
    elif isinstance(confidence, (int, float)) and confidence < 0.6:
        note = "The AI read has modest confidence on this clip; treat the numbers as a guide, not a verdict."
    else:
        note = "A clean side-on clip; the measured drift is a fair guide."
    return {
        "summary": summary[:1200],
        "cues": cues_src,
        "drill": {"name": "Pause pulls to the knee, then finish", "sets_reps": "3x3 at 60%"},
        "confidence_note": note,
        "suggested_score": score if isinstance(score, int) else None,
        "suggested_band": band if band in BANDS else None,
        "source": "template",
    }


def draft_model() -> str:
    import os

    return (os.getenv("NEBIUS_DRAFT_MODEL") or "").strip() or judge_model()


async def generate_draft(
    *,
    client: NebiusChatClient,
    lift: Dict[str, Any],
    note: Optional[str],
    prompts_dir: str,
    model: Optional[str] = None,
    coach_name: str = "your coach",
) -> Tuple[Dict[str, Any], str]:
    """
    (draft, source). `source` is "model" when Nemotron's JSON validated,
    "template" otherwise — never an exception: the coach's page must open.
    """
    try:
        system_prompt = (Path(prompts_dir) / PROMPT_FILE).read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("Coach-draft prompt missing (%s); using the template", exc)
        return template_draft(lift, coach_name=coach_name), "template"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Coach: {coach_name}\n\n{_lift_summary_for_prompt(lift, note)}"},
    ]
    try:
        draft_budget().check()
        text = await client.create_chat_completion(
            messages=messages, model=model or draft_model(), max_tokens=DRAFT_MAX_TOKENS, extra_body=NO_THINKING_EXTRA_BODY
        )
        draft = parse_draft(text)
    except BudgetExceeded as exc:
        logger.info("Coach draft budget spent (%s); using the template", exc.detail)
        return template_draft(lift, coach_name=coach_name), "template"
    except (ValueError, ValidationError) as exc:
        logger.warning("Coach draft was not valid JSON (%s); using the template", exc)
        return template_draft(lift, coach_name=coach_name), "template"
    except Exception as exc:  # a provider error or timeout: the coach still gets a page
        logger.warning("Coach draft call failed (%s); using the template", exc)
        return template_draft(lift, coach_name=coach_name), "template"
    body = draft.as_dict(source="model")
    if body.get("suggested_score") is None and isinstance(lift.get("score"), int):
        body["suggested_score"] = lift["score"]
    if body.get("suggested_band") is None and str(lift.get("band") or "") in BANDS:
        body["suggested_band"] = str(lift["band"])
    return body, "model"
