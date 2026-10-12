from __future__ import annotations

import json
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator


class LiftType(str, Enum):
    snatch = "snatch"
    clean = "clean"
    jerk = "jerk"


Band = Literal["elite", "solid", "mixed", "poor", "fail", "unknown"]


class LiftJudgment(BaseModel):
    lift: LiftType
    score: Optional[int] = Field(default=None, ge=1, le=10)
    band: Band
    max_horizontal_drift: str = Field(default="")
    cues: List[str] = Field(default_factory=list)
    cues_he: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_cues(self) -> "LiftJudgment":
        # Allow 0 cues only when band unknown (media problem)
        if self.band != "unknown":
            if not (1 <= len(self.cues) <= 2):
                raise ValueError("cues must contain 1 or 2 items")
            if not (1 <= len(self.cues_he) <= 2):
                raise ValueError("cues_he must contain 1 or 2 items")
        return self


def parse_and_validate_json(text: str) -> LiftJudgment:
    """
    Parse raw model text as JSON and validate against LiftJudgment schema.
    Attempts to extract a JSON object if surrounding prose or fences slipped in.
    """
    # Fast path: direct JSON
    try:
        obj = json.loads(text)
        return LiftJudgment.model_validate(obj)
    except (json.JSONDecodeError, ValidationError):
        pass

    # Attempt to extract the first {...} object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        obj = json.loads(snippet)  # Let JSON errors raise
        return LiftJudgment.model_validate(obj)

    # Unable to parse
    raise ValueError("Response did not contain a valid JSON object")

