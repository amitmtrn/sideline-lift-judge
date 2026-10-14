from __future__ import annotations

import json
from enum import Enum
from typing import Iterator, List, Literal, Optional

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
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_cues(self) -> "LiftJudgment":
        # Allow 0 cues only when band unknown (media problem)
        if self.band != "unknown":
            if not (1 <= len(self.cues) <= 2):
                raise ValueError("cues must contain 1 or 2 items")
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

    # Reasoning models can wrap the object in prose that itself contains braces,
    # so try every balanced {...} candidate and keep the first one that validates.
    last_error: Optional[Exception] = None
    for snippet in _iter_json_object_candidates(text):
        try:
            return LiftJudgment.model_validate(json.loads(snippet))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc

    # Unable to parse
    if last_error is not None:
        raise ValueError(f"Response did not contain a valid judgment JSON object: {last_error}")
    raise ValueError("Response did not contain a valid JSON object")


def _iter_json_object_candidates(text: str) -> Iterator[str]:
    """
    Yield the balanced {...} substring starting at each opening brace, in order.
    String literals are skipped so braces and escapes inside them do not affect
    the depth count.
    """
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            ch = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break

