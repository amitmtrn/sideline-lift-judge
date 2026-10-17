from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from server.schemas.json_parse import iter_json_object_candidates, parse_model_json


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

    # Set by the server after the judge replies, never by the model itself:
    # clip-quality warnings (FR-8) and the rendered replay (FR-10).
    warnings: List[str] = Field(default_factory=list)
    annotated_video_url: Optional[str] = Field(default=None)

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
    return parse_model_json(text, LiftJudgment, what="judgment JSON object")


# Kept as an alias: the brace scanner lives in app.schemas.json_parse now.
_iter_json_object_candidates = iter_json_object_candidates

