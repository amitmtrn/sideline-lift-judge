from __future__ import annotations

import json
from typing import Iterator, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


def iter_json_object_candidates(text: str) -> Iterator[str]:
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


def parse_model_json(text: str, model: Type[ModelT], *, what: str = "JSON object") -> ModelT:
    """
    Parse raw model text as JSON and validate against `model`.
    Extracts a JSON object if surrounding prose or fences slipped in.
    """
    # Fast path: direct JSON
    try:
        return model.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        pass

    # Reasoning models can wrap the object in prose that itself contains braces,
    # so try every balanced {...} candidate and keep the first one that validates.
    last_error: Optional[Exception] = None
    for snippet in iter_json_object_candidates(text):
        try:
            return model.model_validate(json.loads(snippet))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc

    if last_error is not None:
        raise ValueError(f"Response did not contain a valid {what}: {last_error}")
    raise ValueError(f"Response did not contain a valid {what}")
