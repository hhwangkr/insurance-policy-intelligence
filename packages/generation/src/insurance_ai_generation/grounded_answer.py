from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_CITATION_BRACKET_RE: re.Pattern[str] = re.compile(r"\[C(\d+)\]")


def grounded_answer_json_schema() -> dict[str, Any]:
    """JSON Schema for ``GroundedAnswer``-shaped model output (Ollama ``format`` field)."""
    return {
        "type": "object",
        "required": ["answer", "citations_used", "insufficient_context"],
        "properties": {
            "answer": {"type": "string"},
            "citations_used": {"type": "array", "items": {"type": "string"}},
            "insufficient_context": {"type": "boolean"},
        },
    }


class GroundedAnswer(BaseModel):
    """Structured answer payload for a future LLM step (no provider here)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    answer: str
    citations_used: list[str] = Field(default_factory=list)
    insufficient_context: bool = False


class GroundedAnswerValidation(BaseModel):
    """Result of local citation checks (no LLM)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    is_valid: bool
    errors: list[str]
    citation_ids_in_text: list[str]
    allowed_citation_ids: list[str]


def extract_citation_ids_from_text(answer: str) -> set[str]:
    """Return citation handles like ``C1``, ``C10`` found as ``[C1]``, ``[C10]`` in text."""
    return {f"C{m}" for m in _CITATION_BRACKET_RE.findall(answer)}


def validate_answer_citations(
    answer: GroundedAnswer,
    allowed_citation_ids: set[str],
) -> GroundedAnswerValidation:
    """Check answer text and ``citations_used`` against allowed retrieval citation IDs."""
    ids_in_text = extract_citation_ids_from_text(answer.answer)
    used_set = set(answer.citations_used)
    errors: list[str] = []

    if not answer.insufficient_context:
        if not answer.answer.strip():
            errors.append("answer must not be empty when insufficient_context is False")
        if not ids_in_text:
            errors.append(
                "answer must include at least one citation marker like [C1] when "
                "insufficient_context is False",
            )

    for cid in sorted(used_set - allowed_citation_ids):
        errors.append(f"citations_used contains id not in allowed set: {cid}")

    for cid in sorted(ids_in_text - allowed_citation_ids):
        errors.append(f"answer text cites id not in allowed set: [{cid}]")

    if used_set != ids_in_text:
        errors.append(
            "citations_used does not match citation markers in answer text: "
            f"citations_used={sorted(used_set)} text_markers={sorted(ids_in_text)}",
        )

    return GroundedAnswerValidation(
        is_valid=not errors,
        errors=errors,
        citation_ids_in_text=sorted(ids_in_text),
        allowed_citation_ids=sorted(allowed_citation_ids),
    )
