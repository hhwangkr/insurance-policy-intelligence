from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from insurance_ai_retrieval.citation_context import CitationContextBundle

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


def _unique_citation_ids_preserve_order(ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in ids:
        cid = raw.strip()
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def grounded_answer_json_schema_for_citations(
    allowed_citation_ids: Sequence[str],
) -> dict[str, Any]:
    """JSON Schema for ``GroundedAnswer`` with ``citations_used`` constrained to allowed IDs."""
    unique = _unique_citation_ids_preserve_order(allowed_citation_ids)
    if not unique:
        return grounded_answer_json_schema()
    return {
        "type": "object",
        "required": ["answer", "citations_used", "insufficient_context"],
        "properties": {
            "answer": {"type": "string"},
            "citations_used": {
                "type": "array",
                "items": {"type": "string", "enum": unique},
            },
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


def render_grounded_answer_with_citations(answer: GroundedAnswer) -> str:
    """Format ``answer.answer`` for display by appending missing ``[Cn]`` markers.

    When ``insufficient_context`` is True, returns the prose unchanged. Otherwise, for each id
    in ``citations_used`` (list order, de-duplicated by first occurrence), appends a ``[id]``
    token only if that id is not already present as a ``[C…]`` marker in the body. Does not add
    ids beyond ``citations_used``. Display-only; does not change validation semantics.
    """
    if answer.insufficient_context:
        return answer.answer
    body = answer.answer.rstrip()
    if not answer.citations_used:
        return body
    existing = extract_citation_ids_from_text(body)
    ordered = _unique_citation_ids_preserve_order(answer.citations_used)
    missing = [cid for cid in ordered if cid not in existing]
    if not missing:
        return body
    suffix = "".join(f"[{cid}]" for cid in missing)
    if body:
        return f"{body} {suffix}"
    return suffix


def render_citation_summary(answer: GroundedAnswer, context_bundle: CitationContextBundle) -> str:
    """Human-readable lines for each ``citations_used`` id: ``[C2] title, page_start-page_end``.

    Looks up metadata only from ``context_bundle``; order follows ``citations_used``. When
    ``insufficient_context`` is True or ``citations_used`` is empty, returns an empty string.
    Missing bundle rows render as ``[Cx] (not in context bundle)`` so output stays deterministic.
    """
    if answer.insufficient_context or not answer.citations_used:
        return ""
    by_id = {e.citation_id: e for e in context_bundle.citations}
    lines: list[str] = []
    for cid in _unique_citation_ids_preserve_order(answer.citations_used):
        entry = by_id.get(cid)
        if entry is None:
            lines.append(f"[{cid}] (not in context bundle)")
        else:
            lines.append(f"[{cid}] {entry.section_title}, {entry.page_start}-{entry.page_end}")
    return "\n".join(lines)


def validate_answer_citations(
    answer: GroundedAnswer,
    allowed_citation_ids: set[str],
    *,
    require_inline_markers: bool = False,
) -> GroundedAnswerValidation:
    """Check ``citations_used`` and optional inline ``[Cn]`` markers against allowed IDs."""
    ids_in_text = extract_citation_ids_from_text(answer.answer)
    used_set = set(answer.citations_used)
    errors: list[str] = []

    for cid in sorted(ids_in_text - allowed_citation_ids):
        errors.append(f"answer text cites id not in allowed set: [{cid}]")

    for cid in sorted(used_set - allowed_citation_ids):
        errors.append(f"citations_used contains id not in allowed set: {cid}")

    if not answer.insufficient_context:
        if not answer.answer.strip():
            errors.append("answer must not be empty when insufficient_context is False")
        if require_inline_markers:
            if not ids_in_text:
                errors.append(
                    "answer must include at least one citation marker like [C1] when "
                    "insufficient_context is False",
                )
            if used_set != ids_in_text:
                errors.append(
                    "citations_used does not match citation markers in answer text: "
                    f"citations_used={sorted(used_set)} text_markers={sorted(ids_in_text)}",
                )
        elif not answer.citations_used:
            errors.append(
                "citations_used must be non-empty when insufficient_context is False "
                "(structured citation mode)",
            )

    return GroundedAnswerValidation(
        is_valid=not errors,
        errors=errors,
        citation_ids_in_text=sorted(ids_in_text),
        allowed_citation_ids=sorted(allowed_citation_ids),
    )
