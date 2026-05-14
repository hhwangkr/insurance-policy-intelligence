from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from insurance_ai_generation.generate_grounded_answer import GroundedAnswerGenerationResult
from insurance_ai_generation.grounded_answer import (
    GroundedAnswer,
    GroundedAnswerValidation,
    render_citation_summary,
    render_grounded_answer_with_citations,
)
from insurance_ai_retrieval.citation_context import CitationContextBundle, CitationContextEntry

_TEXT_PREVIEW_LIMIT = 500


def _preview_text(text: str, *, limit: int = _TEXT_PREVIEW_LIMIT) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


class CitationContextPreview(BaseModel):
    """One retrieved passage slice for inspection (no full RAG / LLM)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    citation_id: str
    rank: int = Field(ge=1, description="1-based retrieval rank in the bundle order.")
    score: float
    section_title: str
    section_type: str
    page_start: int
    page_end: int
    text_preview: str


class GroundedAnswerInspection(BaseModel):
    """JSON-serializable manual QA payload: bundle + generation + cited/uncited split."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str
    filters: dict[str, Any]
    provider_name: str
    model_name: str | None
    validation_is_valid: bool
    validation_errors: list[str]
    raw_answer: str
    rendered_answer: str
    citations_used: list[str]
    citation_summary: str
    all_citation_contexts: list[CitationContextPreview]
    cited_contexts: list[CitationContextPreview]
    uncited_contexts: list[CitationContextPreview]
    manual_review_notes: str = ""


def _entry_to_preview(entry: CitationContextEntry, *, rank: int) -> CitationContextPreview:
    return CitationContextPreview(
        citation_id=entry.citation_id,
        rank=rank,
        score=float(entry.score),
        section_title=entry.section_title,
        section_type=entry.section_type,
        page_start=entry.page_start,
        page_end=entry.page_end,
        text_preview=_preview_text(entry.text),
    )


def build_grounded_answer_inspection(
    bundle: CitationContextBundle,
    result: GroundedAnswerGenerationResult,
) -> GroundedAnswerInspection:
    """Assemble cited vs uncited retrieval rows and display fields (deterministic, no LLM)."""
    used = {cid.strip() for cid in result.answer.citations_used if cid.strip()}
    all_previews: list[CitationContextPreview] = []
    cited: list[CitationContextPreview] = []
    uncited: list[CitationContextPreview] = []
    for i, entry in enumerate(bundle.citations, start=1):
        prev = _entry_to_preview(entry, rank=i)
        all_previews.append(prev)
        if entry.citation_id in used:
            cited.append(prev)
        else:
            uncited.append(prev)
    return GroundedAnswerInspection(
        query=bundle.query,
        filters=dict(bundle.filters),
        provider_name=result.provider_name,
        model_name=result.model_name,
        validation_is_valid=result.validation.is_valid,
        validation_errors=list(result.validation.errors),
        raw_answer=result.answer.answer,
        rendered_answer=render_grounded_answer_with_citations(result.answer),
        citations_used=list(result.answer.citations_used),
        citation_summary=render_citation_summary(result.answer, bundle),
        all_citation_contexts=all_previews,
        cited_contexts=cited,
        uncited_contexts=uncited,
        manual_review_notes="",
    )


def generation_result_from_generate_answer_json(
    data: dict[str, Any],
) -> GroundedAnswerGenerationResult:
    """Parse stdout JSON from ``generate_answer`` (extra keys like ``rendered_answer`` ignored)."""
    required = ("answer", "validation", "provider_name", "raw_text")
    missing = [k for k in required if k not in data]
    if missing:
        msg = f"generation JSON missing keys: {missing}"
        raise ValueError(msg)
    model_raw = data.get("model_name")
    model_name: str | None = None if model_raw is None else str(model_raw)
    return GroundedAnswerGenerationResult(
        answer=GroundedAnswer.model_validate(data["answer"]),
        validation=GroundedAnswerValidation.model_validate(data["validation"]),
        provider_name=str(data["provider_name"]),
        model_name=model_name,
        raw_text=str(data["raw_text"]),
    )


def _markdown_context_block(ctx: CitationContextPreview) -> str:
    lines = [
        f"- **{ctx.citation_id}** (rank {ctx.rank}, score {ctx.score:.6f})",
        f"  - {ctx.section_title} · {ctx.section_type} · p.{ctx.page_start}–{ctx.page_end}",
        f"  - Preview: {ctx.text_preview}",
    ]
    return "\n".join(lines) + "\n"


def inspection_report_to_markdown(report: GroundedAnswerInspection) -> str:
    """Human-readable report for manual faithfulness review (no LLM)."""
    parts: list[str] = [
        "# Grounded answer inspection\n",
        "## Run metadata\n",
        f"- **Query:** {report.query}",
        f"- **Provider:** {report.provider_name}",
        f"- **Model:** {report.model_name or '(none)'}",
        f"- **validation.is_valid:** {report.validation_is_valid}",
        "",
        "## Retrieval filters (bundle snapshot)\n",
        "```json",
        json.dumps(report.filters, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Validation errors\n",
        (
            "(none)"
            if not report.validation_errors
            else "\n".join(f"- {e}" for e in report.validation_errors)
        ),
        "",
        "## Citations used\n",
        ", ".join(report.citations_used) if report.citations_used else "(empty)",
        "",
        "## Citation summary\n",
        "```",
        report.citation_summary or "(empty)",
        "```",
        "",
        "## Raw answer\n",
        "```",
        report.raw_answer,
        "```",
        "",
        "## Rendered answer\n",
        "```",
        report.rendered_answer,
        "```",
        "",
        "## Cited contexts\n",
    ]
    if not report.cited_contexts:
        parts.append("(none)\n")
    else:
        for c in report.cited_contexts:
            parts.append(_markdown_context_block(c))
    parts.append("## Uncited contexts (retrieved but not in citations_used)\n")
    if not report.uncited_contexts:
        parts.append("(none)\n")
    else:
        for c in report.uncited_contexts:
            parts.append(_markdown_context_block(c))
    parts.append("## All retrieved contexts (ranked)\n")
    for c in report.all_citation_contexts:
        parts.append(_markdown_context_block(c))
    parts.extend(
        [
            "## Manual review notes\n",
            report.manual_review_notes or "(fill in during human QA)",
            "",
        ],
    )
    return "\n".join(parts) + "\n"
