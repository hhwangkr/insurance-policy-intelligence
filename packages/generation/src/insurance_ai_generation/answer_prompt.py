from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from insurance_ai_retrieval.citation_context import CitationContextBundle, CitationContextEntry


class ChatMessage(BaseModel):
    """One role/content pair for chat-style LLM APIs."""

    model_config = ConfigDict(str_strip_whitespace=True)

    role: Literal["system", "user"]
    content: str = Field(..., min_length=1)


class GroundedAnswerPrompt(BaseModel):
    """Deterministic messages for a future grounded answer call (no LLM here)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    messages: list[ChatMessage]
    query: str
    citation_ids: list[str]


def _display_optional(value: str | None) -> str:
    return value if value is not None else "n/a"


_VALID_JSON_EXAMPLE = "\n".join(
    [
        "Valid structured JSON example (citations_used lists passages; inline [C3] optional here):",
        "{",
        '  "answer": "보험금 지급 지연 이자는 별표 3의 기준에 따라 계산됩니다.",',
        '  "citations_used": ["C3"],',
        '  "insufficient_context": false',
        "}",
    ],
)

_INVALID_JSON_EXAMPLE = "\n".join(
    [
        "Invalid example (do not do this):",
        "{",
        '  "answer": "Some grounded answer text.",',
        '  "citations_used": [],',
        '  "insufficient_context": false',
        "}",
        "Reason: when insufficient_context is false, citations_used must list the C-style IDs "
        "you used (cannot be empty).",
    ],
)

_SYSTEM_INSTRUCTIONS = "\n".join(
    [
        "You are an insurance policy assistant working in strict retrieval-grounded mode.",
        "",
        "Rules:",
        "- Answer ONLY using information supported by the provided citation passages below. "
        "Treat each passage as policy text scoped by its metadata (insurer, product, variant, "
        "section).",
        "- Do NOT use outside knowledge, general legal advice, or information not present in the "
        "citations.",
        "- Do NOT guess or speculate. If the excerpts are insufficient to answer safely, set "
        "insufficient_context to true and explain briefly in answer; do not invent details.",
        "",
        "Language:",
        "- Answer in the same language as the user's question.",
        "- Do not switch languages unless the user explicitly asks for another language.",
        "",
        "Citations:",
        "- Valid citation IDs for JSON are ONLY the handles listed in the user message under "
        '"VALID CITATION IDS" (e.g. C1, C2).',
        '- In JSON, citations_used must contain ONLY those IDs as plain strings like "C1", '
        '"C2". Do not translate, rename, or substitute them.',
        "- Do NOT put section titles, article names, or clause labels in citations_used. "
        "Use only C-style IDs from the list.",
        "- When insufficient_context is false, citations_used must be non-empty and list every "
        "passage ID you relied on (structured citation contract).",
        "- Inline square-bracket markers like [C1], [C2] in the answer are preferred for "
        "readability but not mandatory in the default contract.",
        "- If you include inline markers, use ONLY ASCII square brackets exactly like [C1], "
        "[C2]. Do NOT use (C1), (C1 clause refs), bare C1, section titles, article names, or "
        "page numbers instead of [C1]-style markers.",
        "- Downstream display may append missing [Cn] markers deterministically from "
        "citations_used; still list correct IDs in citations_used.",
        "",
        "Citation selection (retrieval order):",
        "- Passages are provided in retrieval rank order (earlier passages are typically more "
        "similar to the question). Prefer higher-ranked passages when they are relevant.",
        "- Do not ignore a top-ranked passage that directly answers the question in favor of "
        "lower-ranked ones unless it is clearly irrelevant or you explain why the excerpts you "
        "use supersede it.",
        "- For questions about calculation methods, interest or yield schedules, rates or "
        "multipliers, eligibility or payment criteria, tables or appendices, or defined terms "
        "used in those schedules, pay special attention to appendix/table-style excerpts when "
        "they contain the concrete rule, schedule, or definition.",
        "- When an article clause points to an appendix/table passage for the actual numbers or "
        "table, and that appendix/table is among the provided citations, include both IDs in "
        "citations_used when both are needed for a precise answer.",
        "- Avoid broad regulatory, industry, or generic legal generalizations that are not stated "
        "in the cited passages; keep claims tied to what those excerpts actually say.",
        "",
        "Output format (mandatory):",
        "- Return ONLY valid JSON. Do not wrap it in markdown fences; do not add any text before "
        "or after the JSON object.",
        '- Use exactly these keys: "answer", "citations_used", "insufficient_context".',
        '- If the excerpts are insufficient to answer, set "insufficient_context": true and '
        "give a brief explanation in answer; citations_used may be empty.",
        "",
        _VALID_JSON_EXAMPLE,
        "",
        _INVALID_JSON_EXAMPLE,
        "",
        "The user message contains [Question], retrieval filter context, VALID CITATION IDS, "
        "and [Citations]. Stay within those passages only.",
    ],
)


def _format_citation_block(c: CitationContextEntry) -> str:
    lines = [
        f"[{c.citation_id}]",
        f"section_title: {c.section_title}",
        f"section_type: {c.section_type}",
        f"insurer: {_display_optional(c.insurer)}",
        f"product_type: {_display_optional(c.product_type)}",
        f"policy_unit_name: {_display_optional(c.policy_unit_name)}",
        f"variant_name: {_display_optional(c.variant_name)}",
        f"pages: {c.page_start}-{c.page_end}",
        "text:",
        c.text,
        "",
    ]
    return "\n".join(lines)


def _build_user_content(bundle: CitationContextBundle) -> str:
    parts: list[str] = [
        "[Question]",
        bundle.query,
        "",
        "[Retrieval filters (metadata scope)]",
        json.dumps(bundle.filters, ensure_ascii=False, sort_keys=True),
        "",
        "[Citations]",
    ]
    if not bundle.citations:
        parts.append(
            "(No citation passages were retrieved. The provided policy excerpts are insufficient "
            "as given; do not fabricate citations or policy text.)",
        )
        parts.append("")
    else:
        id_list = ", ".join(c.citation_id for c in bundle.citations)
        parts.extend(
            [
                "VALID CITATION IDS:",
                id_list,
                "",
                "Use only these IDs. Do not translate them.",
                "",
                "Valid citation IDs are ONLY the tokens above. "
                "Do not use any other citation names, section titles, or article numbers as "
                "citation identifiers.",
                "",
            ],
        )
        for c in bundle.citations:
            parts.append(_format_citation_block(c))
    return "\n".join(parts).rstrip() + "\n"


def build_grounded_answer_prompt(bundle: CitationContextBundle) -> GroundedAnswerPrompt:
    """Build chat messages from a query-time bundle (pure function; no network)."""
    citation_ids = [c.citation_id for c in bundle.citations]
    messages = [
        ChatMessage(role="system", content=_SYSTEM_INSTRUCTIONS),
        ChatMessage(role="user", content=_build_user_content(bundle)),
    ]
    return GroundedAnswerPrompt(messages=messages, query=bundle.query, citation_ids=citation_ids)


def format_grounded_answer_prompt_json(prompt: GroundedAnswerPrompt) -> str:
    """JSON suitable for stdout / debug logs (UTF-8 friendly)."""
    payload = {
        "query": prompt.query,
        "citation_ids": prompt.citation_ids,
        "messages": [m.model_dump() for m in prompt.messages],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
