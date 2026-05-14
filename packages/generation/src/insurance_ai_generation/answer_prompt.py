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
        "- Do NOT guess or speculate. If the excerpts are insufficient to answer safely, state "
        "clearly that the provided policy excerpts are insufficient. Do not invent details.",
        "- When you support a claim with evidence, cite using the exact labels given "
        "(e.g. [C1], [C2]). Do NOT invent citation IDs.",
        "- Wording and coverage differ by insurer, product, and variant; do not generalize unless "
        "the citations justify it.",
        "- Reply in the SAME language as the user's question unless they explicitly ask for "
        "another language.",
        "",
        "Output format (mandatory):",
        "- Return ONLY valid JSON. Do not wrap it in markdown fences; do not add any text before "
        "or after the JSON object.",
        '- Use exactly these keys: "answer", "citations_used", "insufficient_context".',
        '- The "answer" string MUST include at least one bracket citation like [C1] that matches '
        "a passage you used, unless you set insufficient_context to true.",
        '- "citations_used" MUST list every citation ID (e.g. C1, C2) that appears as [C1], '
        "[C2] in answer, in any order, with no extras.",
        '- If the excerpts are insufficient to answer, set "insufficient_context": true and '
        "give a brief explanation in answer; citations_used may be empty.",
        "",
        "The next user message contains [Question], retrieval filter context, and [Citations]. "
        "Stay within those passages only.",
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
