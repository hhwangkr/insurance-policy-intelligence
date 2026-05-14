from __future__ import annotations

from pathlib import Path

from insurance_ai_generation.answer_prompt import (
    build_grounded_answer_prompt,
    format_grounded_answer_prompt_json,
)
from insurance_ai_retrieval.citation_context import CitationContextBundle, CitationContextEntry


def _entry(**kwargs: object) -> CitationContextEntry:
    defaults: dict[str, object] = {
        "citation_id": "C1",
        "chunk_id": "doc::c1",
        "section_id": "doc::s1",
        "section_title": "제1조 (목적)",
        "section_type": "article",
        "document_id": "doc_policy_terms_20260101_a1b2c3d4",
        "insurer": "kyobolife",
        "product_type": "annuity",
        "policy_unit_name": "개인연금",
        "variant_name": "적립형",
        "page_start": 2,
        "page_end": 3,
        "char_start": 0,
        "char_end": 50,
        "score": 0.88,
        "text": "본문 텍스트",
    }
    defaults.update(kwargs)
    return CitationContextEntry.model_validate(defaults)


def _bundle(**kwargs: object) -> CitationContextBundle:
    defaults: dict[str, object] = {
        "query": "해약환급금은요?",
        "filters": {"insurer": "kyobolife", "product_type": "annuity"},
        "top_k": 5,
        "dedupe_section": True,
        "citations": [_entry()],
    }
    defaults.update(kwargs)
    return CitationContextBundle.model_validate(defaults)


def test_build_grounded_answer_prompt_system_requires_json_only() -> None:
    p = build_grounded_answer_prompt(_bundle())
    system = p.messages[0].content
    assert "Return ONLY valid JSON" in system
    assert "insufficient_context" in system
    b = _bundle(
        citations=[
            _entry(citation_id="C1", section_title="제A조", page_start=1, page_end=1),
            _entry(
                citation_id="C2",
                chunk_id="doc::c2",
                section_id="doc::s2",
                section_title="제B조",
                text="둘째",
                page_start=4,
                page_end=5,
            ),
        ],
    )
    p = build_grounded_answer_prompt(b)
    user = p.messages[1].content
    assert "[C1]" in user and "[C2]" in user
    assert "제A조" in user and "제B조" in user
    assert "pages: 1-1" in user
    assert "pages: 4-5" in user


def test_prompt_grounding_and_insufficient_instructions() -> None:
    p = build_grounded_answer_prompt(_bundle())
    system = p.messages[0].content
    assert "provided citation" in system.lower()
    assert "do not guess" in system.lower() or "guess" in system.lower()
    assert "insufficient" in system.lower()


def test_prompt_preserves_korean_query() -> None:
    q = "보험금 지급이 늦어지면 이자는 어떻게 계산돼?"
    p = build_grounded_answer_prompt(_bundle(query=q))
    assert p.query == q
    assert f"[Question]\n{q}" in p.messages[1].content


def test_prompt_empty_citations() -> None:
    p = build_grounded_answer_prompt(_bundle(citations=[]))
    assert p.citation_ids == []
    user = p.messages[1].content
    assert "No citation passages" in user
    assert "insufficient" in user.lower()
    assert "[C1]" not in user


def test_prompt_deterministic_json() -> None:
    b = _bundle(
        citations=[
            _entry(citation_id="C1", text="x"),
            _entry(
                citation_id="C2",
                chunk_id="d::2",
                section_id="d::s2",
                section_title="제2조",
                text="y",
                char_start=1,
                char_end=2,
            ),
        ],
        filters={"z": 1, "a": 0},
    )
    p1 = build_grounded_answer_prompt(b)
    p2 = build_grounded_answer_prompt(b)
    assert format_grounded_answer_prompt_json(p1) == format_grounded_answer_prompt_json(p2)


def test_answer_prompt_module_has_no_llm_client_import() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "src" / "insurance_ai_generation" / "answer_prompt.py"
    )
    src = path.read_text(encoding="utf-8")
    banned = ("openai", "anthropic", "cohere", "litellm", "google.generativeai")
    lower = src.lower()
    for name in banned:
        assert name not in lower, f"unexpected token {name!r} in answer_prompt.py"
