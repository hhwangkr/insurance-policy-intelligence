from __future__ import annotations

from pathlib import Path

from insurance_ai_generation.grounded_answer import (
    GroundedAnswer,
    extract_citation_ids_from_text,
    validate_answer_citations,
)


def test_extract_citation_ids_including_c10() -> None:
    text = "참고 [C1] 및 [C10] 끝 [C2]"
    assert extract_citation_ids_from_text(text) == {"C1", "C2", "C10"}


def test_valid_korean_answer_with_citations() -> None:
    ga = GroundedAnswer(
        answer="지연 이자는 [C1]과 [C2]에 따르며, 기간별 이율이 정리되어 있습니다.",
        citations_used=["C1", "C2"],
    )
    v = validate_answer_citations(ga, {"C1", "C2", "C3"})
    assert v.is_valid
    assert v.errors == []
    assert v.citation_ids_in_text == ["C1", "C2"]
    assert v.allowed_citation_ids == ["C1", "C2", "C3"]


def test_invented_citation_in_text_fails() -> None:
    ga = GroundedAnswer(
        answer="잘못된 인용 [C9]입니다.",
        citations_used=["C9"],
    )
    v = validate_answer_citations(ga, {"C1", "C2", "C3", "C4", "C5"})
    assert not v.is_valid
    assert any("not in allowed set" in e for e in v.errors)


def test_citations_used_not_matching_text_fails() -> None:
    ga = GroundedAnswer(
        answer="첫째만 [C1] 언급.",
        citations_used=["C1", "C2"],
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert not v.is_valid
    assert any("does not match citation markers" in e for e in v.errors)


def test_text_citation_not_listed_in_citations_used_fails() -> None:
    ga = GroundedAnswer(
        answer="둘 다 필요 [C1][C2].",
        citations_used=["C1"],
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert not v.is_valid
    assert any("does not match citation markers" in e for e in v.errors)


def test_citations_used_id_missing_from_text_fails() -> None:
    ga = GroundedAnswer(
        answer="하나만 [C1].",
        citations_used=["C1", "C2"],
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert not v.is_valid


def test_empty_answer_fails_when_not_insufficient() -> None:
    ga = GroundedAnswer(answer="   ", citations_used=[], insufficient_context=False)
    v = validate_answer_citations(ga, {"C1"})
    assert not v.is_valid
    assert any("must not be empty" in e for e in v.errors)


def test_no_citation_markers_fails_when_not_insufficient() -> None:
    ga = GroundedAnswer(
        answer="근거 없이 길게 썼지만 대괄호 인용이 없습니다.",
        citations_used=[],
        insufficient_context=False,
    )
    v = validate_answer_citations(ga, {"C1"})
    assert not v.is_valid
    assert any("at least one citation marker" in e for e in v.errors)


def test_insufficient_context_allows_empty_no_citations() -> None:
    ga = GroundedAnswer(
        answer="",
        citations_used=[],
        insufficient_context=True,
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert v.is_valid
    assert v.citation_ids_in_text == []


def test_validation_sorted_lists_deterministic() -> None:
    ga = GroundedAnswer(
        answer="[C2] 다음 [C1]",
        citations_used=["C2", "C1"],
    )
    v1 = validate_answer_citations(ga, {"C3", "C1", "C2"})
    v2 = validate_answer_citations(ga, {"C3", "C1", "C2"})
    assert v1.model_dump() == v2.model_dump()
    assert v1.citation_ids_in_text == ["C1", "C2"]
    assert v1.allowed_citation_ids == ["C1", "C2", "C3"]


def test_grounded_answer_module_no_llm_strings() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "insurance_ai_generation"
        / "grounded_answer.py"
    )
    src = path.read_text(encoding="utf-8").lower()
    for token in ("openai", "anthropic", "cohere", "litellm"):
        assert token not in src
