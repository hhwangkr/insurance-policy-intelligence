from __future__ import annotations

from pathlib import Path

from insurance_ai_generation.grounded_answer import (
    GroundedAnswer,
    extract_citation_ids_from_text,
    grounded_answer_json_schema,
    grounded_answer_json_schema_for_citations,
    render_citation_summary,
    render_grounded_answer_with_citations,
    validate_answer_citations,
)
from insurance_ai_retrieval.citation_context import (
    CitationContextBundle,
    CitationContextEntry,
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


def test_structured_allows_citations_used_without_matching_text() -> None:
    ga = GroundedAnswer(
        answer="첫째만 [C1] 언급.",
        citations_used=["C1", "C2"],
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert v.is_valid


def test_structured_allows_extra_markers_in_text_not_listed_in_citations_used() -> None:
    ga = GroundedAnswer(
        answer="둘 다 필요 [C1][C2].",
        citations_used=["C1"],
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert v.is_valid


def test_structured_allows_citations_used_not_all_marked_in_text() -> None:
    ga = GroundedAnswer(
        answer="하나만 [C1].",
        citations_used=["C1", "C2"],
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert v.is_valid


def test_empty_answer_fails_when_not_insufficient() -> None:
    ga = GroundedAnswer(answer="   ", citations_used=["C1"], insufficient_context=False)
    v = validate_answer_citations(ga, {"C1"})
    assert not v.is_valid
    assert any("must not be empty" in e for e in v.errors)


def test_structured_requires_non_empty_citations_used() -> None:
    ga = GroundedAnswer(
        answer="근거 없이 길게 썼지만 citations_used가 비었습니다.",
        citations_used=[],
        insufficient_context=False,
    )
    v = validate_answer_citations(ga, {"C1"})
    assert not v.is_valid
    assert any("citations_used must be non-empty" in e for e in v.errors)


def test_structured_passes_without_inline_markers() -> None:
    ga = GroundedAnswer(
        answer="Explanation without any bracket markers.",
        citations_used=["C1"],
        insufficient_context=False,
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert v.is_valid


def test_inline_strict_requires_markers_and_match() -> None:
    ga = GroundedAnswer(
        answer="Explanation without any bracket markers.",
        citations_used=["C1"],
        insufficient_context=False,
    )
    v = validate_answer_citations(ga, {"C1", "C2"}, require_inline_markers=True)
    assert not v.is_valid
    assert any("at least one citation marker" in e for e in v.errors)


def test_inline_strict_citations_used_must_match_text() -> None:
    ga = GroundedAnswer(
        answer="첫째만 [C1] 언급.",
        citations_used=["C1", "C2"],
    )
    v = validate_answer_citations(ga, {"C1", "C2"}, require_inline_markers=True)
    assert not v.is_valid
    assert any("does not match citation markers" in e for e in v.errors)


def test_insufficient_context_allows_empty_no_citations() -> None:
    ga = GroundedAnswer(
        answer="",
        citations_used=[],
        insufficient_context=True,
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert v.is_valid
    assert v.citation_ids_in_text == []


def test_insufficient_context_rejects_disallowed_citation_in_used() -> None:
    ga = GroundedAnswer(
        answer="",
        citations_used=["C9"],
        insufficient_context=True,
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert not v.is_valid
    assert any("not in allowed set" in e for e in v.errors)


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


def test_grounded_answer_json_schema_shape() -> None:
    s = grounded_answer_json_schema()
    assert s["type"] == "object"
    assert set(s["required"]) == {"answer", "citations_used", "insufficient_context"}
    assert set(s["properties"]) == {"answer", "citations_used", "insufficient_context"}


def test_grounded_answer_json_schema_for_citations_enum() -> None:
    s = grounded_answer_json_schema_for_citations(["C2", "C1", "C2"])
    items = s["properties"]["citations_used"]["items"]
    assert items["type"] == "string"
    assert items["enum"] == ["C2", "C1"]


def test_grounded_answer_json_schema_for_citations_empty_falls_back() -> None:
    s = grounded_answer_json_schema_for_citations([])
    assert s == grounded_answer_json_schema()


def test_section_title_in_citations_used_invalid() -> None:
    ga = GroundedAnswer(
        answer="Some claim without markers.",
        citations_used=["第34条（保险合同贷款）"],
        insufficient_context=False,
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert not v.is_valid
    assert any("not in allowed set" in e for e in v.errors)


def test_structured_rejects_disallowed_marker_in_text() -> None:
    ga = GroundedAnswer(
        answer="Bad marker [C99] in text.",
        citations_used=["C1"],
        insufficient_context=False,
    )
    v = validate_answer_citations(ga, {"C1", "C2"})
    assert not v.is_valid
    assert any("answer text cites id not in allowed set" in e for e in v.errors)


def test_render_appends_missing_c3_in_citations_used_order() -> None:
    ga = GroundedAnswer(
        answer="보험금 지급 지연 이자 설명.",
        citations_used=["C3"],
        insufficient_context=False,
    )
    out = render_grounded_answer_with_citations(ga)
    assert out.endswith("[C3]")
    assert "보험금 지급 지연 이자 설명." in out


def test_render_deterministic_same_input() -> None:
    ga = GroundedAnswer(
        answer="Body.",
        citations_used=["C2", "C3"],
        insufficient_context=False,
    )
    assert render_grounded_answer_with_citations(ga) == render_grounded_answer_with_citations(ga)


def test_render_does_not_invent_beyond_citations_used() -> None:
    ga = GroundedAnswer(
        answer="Only C1 in text [C1].",
        citations_used=["C1"],
        insufficient_context=False,
    )
    out = render_grounded_answer_with_citations(ga)
    assert out == "Only C1 in text [C1]."


def test_render_insufficient_returns_body_unchanged() -> None:
    ga = GroundedAnswer(
        answer="No markers here.",
        citations_used=["C1"],
        insufficient_context=True,
    )
    assert render_grounded_answer_with_citations(ga) == "No markers here."


def test_render_preserves_existing_inline_when_all_ids_present() -> None:
    ga = GroundedAnswer(
        answer="See [C2][C3][C5].",
        citations_used=["C2", "C3", "C5"],
        insufficient_context=False,
    )
    assert render_grounded_answer_with_citations(ga) == "See [C2][C3][C5]."


def test_render_dedupes_citations_used_preserving_first_occurrence_order() -> None:
    ga = GroundedAnswer(
        answer="No markers.",
        citations_used=["C2", "C2", "C3"],
        insufficient_context=False,
    )
    assert render_grounded_answer_with_citations(ga) == "No markers. [C2][C3]"


def test_render_partial_inline_appends_only_missing_ids() -> None:
    ga = GroundedAnswer(
        answer="First [C2] only.",
        citations_used=["C2", "C3", "C5"],
        insufficient_context=False,
    )
    assert render_grounded_answer_with_citations(ga) == "First [C2] only. [C3][C5]"


def test_render_appends_multiple_in_list_order() -> None:
    ga = GroundedAnswer(
        answer="X.",
        citations_used=["C2", "C3"],
        insufficient_context=False,
    )
    assert render_grounded_answer_with_citations(ga) == "X. [C2][C3]"


def test_render_appends_c2_c3_c5_suffix() -> None:
    ga = GroundedAnswer(
        answer="EXAONE style prose without markers.",
        citations_used=["C2", "C3", "C5"],
        insufficient_context=False,
    )
    assert render_grounded_answer_with_citations(ga) == (
        "EXAONE style prose without markers. [C2][C3][C5]"
    )


def test_render_citation_summary_lines_from_bundle() -> None:
    bundle = CitationContextBundle(
        query="q",
        filters={},
        top_k=2,
        dedupe_section=False,
        citations=[
            CitationContextEntry(
                citation_id="C2",
                chunk_id="ck",
                section_id="sec",
                section_title="Article Two",
                section_type="article",
                document_id="doc",
                page_start=3,
                page_end=4,
                char_start=0,
                char_end=10,
                score=0.9,
                text="body",
            ),
            CitationContextEntry(
                citation_id="C3",
                chunk_id="ck2",
                section_id="sec2",
                section_title="Article Three",
                section_type="article",
                document_id="doc",
                page_start=10,
                page_end=10,
                char_start=0,
                char_end=5,
                score=0.8,
                text="b",
            ),
        ],
    )
    ga = GroundedAnswer(
        answer="x",
        citations_used=["C3", "C2"],
        insufficient_context=False,
    )
    out = render_citation_summary(ga, bundle)
    assert out == ("[C3] Article Three, 10-10\n[C2] Article Two, 3-4")


def test_render_citation_summary_insufficient_is_empty() -> None:
    bundle = CitationContextBundle(
        query="q",
        filters={},
        top_k=0,
        dedupe_section=False,
        citations=[],
    )
    ga = GroundedAnswer(
        answer="x",
        citations_used=["C1"],
        insufficient_context=True,
    )
    assert render_citation_summary(ga, bundle) == ""


def test_render_citation_summary_missing_id_in_bundle() -> None:
    bundle = CitationContextBundle(
        query="q",
        filters={},
        top_k=1,
        dedupe_section=False,
        citations=[
            CitationContextEntry(
                citation_id="C1",
                chunk_id="ck",
                section_id="sec",
                section_title="Only C1",
                section_type="article",
                document_id="doc",
                page_start=1,
                page_end=2,
                char_start=0,
                char_end=1,
                score=1.0,
                text="t",
            ),
        ],
    )
    ga = GroundedAnswer(
        answer="x",
        citations_used=["C9"],
        insufficient_context=False,
    )
    assert render_citation_summary(ga, bundle) == "[C9] (not in context bundle)"


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
