from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from insurance_ai_retrieval.index_engine import SearchHit
from insurance_ai_retrieval.metadata import ChunkMetadataRecord
from insurance_ai_retrieval.retrieval_evaluation import (
    EvalRunSummary,
    QueryEvalResult,
    QueryFilters,
    RetrievalQuerySpec,
    all_hits_match_filters,
    evaluate_hits,
    expectation_met,
    hit_at_rank_prefix,
    load_retrieval_queries,
    render_markdown_report,
    title_matches_any,
    write_report,
)

# Demo-corpus document_id strings are synthetic fixtures for stable metadata shapes (not a
# requirement those products always remain in-repo). Curated section-title expectations live in
# data/eval/retrieval_queries.yaml.

_KYOBO_DOC = "kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62"


def _meta(
    *,
    section_title: str,
    section_type: str = "article",
    document_id: str = _KYOBO_DOC,
) -> ChunkMetadataRecord:
    return ChunkMetadataRecord(
        chunk_id="c1",
        section_id="s1",
        section_title=section_title,
        section_type=section_type,
        document_id=document_id,
        insurer="kyobolife",
        product_type="annuity",
        product_name="kyobo ro annuity insurance",
        policy_unit_id=None,
        policy_unit_name=None,
        variant_name=None,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=10,
        text="x",
    )


def _hit(rank: int, *, title: str, section_type: str = "article") -> SearchHit:
    return SearchHit(
        rank=rank, score=0.9, metadata=_meta(section_title=title, section_type=section_type)
    )


def test_title_matches_any_whitespace_variants() -> None:
    assert title_matches_any("( 별표 3 )", ["( 별표 3 )"])
    assert title_matches_any("(  별표  3  )", ["( 별표 3 )"])
    assert title_matches_any("(별표3)", ["( 별표 3 )"])


def test_expectation_met_types_only() -> None:
    spec = RetrievalQuerySpec(
        id="types_only",
        query="q",
        filters=QueryFilters(insurer="kyobolife", product_type="annuity"),
        expected_any_section_titles=[],
        expected_any_section_types=["appendix"],
        top_k=3,
    )
    meta = _meta(section_title="anything", section_type="appendix")
    assert expectation_met(meta, spec) is True


def test_hit_at_rank_prefix_respects_depth() -> None:
    spec = RetrievalQuerySpec(
        id="t",
        query="q",
        filters=QueryFilters(insurer="kyobolife", product_type="annuity"),
        expected_any_section_titles=["wanted"],
        top_k=5,
    )
    hits = [
        _hit(1, title="nope"),
        _hit(2, title="nope"),
        _hit(3, title="wanted"),
    ]
    assert hit_at_rank_prefix(hits, spec, 1) is False
    assert hit_at_rank_prefix(hits, spec, 3) is True
    assert hit_at_rank_prefix(hits, spec, 5) is True


def test_expectation_met_with_section_types() -> None:
    spec = RetrievalQuerySpec(
        id="types",
        query="q",
        filters=QueryFilters(insurer="kyobolife", product_type="annuity"),
        expected_any_section_titles=["표"],
        expected_any_section_types=["appendix"],
        top_k=3,
    )
    meta_ok = _meta(section_title="표", section_type="appendix")
    meta_wrong_type = _meta(section_title="표", section_type="article")
    assert expectation_met(meta_ok, spec) is True
    assert expectation_met(meta_wrong_type, spec) is False


def test_load_retrieval_queries_roundtrip(tmp_path: Path) -> None:
    payload = {
        "queries": [
            {
                "id": "q1",
                "query": "hello",
                "filters": {"insurer": "kyobolife", "product_type": "annuity"},
                "expected_any_section_titles": ["제1조"],
                "top_k": 3,
            }
        ]
    }
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = load_retrieval_queries(path)
    assert len(loaded) == 1
    assert loaded[0].id == "q1"
    assert loaded[0].top_k == 3


def test_load_retrieval_queries_rejects_empty_expectations(tmp_path: Path) -> None:
    payload = {
        "queries": [
            {
                "id": "bad",
                "query": "hello",
                "filters": {"insurer": "kyobolife", "product_type": "annuity"},
                "expected_any_section_titles": [],
                "top_k": 3,
            }
        ]
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid retrieval queries YAML"):
        load_retrieval_queries(path)


def test_evaluate_hits_marks_filter_mismatch() -> None:
    spec = RetrievalQuerySpec(
        id="f",
        query="q",
        filters=QueryFilters(insurer="kyobolife", product_type="annuity"),
        expected_any_section_titles=["제1조"],
        top_k=2,
    )
    filters = spec.to_search_filters()
    good = _hit(1, title="제1조")
    bad_meta = ChunkMetadataRecord(
        chunk_id="c2",
        section_id="s2",
        section_title="other",
        section_type="article",
        document_id="samsunglife_cancer_internet_cancer_insurance_policy_terms_20260101_39af0c18",
        insurer="samsunglife",
        product_type="cancer",
        product_name="internet cancer insurance",
        policy_unit_id=None,
        policy_unit_name=None,
        variant_name=None,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=10,
        text="x",
    )
    bad = SearchHit(rank=2, score=0.5, metadata=bad_meta)
    r = evaluate_hits([good, bad], spec, filters=filters)
    assert r.pass_at_1 is True
    assert r.all_filters_matched is False


def test_all_hits_match_filters_true_when_consistent() -> None:
    spec = RetrievalQuerySpec(
        id="ok",
        query="q",
        filters=QueryFilters(insurer="kyobolife", product_type="annuity"),
        expected_any_section_titles=["제1조"],
        top_k=2,
    )
    filters = spec.to_search_filters()
    hits = [_hit(1, title="제1조"), _hit(2, title="제2조")]
    assert all_hits_match_filters(hits, filters) is True


def test_render_markdown_report_contains_sections(tmp_path: Path) -> None:
    summary = EvalRunSummary(
        total_queries=1,
        hit_at_1=1.0,
        hit_at_3=1.0,
        hit_at_5=1.0,
        per_query=[
            QueryEvalResult(
                query_id="q1",
                pass_at_1=True,
                pass_at_3=True,
                pass_at_5=True,
                all_filters_matched=True,
                top_section_titles=["제1조"],
                top_section_types=["article"],
                hit_detail="ok",
            )
        ],
    )
    md = render_markdown_report(
        summary, index_dir=tmp_path / "idx", queries_path=tmp_path / "q.yaml"
    )
    assert "total_queries" in md
    assert "`q1`" in md
    assert "hit@1" in md


def test_write_report_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "out.md"
    write_report(out, "# hi\n")
    assert out.read_text(encoding="utf-8").startswith("# hi")


def test_hit_at_rank_prefix_empty_hits() -> None:
    spec = RetrievalQuerySpec(
        id="empty",
        query="q",
        filters=QueryFilters(insurer="kyobolife", product_type="annuity"),
        expected_any_section_titles=["x"],
        top_k=5,
    )
    assert hit_at_rank_prefix([], spec, 5) is False
