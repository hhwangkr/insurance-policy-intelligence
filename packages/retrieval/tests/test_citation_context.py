from __future__ import annotations

from pathlib import Path

import pytest
from demo_corpus_fixtures import (
    INSURER_KYOBO,
    KYOBO_ANNUITY_DOCUMENT_ID,
    KYOBO_ANNUITY_SYNTHETIC_DOCUMENT_ID,
    PRODUCT_TYPE_ANNUITY,
)
from test_retrieval_baseline import FakeEmbedder, _artifact, _chunk

from insurance_ai_retrieval.citation_context import (
    build_citation_context,
    citation_bundle_from_hits,
    format_citation_bundle_json,
    search_filters_to_mapping,
)
from insurance_ai_retrieval.index_engine import SearchFilters, SearchHit, build_local_index
from insurance_ai_retrieval.metadata import ChunkMetadataRecord, enrich_chunk_metadata

# ``KYOBO_ANNUITY_*`` constants are staged-demo shapes for metadata + filter wiring (see
# ``demo_corpus_fixtures``), not benchmarks and not a rule that every new PDF needs tests here.


def test_search_filters_to_mapping_roundtrip_keys() -> None:
    f = SearchFilters(
        insurer=INSURER_KYOBO,
        product_type=PRODUCT_TYPE_ANNUITY,
        variant_name="적립형",
        include_section_types=frozenset({"article", "appendix"}),
        exclude_section_types=frozenset({"toc"}),
        use_default_section_type_excludes=False,
    )
    m = search_filters_to_mapping(f)
    assert m["insurer"] == INSURER_KYOBO
    assert m["product_type"] == PRODUCT_TYPE_ANNUITY
    assert m["variant_name"] == "적립형"
    assert set(m["include_section_types"]) == {"appendix", "article"}
    assert m["exclude_section_types"] == ["toc"]


def test_citation_bundle_from_hits_ids_and_fields() -> None:
    ch = _chunk(
        document_id=KYOBO_ANNUITY_DOCUMENT_ID,
        chunk_id="doc::c1",
        section_id="doc::s1",
        section_title="제1조 (목적)",
        text="본문 일부",
        page_start=3,
        page_end=3,
        char_start=100,
        char_end=200,
    )
    meta = enrich_chunk_metadata(ChunkMetadataRecord.from_document_chunk(ch))
    hits = [
        SearchHit(rank=1, score=0.91, metadata=meta),
        SearchHit(
            rank=2,
            score=0.42,
            metadata=meta.model_copy(update={"chunk_id": "doc::c2", "text": "둘째"}),
        ),
    ]
    filters = SearchFilters(insurer=INSURER_KYOBO, product_type=PRODUCT_TYPE_ANNUITY)
    bundle = citation_bundle_from_hits(
        query="테스트 질의",
        filters=filters,
        top_k=5,
        dedupe_section=True,
        hits=hits,
    )
    assert bundle.query == "테스트 질의"
    assert bundle.top_k == 5
    assert bundle.dedupe_section is True
    assert len(bundle.citations) == 2
    assert bundle.citations[0].citation_id == "C1"
    assert bundle.citations[1].citation_id == "C2"
    c0 = bundle.citations[0]
    assert c0.score == pytest.approx(0.91)
    assert c0.chunk_id == "doc::c1"
    assert c0.section_id == "doc::s1"
    assert c0.section_title == "제1조 (목적)"
    assert c0.insurer == INSURER_KYOBO
    assert c0.product_type == PRODUCT_TYPE_ANNUITY
    assert c0.insurer_display_name == "교보생명"
    assert c0.product_type_display_name == "연금보험"
    assert c0.product_display_name is not None
    assert c0.text == "본문 일부"
    assert c0.page_start == 3 and c0.page_end == 3
    assert c0.char_start == 100 and c0.char_end == 200


def test_format_citation_bundle_json_utf8() -> None:
    bundle = citation_bundle_from_hits(
        query="해약",
        filters=SearchFilters(variant_name="적립형"),
        top_k=3,
        dedupe_section=False,
        hits=[],
    )
    s = format_citation_bundle_json(bundle)
    assert "해약" in s
    assert "\n" in s


def test_build_citation_context_end_to_end(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    doc = KYOBO_ANNUITY_SYNTHETIC_DOCUMENT_ID
    art = _artifact(
        document_id=doc,
        chunks=[
            _chunk(
                document_id=doc,
                chunk_id=f"{doc}::chunk::0000::000",
                section_id=f"{doc}::sec::0000",
                text="보험금 지연 이자 약관 문구",
                char_start=0,
                char_end=20,
            ),
            _chunk(
                document_id=doc,
                chunk_id=f"{doc}::chunk::0001::000",
                section_id=f"{doc}::sec::0001",
                text="청약 철회 안내",
                char_start=21,
                char_end=40,
                chunk_index=0,
            ),
        ],
    )
    (chunks_dir / "x.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    embedder = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=embedder, batch_size=8)

    bundle = build_citation_context(
        index_dir=index_dir,
        query="지연 이자",
        embedder=embedder,
        filters=SearchFilters(insurer=INSURER_KYOBO, product_type=PRODUCT_TYPE_ANNUITY),
        top_k=2,
        dedupe_section=False,
    )
    assert len(bundle.citations) == 2
    assert bundle.citations[0].citation_id == "C1"
    texts = {bundle.citations[0].text, bundle.citations[1].text}
    assert any("이자" in t for t in texts)


def test_build_citation_context_dedupe_section(tmp_path: Path) -> None:
    """Two chunks same section_id: dedupe keeps one row in bundle."""
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    doc = KYOBO_ANNUITY_SYNTHETIC_DOCUMENT_ID
    sid = f"{doc}::sec::same"
    art = _artifact(
        document_id=doc,
        chunks=[
            _chunk(
                document_id=doc,
                chunk_id=f"{doc}::chunk::0000::000",
                section_id=sid,
                text="alpha chunk",
                char_start=0,
                char_end=5,
            ),
            _chunk(
                document_id=doc,
                chunk_id=f"{doc}::chunk::0000::001",
                section_id=sid,
                text="beta chunk same section",
                char_start=6,
                char_end=30,
                chunk_index=1,
            ),
        ],
    )
    (chunks_dir / "x.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    embedder = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=embedder, batch_size=8)

    bundle = build_citation_context(
        index_dir=index_dir,
        query="alpha",
        embedder=embedder,
        filters=SearchFilters(insurer=INSURER_KYOBO, product_type=PRODUCT_TYPE_ANNUITY),
        top_k=5,
        dedupe_section=True,
    )
    assert len(bundle.citations) == 1
    assert bundle.citations[0].citation_id == "C1"
