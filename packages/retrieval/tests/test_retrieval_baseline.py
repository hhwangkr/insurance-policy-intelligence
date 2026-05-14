from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from insurance_ai_retrieval.chunks_io import flatten_chunks_sorted, load_chunk_artifacts_from_dir
from insurance_ai_retrieval.e5_text import format_e5_passage, format_e5_query
from insurance_ai_retrieval.index_engine import (
    SearchFilters,
    build_local_index,
    load_index_config,
    load_metadata_rows,
    search_local_index,
)
from insurance_ai_retrieval.metadata import ChunkMetadataRecord, enrich_chunk_metadata
from insurance_ai_shared.models.chunk import ChunkingConfig, DocumentChunk, DocumentChunksArtifact


def _dt() -> datetime:
    return datetime(2026, 5, 14, tzinfo=UTC)


def _chunk(**overrides: object) -> DocumentChunk:
    data: dict[str, object] = {
        "document_id": "doc_a",
        "chunk_id": "doc_a::chunk::0000::000",
        "section_id": "doc_a::sec::0000",
        "section_type": "article",
        "section_title": "제1조",
        "parent_section_id": None,
        "policy_unit_id": None,
        "policy_unit_name": None,
        "variant_name": None,
        "chunk_index": 0,
        "text": "alpha",
        "page_start": 1,
        "page_end": 1,
        "char_start": 10,
        "char_end": 15,
        "char_count": 5,
        "token_estimate": 2,
        "chunking_strategy": "section_aware_paragraph_v1",
        "source_section_char_start": 0,
        "source_section_char_end": 5,
    }
    data.update(overrides)
    return DocumentChunk.model_validate(data)


def _artifact(*, document_id: str, chunks: list[DocumentChunk]) -> DocumentChunksArtifact:
    return DocumentChunksArtifact(
        document_id=document_id,
        source_sections_created_at=_dt(),
        generated_at=_dt(),
        chunks=chunks,
        chunking_config=ChunkingConfig(),
    )


class FakeEmbedder:
    """Deterministic fake vectors (no model download)."""

    def __init__(self, *, dim: int = 8) -> None:
        self.model_name = "fake/test"
        self._dim = dim

    def encode_passages(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        del batch_size
        mat = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, s in enumerate(texts):
            seed = sum(ord(c) for c in s) + i * 17
            for j in range(self._dim):
                mat[i, j] = float((seed + j * 31) % 251) / 250.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        out: np.ndarray = (mat / norms).astype(np.float32)
        return out

    def encode_query(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for j in range(self._dim):
            v[j] = float((len(text) + j * 13) % 251) / 250.0
        n = float(np.linalg.norm(v))
        out: np.ndarray = (v / max(n, 1e-9)).astype(np.float32)
        return out


def test_load_chunk_artifacts_from_dir(tmp_path: Path) -> None:
    d = tmp_path / "chunks"
    d.mkdir()
    art = _artifact(
        document_id="doc_z",
        chunks=[_chunk(document_id="doc_z", chunk_id="doc_z::c1", text="hello")],
    )
    (d / "doc_z.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    loaded = load_chunk_artifacts_from_dir(d)
    assert len(loaded) == 1
    assert loaded[0].document_id == "doc_z"


def test_load_chunk_artifacts_missing_dir_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="chunks directory not found"):
        load_chunk_artifacts_from_dir(missing)


def test_load_chunk_artifacts_empty_dir_raises(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(ValueError, match=r"no \*\.chunks\.json"):
        load_chunk_artifacts_from_dir(d)


def test_flatten_chunks_sorted_order() -> None:
    a = _chunk(
        document_id="doc_b",
        chunk_id="doc_b::2",
        char_start=20,
        char_end=25,
        text="b",
    )
    b = _chunk(
        document_id="doc_b",
        chunk_id="doc_b::1",
        char_start=10,
        char_end=15,
        text="a",
    )
    c = _chunk(
        document_id="doc_a",
        chunk_id="doc_a::1",
        char_start=1,
        char_end=2,
        text="z",
    )
    art_b = _artifact(document_id="doc_b", chunks=[a, b])
    art_a = _artifact(document_id="doc_a", chunks=[c])
    flat = flatten_chunks_sorted([art_b, art_a])
    assert [x.chunk_id for x in flat] == ["doc_a::1", "doc_b::1", "doc_b::2"]


def test_chunk_metadata_record_from_document_chunk() -> None:
    ch = _chunk(
        document_id="kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62",
    )
    row = ChunkMetadataRecord.from_document_chunk(ch)
    assert row.chunk_id == ch.chunk_id
    assert row.section_id == ch.section_id
    assert row.policy_unit_id is ch.policy_unit_id
    assert row.insurer == "kyobolife"
    assert row.product_type == "annuity"
    assert row.product_name == "kyobo ro annuity insurance"


def test_e5_query_and_passage_format_helpers() -> None:
    assert format_e5_query("hello").startswith("query:")
    assert format_e5_query("query: x") == "query: x"
    assert format_e5_passage("body").startswith("passage:")
    assert format_e5_passage("passage: y") == "passage: y"


def test_build_local_index_empty_chunks_raises(tmp_path: Path) -> None:
    d = tmp_path / "chunks"
    d.mkdir()
    empty_art = _artifact(document_id="doc_e", chunks=[])
    (d / "doc_e.chunks.json").write_text(empty_art.model_dump_json(), encoding="utf-8")
    embedder = FakeEmbedder()
    with pytest.raises(ValueError, match="no chunks found"):
        build_local_index(chunks_dir=d, index_dir=tmp_path / "idx", embedder=embedder, batch_size=8)


def test_build_and_search_roundtrip(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    art = _artifact(
        document_id="doc_r",
        chunks=[
            _chunk(
                document_id="doc_r",
                chunk_id="doc_r::chunk::0000::000",
                text="보험금 지급 지연 이자",
                char_start=0,
                char_end=10,
            ),
            _chunk(
                document_id="doc_r",
                chunk_id="doc_r::chunk::0001::000",
                section_id="doc_r::sec::0001",
                text="청약 철회 기간 안내",
                char_start=11,
                char_end=22,
                chunk_index=0,
            ),
        ],
    )
    (chunks_dir / "doc_r.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")

    emb = FakeEmbedder()
    cfg = build_local_index(
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        embedder=emb,
        batch_size=4,
    )
    assert cfg.num_chunks == 2
    assert cfg.model_name == "fake/test"

    hits = search_local_index(
        index_dir=index_dir,
        query="보험금",
        embedder=emb,
        top_k=2,
    )
    assert len(hits) == 2
    assert hits[0].rank == 1
    assert 0.0 <= hits[0].score <= 1.0001


def test_search_result_includes_citation_fields(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    doc_id = "kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62"
    ch = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::0000::000",
        policy_unit_id=f"{doc_id}::pu::1",
        variant_name="적립형",
        policy_unit_name="상품",
    )
    art = _artifact(document_id=doc_id, chunks=[ch])
    (chunks_dir / f"{doc_id}.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    emb = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=emb, batch_size=8)
    hits = search_local_index(index_dir=index_dir, query="q", embedder=emb, top_k=1)
    m = hits[0].metadata
    assert m.section_id == ch.section_id
    assert m.section_title == ch.section_title
    assert m.section_type == ch.section_type
    assert m.document_id == ch.document_id
    assert m.insurer == "kyobolife"
    assert m.product_type == "annuity"
    assert m.policy_unit_id == ch.policy_unit_id
    assert m.variant_name == ch.variant_name
    assert m.page_start == ch.page_start
    assert m.page_end == ch.page_end
    assert m.char_start == ch.char_start
    assert m.char_end == ch.char_end


def test_search_model_mismatch_raises(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    art = _artifact(document_id="doc_m", chunks=[_chunk(document_id="doc_m", chunk_id="doc_m::1")])
    (chunks_dir / "doc_m.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    build_local_index(
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        embedder=FakeEmbedder(),
        batch_size=8,
    )

    class OtherFake(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__(dim=8)
            self.model_name = "other/model"

    with pytest.raises(ValueError, match="does not match index"):
        search_local_index(
            index_dir=index_dir,
            query="x",
            embedder=OtherFake(),
            top_k=1,
        )


def test_load_index_config_roundtrip(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    art = _artifact(document_id="doc_cfg", chunks=[_chunk(document_id="doc_cfg", chunk_id="x")])
    (chunks_dir / "doc_cfg.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    build_local_index(
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        embedder=FakeEmbedder(),
        batch_size=8,
    )
    cfg = load_index_config(index_dir)
    assert cfg.backend == "numpy_normalized_dot"
    rows = load_metadata_rows(index_dir)
    assert len(rows) == 1


def test_default_section_filters_exclude_toc(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    doc_id = "kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62"
    toc = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::toc::000",
        section_id=f"{doc_id}::sec::toc",
        section_type="toc",
        text="alpha toc only",
    )
    body = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::body::000",
        section_id=f"{doc_id}::sec::body",
        section_type="article",
        text="alpha article body",
    )
    art = _artifact(document_id=doc_id, chunks=[toc, body])
    (chunks_dir / f"{doc_id}.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    emb = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=emb, batch_size=8)
    hits = search_local_index(index_dir=index_dir, query="alpha", embedder=emb, top_k=5)
    assert hits
    assert all(h.metadata.section_type != "toc" for h in hits)


def test_search_filters_by_insurer_and_product_type(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    kyobo = "kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62"
    cancer = "samsunglife_cancer_internet_cancer_insurance_policy_terms_20260101_39af0c18"
    k_chunk = _chunk(
        document_id=kyobo,
        chunk_id=f"{kyobo}::chunk::0000::000",
        section_id=f"{kyobo}::sec::0000",
        text="청약 철회",
    )
    s_chunk = _chunk(
        document_id=cancer,
        chunk_id=f"{cancer}::chunk::0000::000",
        section_id=f"{cancer}::sec::0000",
        text="청약 철회",
    )
    (chunks_dir / f"{kyobo}.chunks.json").write_text(
        _artifact(document_id=kyobo, chunks=[k_chunk]).model_dump_json(),
        encoding="utf-8",
    )
    (chunks_dir / f"{cancer}.chunks.json").write_text(
        _artifact(document_id=cancer, chunks=[s_chunk]).model_dump_json(),
        encoding="utf-8",
    )
    emb = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=emb, batch_size=8)
    hits = search_local_index(
        index_dir=index_dir,
        query="청약 철회",
        embedder=emb,
        top_k=5,
        filters=SearchFilters(insurer="kyobolife", product_type="annuity"),
    )
    assert len(hits) == 1
    assert hits[0].metadata.document_id == kyobo


def test_search_filters_by_document_id(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    doc_id = "samsunglife_whole_life_balance_whole_life_insurance_policy_terms_20260301_1699395d"
    a = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::0000::000",
        section_id=f"{doc_id}::sec::0000",
        text="해약환급금",
    )
    b = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::0001::000",
        section_id=f"{doc_id}::sec::0001",
        text="other",
    )
    art = _artifact(document_id=doc_id, chunks=[a, b])
    (chunks_dir / f"{doc_id}.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    emb = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=emb, batch_size=8)
    hits = search_local_index(
        index_dir=index_dir,
        query="해약환급금",
        embedder=emb,
        top_k=5,
        filters=SearchFilters(document_id=doc_id),
    )
    assert {h.metadata.chunk_id for h in hits} == {a.chunk_id, b.chunk_id}


def test_search_filters_by_product_name_substring(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    doc_id = "samsunglife_cancer_internet_cancer_insurance_policy_terms_20260101_39af0c18"
    ch = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::0000::000",
        section_id=f"{doc_id}::sec::0000",
        text="암",
    )
    art = _artifact(document_id=doc_id, chunks=[ch])
    (chunks_dir / f"{doc_id}.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    emb = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=emb, batch_size=8)
    hits = search_local_index(
        index_dir=index_dir,
        query="암",
        embedder=emb,
        top_k=5,
        filters=SearchFilters(product_name="internet"),
    )
    assert len(hits) == 1


def test_search_filters_by_variant_name(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    doc_id = "kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62"
    match = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::0000::000",
        section_id=f"{doc_id}::sec::0000",
        variant_name="적립형",
        text="보험금",
    )
    miss = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::0001::000",
        section_id=f"{doc_id}::sec::0001",
        variant_name="즉시형",
        text="보험금",
    )
    art = _artifact(document_id=doc_id, chunks=[match, miss])
    (chunks_dir / f"{doc_id}.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    emb = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=emb, batch_size=8)
    hits = search_local_index(
        index_dir=index_dir,
        query="보험금",
        embedder=emb,
        top_k=5,
        filters=SearchFilters(variant_name="적립"),
    )
    assert len(hits) == 1
    assert hits[0].metadata.variant_name == "적립형"


def test_search_empty_filtered_candidates_raises(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    doc_id = "kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62"
    ch = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::0000::000",
        section_id=f"{doc_id}::sec::0000",
        text="x",
    )
    art = _artifact(document_id=doc_id, chunks=[ch])
    (chunks_dir / f"{doc_id}.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    emb = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=emb, batch_size=8)
    with pytest.raises(ValueError, match="no chunks matched metadata and section filters"):
        search_local_index(
            index_dir=index_dir,
            query="x",
            embedder=emb,
            top_k=5,
            filters=SearchFilters(insurer="not_a_real_insurer"),
        )


def test_dedupe_section_returns_one_chunk_per_section(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    doc_id = (
        "miraeassetlife_variable_annuity_variable_annuity_insurance_policy_terms_20260401_76283b26"
    )
    sec = f"{doc_id}::sec::dup"
    first = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::0000::000",
        section_id=sec,
        text="dup section chunk a",
        char_start=0,
        char_end=10,
    )
    second = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::0000::001",
        section_id=sec,
        text="dup section chunk b",
        char_start=11,
        char_end=22,
        chunk_index=1,
    )
    art = _artifact(document_id=doc_id, chunks=[first, second])
    (chunks_dir / f"{doc_id}.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    emb = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=emb, batch_size=8)
    hits = search_local_index(
        index_dir=index_dir,
        query="dup",
        embedder=emb,
        top_k=5,
        dedupe_section=True,
    )
    assert len(hits) == 1


def test_exclude_section_types_removes_matches(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    index_dir = tmp_path / "index"
    chunks_dir.mkdir()
    doc_id = "kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62"
    appendix = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::apx::000",
        section_id=f"{doc_id}::sec::apx",
        section_type="appendix",
        text="target text",
    )
    article = _chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}::chunk::art::000",
        section_id=f"{doc_id}::sec::art",
        section_type="article",
        text="other",
    )
    art = _artifact(document_id=doc_id, chunks=[appendix, article])
    (chunks_dir / f"{doc_id}.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    emb = FakeEmbedder()
    build_local_index(chunks_dir=chunks_dir, index_dir=index_dir, embedder=emb, batch_size=8)
    hits = search_local_index(
        index_dir=index_dir,
        query="target",
        embedder=emb,
        top_k=5,
        filters=SearchFilters(
            use_default_section_type_excludes=False,
            exclude_section_types=frozenset({"appendix"}),
        ),
    )
    assert len(hits) == 1
    assert hits[0].metadata.section_type == "article"


def test_enrich_chunk_metadata_restores_derivation_from_document_id() -> None:
    doc_id = "kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62"
    ch = _chunk(document_id=doc_id)
    base = ChunkMetadataRecord.from_document_chunk(ch)
    legacy = base.model_copy(update={"insurer": None, "product_type": None, "product_name": None})
    restored = enrich_chunk_metadata(legacy)
    assert restored.insurer == "kyobolife"
    assert restored.product_type == "annuity"
    assert restored.product_name == "kyobo ro annuity insurance"
