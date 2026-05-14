from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from insurance_ai_ingestion.chunk_sections import chunks_json_path, main
from insurance_ai_ingestion.section_chunking import chunk_document_sections
from insurance_ai_shared.models.chunk import ChunkingConfig, DocumentChunksArtifact
from insurance_ai_shared.models.section import DocumentSection, DocumentSectionsArtifact


def _dt() -> datetime:
    return datetime(2026, 5, 13, tzinfo=UTC)


def _artifact(*, document_id: str, sections: list[DocumentSection]) -> DocumentSectionsArtifact:
    return DocumentSectionsArtifact(
        document_id=document_id,
        source_document_created_at=_dt(),
        sections=sections,
        created_at=_dt(),
    )


def test_short_article_becomes_one_chunk() -> None:
    doc_id = "doc_chunk_short"
    sec = DocumentSection(
        document_id=doc_id,
        section_id=f"{doc_id}::sec::0000",
        section_type="article",
        title="제1조 (목적)",
        start_page=1,
        end_page=1,
        start_char_offset=10,
        end_char_offset=10 + len("제1조(목적)\n짧은본문입니다."),
        parent_section_id=None,
        text="제1조(목적)\n짧은본문입니다.",
    )
    art = _artifact(document_id=doc_id, sections=[sec])
    out = chunk_document_sections(art, config=ChunkingConfig(max_chars=2200, overlap_chars=250))
    assert len(out.chunks) == 1
    c = out.chunks[0]
    assert c.text == sec.text
    assert c.section_id == sec.section_id
    assert c.chunk_index == 0
    assert c.char_start == 10 and c.char_end == 10 + len(sec.text)


def test_long_article_splits_into_multiple_chunks() -> None:
    doc_id = "doc_chunk_long"
    body = ("단락입니다.\n\n" * 400).strip() + "\n"
    text = "제1조(제목)\n" + body
    sec = DocumentSection(
        document_id=doc_id,
        section_id=f"{doc_id}::sec::0000",
        section_type="article",
        title="제1조 (긴조)",
        start_page=2,
        end_page=5,
        start_char_offset=1000,
        end_char_offset=1000 + len(text),
        parent_section_id=None,
        text=text,
    )
    art = _artifact(document_id=doc_id, sections=[sec])
    out = chunk_document_sections(art, config=ChunkingConfig(max_chars=220, overlap_chars=30))
    assert len(out.chunks) >= 3
    assert all(c.section_id == sec.section_id for c in out.chunks)
    assert out.chunks[0].chunk_index == 0
    assert out.chunks[1].chunk_index == 1


def test_chunks_never_cross_section_boundaries() -> None:
    doc_id = "doc_chunk_bounds"
    a = DocumentSection(
        document_id=doc_id,
        section_id=f"{doc_id}::sec::0000",
        section_type="article",
        title="A",
        start_page=1,
        end_page=1,
        start_char_offset=0,
        end_char_offset=40,
        parent_section_id=None,
        text="A" * 40,
    )
    b = DocumentSection(
        document_id=doc_id,
        section_id=f"{doc_id}::sec::0001",
        section_type="article",
        title="B",
        start_page=1,
        end_page=1,
        start_char_offset=40,
        end_char_offset=90,
        parent_section_id=None,
        text="B" * 50,
    )
    art = _artifact(document_id=doc_id, sections=[a, b])
    out = chunk_document_sections(art, config=ChunkingConfig(max_chars=25, overlap_chars=5))
    by_id = {s.section_id: s for s in art.sections}
    for c in out.chunks:
        s = by_id[c.section_id]
        assert c.char_start >= s.start_char_offset
        assert c.char_end <= s.end_char_offset
        assert c.text == s.text[c.source_section_char_start : c.source_section_char_end]
        assert c.char_start == s.start_char_offset + c.source_section_char_start
        assert c.char_end == s.start_char_offset + c.source_section_char_end


def test_chunk_metadata_preserves_section_id() -> None:
    doc_id = "doc_chunk_meta_sid"
    sec = DocumentSection(
        document_id=doc_id,
        section_id=f"{doc_id}::sec::0042",
        section_type="part",
        title="제2관",
        start_page=3,
        end_page=4,
        start_char_offset=500,
        end_char_offset=560,
        parent_section_id=None,
        text="x" * 60,
    )
    art = _artifact(document_id=doc_id, sections=[sec])
    out = chunk_document_sections(art)
    assert all(ch.section_id == sec.section_id for ch in out.chunks)


def test_policy_variant_chunks_are_distinguishable() -> None:
    """적립형 / 거치형 / 즉시형 (and similar) remain separable via ``variant_name`` on chunks."""
    doc_id = "doc_chunk_variants"
    variants = ("적립형", "거치형", "즉시형")
    sections: list[DocumentSection] = []
    offset = 0
    for i, vn in enumerate(variants):
        body = f"{vn} 본문\n" + ("x\n" * 15)
        sec = DocumentSection(
            document_id=doc_id,
            section_id=f"{doc_id}::sec::{i:04d}",
            section_type="article",
            title=f"제{i + 1}조",
            start_page=1,
            end_page=1,
            start_char_offset=offset,
            end_char_offset=offset + len(body),
            parent_section_id=None,
            text=body,
            policy_unit_id=f"{doc_id}::pu::{i:04d}",
            policy_unit_name="테스트상품",
            variant_name=vn,
        )
        sections.append(sec)
        offset += len(body)
    art = _artifact(document_id=doc_id, sections=sections)
    out = chunk_document_sections(art, config=ChunkingConfig(max_chars=80, overlap_chars=10))
    by_variant: dict[str, list[str]] = {}
    for ch in out.chunks:
        assert ch.variant_name is not None
        by_variant.setdefault(ch.variant_name, []).append(ch.chunk_id)
    assert set(by_variant) == set(variants)
    for vn in variants:
        assert len(by_variant[vn]) >= 1


def test_chunk_metadata_preserves_policy_unit_and_variant() -> None:
    doc_id = "doc_chunk_pu"
    sec = DocumentSection(
        document_id=doc_id,
        section_id=f"{doc_id}::sec::0001",
        section_type="article",
        title="제2조",
        start_page=1,
        end_page=2,
        start_char_offset=0,
        end_char_offset=120,
        parent_section_id=f"{doc_id}::sec::0000",
        text="z" * 120,
        policy_unit_id=f"{doc_id}::pu::0001",
        policy_unit_name="테스트상품",
        variant_name="적립형",
    )
    art = _artifact(document_id=doc_id, sections=[sec])
    out = chunk_document_sections(art, config=ChunkingConfig(max_chars=40, overlap_chars=5))
    for ch in out.chunks:
        assert ch.policy_unit_id == sec.policy_unit_id
        assert ch.policy_unit_name == sec.policy_unit_name
        assert ch.variant_name == "적립형"


def test_appendix_chunks_preserve_section_type() -> None:
    doc_id = "doc_chunk_apx"
    sec = DocumentSection(
        document_id=doc_id,
        section_id=f"{doc_id}::sec::0100",
        section_type="appendix",
        title="( 별표 1 )",
        start_page=10,
        end_page=11,
        start_char_offset=200,
        end_char_offset=260,
        parent_section_id=None,
        text="별표\n" + ("t" * 200),
    )
    art = _artifact(document_id=doc_id, sections=[sec])
    out = chunk_document_sections(art, config=ChunkingConfig(max_chars=80, overlap_chars=10))
    assert all(c.section_type == "appendix" for c in out.chunks)


def test_back_matter_chunks_keep_variant_name_none() -> None:
    doc_id = "doc_chunk_back"
    sec = DocumentSection(
        document_id=doc_id,
        section_id=f"{doc_id}::sec::0999",
        section_type="article",
        title="제1조 (정의)",
        start_page=99,
        end_page=99,
        start_char_offset=10_000,
        end_char_offset=10_000 + 80,
        parent_section_id=None,
        text="q" * 80,
        policy_unit_id=None,
        policy_unit_name=None,
        variant_name=None,
    )
    art = _artifact(document_id=doc_id, sections=[sec])
    out = chunk_document_sections(art, config=ChunkingConfig(max_chars=30, overlap_chars=5))
    assert all(c.variant_name is None for c in out.chunks)
    assert all(c.policy_unit_id is None for c in out.chunks)


def test_chunk_ids_are_deterministic() -> None:
    doc_id = "doc_chunk_ids"
    secs = [
        DocumentSection(
            document_id=doc_id,
            section_id=f"{doc_id}::sec::0000",
            section_type="article",
            title="S0",
            start_page=1,
            end_page=1,
            start_char_offset=0,
            end_char_offset=120,
            parent_section_id=None,
            text="p" * 120,
        ),
        DocumentSection(
            document_id=doc_id,
            section_id=f"{doc_id}::sec::0001",
            section_type="article",
            title="S1",
            start_page=1,
            end_page=1,
            start_char_offset=120,
            end_char_offset=240,
            parent_section_id=None,
            text="q" * 120,
        ),
    ]
    art = _artifact(document_id=doc_id, sections=secs)
    cfg = ChunkingConfig(max_chars=50, overlap_chars=10)
    a1 = chunk_document_sections(art, config=cfg)
    a2 = chunk_document_sections(art, config=cfg)
    assert [c.chunk_id for c in a1.chunks] == [c.chunk_id for c in a2.chunks]
    assert a1.chunks[0].chunk_id == f"{doc_id}::chunk::0000::000"
    assert a1.chunks[1].chunk_id == f"{doc_id}::chunk::0000::001"


def test_overlap_stays_within_same_section() -> None:
    doc_id = "doc_chunk_overlap"
    text = ("줄\n" * 500).strip() + "\n"
    sec = DocumentSection(
        document_id=doc_id,
        section_id=f"{doc_id}::sec::0000",
        section_type="article",
        title="Overlap",
        start_page=1,
        end_page=1,
        start_char_offset=500,
        end_char_offset=500 + len(text),
        parent_section_id=None,
        text=text,
    )
    art = _artifact(document_id=doc_id, sections=[sec])
    out = chunk_document_sections(art, config=ChunkingConfig(max_chars=120, overlap_chars=40))
    sec_chunks = [c for c in out.chunks if c.section_id == sec.section_id]
    assert len(sec_chunks) >= 2
    c0, c1 = sec_chunks[0], sec_chunks[1]
    overlap = c0.char_end - c1.char_start
    assert overlap > 0
    assert c1.char_start >= c0.char_start


def test_cli_writes_chunk_artifact(tmp_path: Path) -> None:
    doc_id = "doc_chunk_cli"
    sec = DocumentSection(
        document_id=doc_id,
        section_id=f"{doc_id}::sec::0000",
        section_type="article",
        title="CLI",
        start_page=1,
        end_page=1,
        start_char_offset=0,
        end_char_offset=20,
        parent_section_id=None,
        text="cli-test-body-here!!",
    )
    art = _artifact(document_id=doc_id, sections=[sec])
    in_dir = tmp_path / "sections"
    out_dir = tmp_path / "chunks"
    in_dir.mkdir()
    (in_dir / f"{doc_id}.sections.json").write_text(art.model_dump_json(indent=2), encoding="utf-8")

    rc = main(["--input-dir", str(in_dir), "--output-dir", str(out_dir)])
    assert rc == 0
    out_path = chunks_json_path(output_dir=out_dir, document_id=doc_id)
    assert out_path.is_file()
    loaded = DocumentChunksArtifact.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert loaded.document_id == doc_id
    assert len(loaded.chunks) == 1
    assert loaded.chunks[0].text == sec.text
