from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from insurance_ai_ingestion.inspect_chunks import (
    format_console_report,
    inspect_chunk_artifacts,
    load_chunk_artifacts,
    main,
    write_markdown_report,
)
from insurance_ai_shared.models.chunk import ChunkingConfig, DocumentChunk, DocumentChunksArtifact


def _dt() -> datetime:
    return datetime(2026, 5, 14, tzinfo=UTC)


def _base_chunk(**overrides: object) -> DocumentChunk:
    data: dict[str, object] = {
        "document_id": "doc_q",
        "chunk_id": "doc_q::chunk::0000::000",
        "section_id": "doc_q::sec::0000",
        "section_type": "article",
        "section_title": "제1조",
        "parent_section_id": None,
        "policy_unit_id": None,
        "policy_unit_name": None,
        "variant_name": None,
        "chunk_index": 0,
        "text": "abcd",
        "page_start": 1,
        "page_end": 1,
        "char_start": 0,
        "char_end": 4,
        "char_count": 4,
        "token_estimate": 1,
        "chunking_strategy": "section_aware_paragraph_v1",
        "source_section_char_start": 0,
        "source_section_char_end": 4,
    }
    data.update(overrides)
    return DocumentChunk.model_validate(data)


def _artifact(*, document_id: str, chunks: list[DocumentChunk]) -> DocumentChunksArtifact:
    return DocumentChunksArtifact(
        document_id=document_id,
        source_sections_created_at=_dt(),
        generated_at=_dt(),
        chunks=chunks,
        chunking_config=ChunkingConfig(max_chars=2200),
    )


def test_quality_summary_counts_chunks() -> None:
    art = _artifact(
        document_id="doc_q",
        chunks=[
            _base_chunk(chunk_id="doc_q::chunk::0000::000", chunk_index=0),
            _base_chunk(
                chunk_id="doc_q::chunk::0000::001",
                chunk_index=1,
                text="xy",
                char_count=2,
                char_end=2,
                source_section_char_end=2,
            ),
        ],
    )
    r = inspect_chunk_artifacts([art])
    assert r.total_documents == 1
    assert r.total_chunks == 2
    assert r.chunks_per_document["doc_q"] == 2
    assert r.average_char_count == 3.0


def test_detects_missing_section_id() -> None:
    art = _artifact(
        document_id="doc_q",
        chunks=[_base_chunk(section_id="   ", chunk_id="doc_q::chunk::0000::000")],
    )
    r = inspect_chunk_artifacts([art])
    assert r.chunks_with_missing_section_id == 1


def test_detects_duplicate_chunk_id() -> None:
    art = _artifact(
        document_id="doc_q",
        chunks=[
            _base_chunk(chunk_id="dup", chunk_index=0),
            _base_chunk(chunk_id="dup", chunk_index=1, text="z", char_count=1, char_end=1),
        ],
    )
    r = inspect_chunk_artifacts([art])
    assert r.duplicate_chunk_ids == 1


def test_detects_variant_without_policy_unit_id() -> None:
    art = _artifact(
        document_id="doc_q",
        chunks=[
            _base_chunk(
                chunk_id="doc_q::chunk::0000::000",
                variant_name="적립형",
                policy_unit_id=None,
            ),
        ],
    )
    r = inspect_chunk_artifacts([art])
    assert r.chunks_with_missing_policy_unit_when_variant == 1


def test_reports_chunks_by_section_type() -> None:
    art = _artifact(
        document_id="doc_q",
        chunks=[
            _base_chunk(
                chunk_id="doc_q::chunk::0000::000",
                section_type="article",
            ),
            _base_chunk(
                chunk_id="doc_q::chunk::0001::000",
                section_type="appendix",
                section_id="doc_q::sec::0001",
                chunk_index=0,
                text="x",
                char_count=1,
                char_end=1,
                source_section_char_end=1,
            ),
        ],
    )
    r = inspect_chunk_artifacts([art])
    assert r.chunks_by_section_type["article"] == 1
    assert r.chunks_by_section_type["appendix"] == 1


def test_reports_chunks_by_variant_name() -> None:
    art = _artifact(
        document_id="doc_q",
        chunks=[
            _base_chunk(chunk_id="a", variant_name=None),
            _base_chunk(
                chunk_id="b",
                chunk_index=1,
                variant_name="적립형",
                policy_unit_id="doc_q::pu::0",
                text="t",
                char_count=1,
                char_end=1,
                source_section_char_end=1,
            ),
        ],
    )
    r = inspect_chunk_artifacts([art])
    assert r.chunks_by_variant_name["<null>"] == 1
    assert r.chunks_by_variant_name["적립형"] == 1


def test_writes_markdown_report(tmp_path: Path) -> None:
    art = _artifact(
        document_id="doc_q",
        chunks=[_base_chunk()],
    )
    r = inspect_chunk_artifacts([art])
    out = tmp_path / "chunk_quality.md"
    write_markdown_report(inspection=r, path=out)
    text = out.read_text(encoding="utf-8")
    assert "### chunks_by_section_type" in text
    assert "### chunks_by_variant_name" in text
    assert "| article | 1 |" in text


def test_cli_loads_directory(tmp_path: Path) -> None:
    art = _artifact(document_id="doc_cli", chunks=[_base_chunk(document_id="doc_cli")])
    d = tmp_path / "chunks"
    d.mkdir()
    (d / "doc_cli.chunks.json").write_text(art.model_dump_json(indent=2), encoding="utf-8")
    report = tmp_path / "out.md"
    rc = main(["--input-dir", str(d), "--report-path", str(report)])
    assert rc == 0
    assert report.is_file()


def test_load_chunk_artifacts_roundtrip(tmp_path: Path) -> None:
    art = _artifact(document_id="doc_rt", chunks=[_base_chunk(document_id="doc_rt")])
    d = tmp_path / "c"
    d.mkdir()
    (d / "doc_rt.chunks.json").write_text(art.model_dump_json(), encoding="utf-8")
    loaded = load_chunk_artifacts(input_dir=d)
    assert len(loaded) == 1
    r = inspect_chunk_artifacts(loaded)
    assert r.total_chunks == 1


def test_format_console_report_includes_top_section() -> None:
    art = _artifact(document_id="doc_q", chunks=[_base_chunk()])
    r = inspect_chunk_artifacts([art])
    s = format_console_report(r)
    assert "inspect_chunks summary" in s
    assert "chunks_by_chunking_strategy" in s
