from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from insurance_ai_ingestion.document_inspection import (
    LOW_TEXT_CHAR_THRESHOLD,
    aggregate_summaries,
    format_markdown_report,
    inspect_document,
    load_documents_from_dir,
)
from insurance_ai_shared.models.document import Document, DocumentMetadata, DocumentPage


def _meta(*, document_id: str = "doc_a") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        insurer="acme",
        product_name="Sample Product",
        product_type="auto",
        product_slug="sample",
        document_type="policy",
        effective_date=date(2026, 1, 1),
        source_file="data/raw/manual/sample.pdf",
        original_filename="in.pdf",
        content_hash="a" * 64,
        collection_method="manual",
        dataset_split="train",
        language="en",
    )


def test_inspect_document_counts_empty_and_low_text() -> None:
    meta = _meta()
    pages = [
        DocumentPage(
            document_id=meta.document_id,
            page_number=1,
            text="",
            char_count=0,
            extraction_method="pymupdf",
        ),
        DocumentPage(
            document_id=meta.document_id,
            page_number=2,
            text="x" * 10,
            char_count=10,
            extraction_method="pymupdf",
        ),
        DocumentPage(
            document_id=meta.document_id,
            page_number=3,
            text="y" * 100,
            char_count=100,
            extraction_method="pymupdf",
        ),
    ]
    doc = Document(
        document_id=meta.document_id,
        metadata=meta,
        pages=pages,
        page_count=len(pages),
        total_char_count=sum(p.char_count for p in pages),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    summary = inspect_document(doc)
    assert summary.empty_page_count == 1
    assert summary.low_text_page_count == 2  # 0 and 10 are both < threshold
    assert summary.first_nonempty_page_number == 2
    assert summary.first_nonempty_page_preview == "x" * 10
    reasons = {p.page_number: p.reason for p in summary.suspicious_pages}
    assert reasons[1] == "empty"
    assert reasons[2] == "low_text"


def test_inspect_document_threshold_boundary() -> None:
    meta = _meta(document_id="doc_b")
    boundary = LOW_TEXT_CHAR_THRESHOLD - 1
    pages = [
        DocumentPage(
            document_id=meta.document_id,
            page_number=1,
            text="z" * boundary,
            char_count=boundary,
            extraction_method="pymupdf",
        ),
        DocumentPage(
            document_id=meta.document_id,
            page_number=2,
            text="z" * LOW_TEXT_CHAR_THRESHOLD,
            char_count=LOW_TEXT_CHAR_THRESHOLD,
            extraction_method="pymupdf",
        ),
    ]
    doc = Document(
        document_id=meta.document_id,
        metadata=meta,
        pages=pages,
        page_count=len(pages),
        total_char_count=sum(p.char_count for p in pages),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    summary = inspect_document(doc)
    assert summary.low_text_page_count == 1
    assert summary.empty_page_count == 0


def test_aggregate_summaries_flags_documents() -> None:
    meta = _meta(document_id="doc_empty")
    doc_empty = Document(
        document_id=meta.document_id,
        metadata=meta,
        pages=[
            DocumentPage(
                document_id=meta.document_id,
                page_number=1,
                text="",
                char_count=0,
                extraction_method="pymupdf",
            )
        ],
        page_count=1,
        total_char_count=0,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    meta_ok = _meta(document_id="doc_ok")
    doc_ok = Document(
        document_id=meta_ok.document_id,
        metadata=meta_ok,
        pages=[
            DocumentPage(
                document_id=meta_ok.document_id,
                page_number=1,
                text="ok",
                char_count=100,
                extraction_method="pymupdf",
            )
        ],
        page_count=1,
        total_char_count=100,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    s1 = inspect_document(doc_empty)
    s2 = inspect_document(doc_ok)
    agg = aggregate_summaries([s1, s2])
    assert agg.total_documents == 2
    assert agg.total_pages == 2
    assert agg.total_chars == 100
    assert agg.documents_with_empty_pages == 1
    assert agg.documents_with_low_text_pages == 1


def test_load_documents_from_dir_reads_json(tmp_path: Path) -> None:
    meta = _meta(document_id="doc_file")
    doc = Document(
        document_id=meta.document_id,
        metadata=meta,
        pages=[
            DocumentPage(
                document_id=meta.document_id,
                page_number=1,
                text="hello",
                char_count=5,
                extraction_method="pymupdf",
            )
        ],
        page_count=1,
        total_char_count=5,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    path = tmp_path / "doc_file.json"
    path.write_text(doc.model_dump_json(), encoding="utf-8")
    loaded = load_documents_from_dir(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].document_id == "doc_file"


def test_load_documents_from_dir_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents_from_dir(tmp_path / "nope")


def test_format_markdown_report_contains_aggregate_table() -> None:
    meta = _meta(document_id="doc_md")
    doc = Document(
        document_id=meta.document_id,
        metadata=meta,
        pages=[
            DocumentPage(
                document_id=meta.document_id,
                page_number=1,
                text="hi",
                char_count=2,
                extraction_method="pymupdf",
            )
        ],
        page_count=1,
        total_char_count=2,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    summary = inspect_document(doc)
    agg = aggregate_summaries([summary])
    md = format_markdown_report([summary], agg)
    assert "# Ingestion quality report" in md
    assert "| Total documents | 1 |" in md
    assert "inspect_documents" in md
