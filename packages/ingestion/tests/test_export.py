from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from insurance_ai_ingestion.export import document_json_path, export_document_json
from insurance_ai_shared.models.document import Document, DocumentMetadata, DocumentPage


def test_document_json_path_joins_output_dir() -> None:
    out = Path("data/processed/documents")
    path = document_json_path(output_dir=out, document_id="my_doc_id")
    assert path == out / "my_doc_id.json"


def test_document_json_path_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        document_json_path(output_dir=Path("."), document_id="   ")


def test_export_document_json_writes_file(tmp_path: Path) -> None:
    meta = DocumentMetadata(
        document_id="doc_export",
        insurer="acme",
        product_name="P",
        product_type="auto",
        product_slug="p",
        document_type="policy",
        effective_date=date(2026, 1, 1),
        source_file="data/raw/manual/x.pdf",
        original_filename="x.pdf",
        content_hash="d" * 64,
        collection_method="manual",
        dataset_split="train",
        language="en",
    )
    pages = [
        DocumentPage(
            document_id="doc_export",
            page_number=1,
            text="x",
            char_count=1,
            extraction_method="pymupdf",
        )
    ]
    doc = Document(
        document_id="doc_export",
        metadata=meta,
        pages=pages,
        page_count=1,
        total_char_count=1,
        created_at=datetime(2026, 5, 12, tzinfo=UTC),
    )
    out_dir = tmp_path / "out"
    path = export_document_json(document=doc, output_dir=out_dir)
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("{")
