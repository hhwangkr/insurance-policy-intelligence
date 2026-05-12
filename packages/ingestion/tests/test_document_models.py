from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from insurance_ai_shared.models.document import Document, DocumentMetadata, DocumentPage


def test_document_metadata_round_trip_json_mode() -> None:
    meta = DocumentMetadata(
        document_id="doc_1",
        insurer="acme",
        product_name="P",
        product_type="auto",
        product_slug="p",
        document_type="policy",
        effective_date=date(2026, 1, 1),
        source_file="data/raw/manual/x.pdf",
        original_filename="x.pdf",
        content_hash="a" * 64,
        collection_method="manual",
        dataset_split="train",
        language="en",
        tags=["t1"],
        notes="n",
    )
    dumped = meta.model_dump(mode="json")
    assert dumped["effective_date"] == "2026-01-01"


def test_document_metadata_rejects_wrong_hash_length() -> None:
    with pytest.raises(ValidationError):
        DocumentMetadata(
            document_id="doc_1",
            insurer="acme",
            product_name="P",
            product_type="auto",
            product_slug="p",
            document_type="policy",
            effective_date=date(2026, 1, 1),
            source_file="data/raw/manual/x.pdf",
            original_filename="x.pdf",
            content_hash="abc",
            collection_method="manual",
            dataset_split="train",
            language="en",
        )


def test_document_page_requires_positive_page_number() -> None:
    with pytest.raises(ValidationError):
        DocumentPage(
            document_id="d",
            page_number=0,
            text="",
            char_count=0,
            extraction_method="pymupdf",
        )


def test_document_totals_match_pages() -> None:
    meta = DocumentMetadata(
        document_id="doc_1",
        insurer="acme",
        product_name="P",
        product_type="auto",
        product_slug="p",
        document_type="policy",
        effective_date=date(2026, 1, 1),
        source_file="data/raw/manual/x.pdf",
        original_filename="x.pdf",
        content_hash="c" * 64,
        collection_method="manual",
        dataset_split="train",
        language="en",
    )
    pages = [
        DocumentPage(
            document_id="doc_1",
            page_number=1,
            text="ab",
            char_count=2,
            extraction_method="pymupdf",
        ),
        DocumentPage(
            document_id="doc_1",
            page_number=2,
            text="cd",
            char_count=2,
            extraction_method="pymupdf",
        ),
    ]
    created = datetime(2026, 5, 12, tzinfo=UTC)
    doc = Document(
        document_id="doc_1",
        metadata=meta,
        pages=pages,
        page_count=len(pages),
        total_char_count=sum(p.char_count for p in pages),
        created_at=created,
    )
    assert doc.page_count == 2
    assert doc.total_char_count == 4
