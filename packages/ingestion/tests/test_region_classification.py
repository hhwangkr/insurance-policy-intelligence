from __future__ import annotations

from datetime import UTC, date, datetime

from insurance_ai_ingestion.region_classification import HeuristicRegionClassifier
from insurance_ai_shared.models.document import Document, DocumentMetadata, DocumentPage


def _meta(document_id: str) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        insurer="test_insurer",
        product_name="테스트상품",
        product_type="annuity",
        product_slug="test_product",
        document_type="policy_terms",
        effective_date=date(2026, 1, 1),
        source_file="data/raw/manual/x.pdf",
        original_filename="x.pdf",
        content_hash="a" * 64,
        collection_method="test",
        dataset_split="unassigned",
        language="ko",
    )


def test_inline_byeolpyo_reference_does_not_force_appendix_region() -> None:
    """Body text referencing '(별표1)' mid-sentence should not label the page as appendix."""
    doc_id = "doc_inline_byeolpyo"
    long_clause = "제3조 본문입니다. 계약자는 (별표1)을 참고하여 신청하여야 합니다. " * 45
    pages = [
        DocumentPage(
            document_id=doc_id,
            page_number=1,
            text=long_clause,
            char_count=len(long_clause),
            extraction_method="pymupdf",
        ),
    ]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(long_clause),
        created_at=datetime.now(UTC),
    )
    regions = HeuristicRegionClassifier().classify_document(doc)
    assert len(regions) == 1
    assert regions[0].region_type != "appendix"
