from __future__ import annotations

from datetime import datetime

from insurance_ai_shared.models.document import Document, DocumentMetadata, DocumentPage


def build_document(
    *,
    metadata: DocumentMetadata,
    pages: list[DocumentPage],
    created_at: datetime | None = None,
) -> Document:
    """Assemble a canonical Document with aggregate stats."""
    stamp = created_at if created_at is not None else Document.utc_now()
    total_char_count = sum(p.char_count for p in pages)
    return Document(
        document_id=metadata.document_id,
        metadata=metadata,
        pages=pages,
        page_count=len(pages),
        total_char_count=total_char_count,
        created_at=stamp,
    )
