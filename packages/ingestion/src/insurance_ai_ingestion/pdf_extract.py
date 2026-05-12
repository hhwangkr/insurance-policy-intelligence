from __future__ import annotations

from pathlib import Path

import fitz

from insurance_ai_shared.models.document import DocumentPage

_EXTRACTION_METHOD = "pymupdf"


def extract_document_pages(*, pdf_path: Path, document_id: str) -> list[DocumentPage]:
    """Extract plain text per page using PyMuPDF (1-based page_number)."""
    pages: list[DocumentPage] = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text()
            page_number = i + 1
            pages.append(
                DocumentPage(
                    document_id=document_id,
                    page_number=page_number,
                    text=text,
                    char_count=len(text),
                    extraction_method=_EXTRACTION_METHOD,
                )
            )
    finally:
        doc.close()
    return pages
