from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_PAGE_EXTRACTION_METHOD = "pymupdf"


class DocumentMetadata(BaseModel):
    """Citation-oriented manifest fields persisted with processed documents."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    document_id: str
    insurer: str
    product_name: str
    product_type: str
    product_slug: str
    document_type: str
    effective_date: date
    source_file: str
    original_filename: str
    source_url: str = ""
    content_hash: Annotated[str, Field(min_length=64, max_length=64)]
    collection_method: str
    dataset_split: str
    language: str
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


class DocumentPage(BaseModel):
    """One PDF page of plain text (1-based page_number)."""

    model_config = ConfigDict(str_strip_whitespace=False)

    document_id: str
    page_number: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)
    extraction_method: str = _DEFAULT_PAGE_EXTRACTION_METHOD


class Document(BaseModel):
    """Canonical processed document: metadata + page texts for downstream chunking/RAG."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str
    metadata: DocumentMetadata
    pages: list[DocumentPage]
    page_count: int = Field(ge=0)
    total_char_count: int = Field(ge=0)
    created_at: datetime

    @staticmethod
    def utc_now() -> datetime:
        return datetime.now(UTC)
