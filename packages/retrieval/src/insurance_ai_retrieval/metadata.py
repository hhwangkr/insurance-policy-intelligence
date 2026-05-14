from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from insurance_ai_retrieval.document_id import parse_document_id
from insurance_ai_shared.models.chunk import DocumentChunk


class ChunkMetadataRecord(BaseModel):
    """Citation-oriented metadata stored next to each embedding row (same order as ``.npy``)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    chunk_id: str
    section_id: str
    section_title: str
    section_type: str
    document_id: str
    insurer: str | None = None
    product_type: str | None = None
    product_name: str | None = None
    policy_unit_id: str | None
    policy_unit_name: str | None
    variant_name: str | None
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    text: str

    @classmethod
    def from_document_chunk(cls, ch: DocumentChunk) -> ChunkMetadataRecord:
        parsed = parse_document_id(ch.document_id)
        return cls(
            chunk_id=ch.chunk_id,
            section_id=ch.section_id,
            section_title=ch.section_title,
            section_type=ch.section_type,
            document_id=ch.document_id,
            insurer=parsed.insurer if parsed else None,
            product_type=parsed.product_type if parsed else None,
            product_name=parsed.product_name if parsed else None,
            policy_unit_id=ch.policy_unit_id,
            policy_unit_name=ch.policy_unit_name,
            variant_name=ch.variant_name,
            page_start=ch.page_start,
            page_end=ch.page_end,
            char_start=ch.char_start,
            char_end=ch.char_end,
            text=ch.text,
        )


def enrich_chunk_metadata(record: ChunkMetadataRecord) -> ChunkMetadataRecord:
    """Fill insurer/product fields from ``document_id`` when missing (older indexes)."""
    if (
        record.insurer is not None
        and record.product_type is not None
        and record.product_name is not None
    ):
        return record
    parsed = parse_document_id(record.document_id)
    if parsed is None:
        return record
    updates: dict[str, str] = {}
    if record.insurer is None:
        updates["insurer"] = parsed.insurer
    if record.product_type is None:
        updates["product_type"] = parsed.product_type
    if record.product_name is None:
        updates["product_name"] = parsed.product_name
    return record.model_copy(update=updates)
