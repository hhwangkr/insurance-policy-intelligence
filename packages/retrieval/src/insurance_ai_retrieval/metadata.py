from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from insurance_ai_shared.models.chunk import DocumentChunk


class ChunkMetadataRecord(BaseModel):
    """Citation-oriented metadata stored next to each embedding row (same order as ``.npy``)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    chunk_id: str
    section_id: str
    section_title: str
    section_type: str
    document_id: str
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
        return cls(
            chunk_id=ch.chunk_id,
            section_id=ch.section_id,
            section_title=ch.section_title,
            section_type=ch.section_type,
            document_id=ch.document_id,
            policy_unit_id=ch.policy_unit_id,
            policy_unit_name=ch.policy_unit_name,
            variant_name=ch.variant_name,
            page_start=ch.page_start,
            page_end=ch.page_end,
            char_start=ch.char_start,
            char_end=ch.char_end,
            text=ch.text,
        )
