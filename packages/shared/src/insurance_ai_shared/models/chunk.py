from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from insurance_ai_shared.models.section import SectionType

DEFAULT_SECTION_CHUNK_MAX_CHARS = 2200
DEFAULT_SECTION_CHUNK_OVERLAP_CHARS = 250
SECTION_AWARE_CHUNK_STRATEGY = "section_aware_paragraph_v1"


class ChunkingConfig(BaseModel):
    """Deterministic knobs for section-aware chunking."""

    model_config = ConfigDict(str_strip_whitespace=True)

    max_chars: int = Field(default=DEFAULT_SECTION_CHUNK_MAX_CHARS, ge=1, le=200_000)
    overlap_chars: int = Field(default=DEFAULT_SECTION_CHUNK_OVERLAP_CHARS, ge=0, le=50_000)
    strategy: str = Field(default=SECTION_AWARE_CHUNK_STRATEGY)


class DocumentChunk(BaseModel):
    """Retrieval-oriented text slice; always contained in exactly one ``DocumentSection``."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str
    chunk_id: str
    section_id: str
    section_type: SectionType
    section_title: str
    parent_section_id: str | None = None
    policy_unit_id: str | None = None
    policy_unit_name: str | None = None
    variant_name: str | None = None
    chunk_index: int = Field(ge=0)
    text: str
    page_start: int = Field(
        ge=1,
        description="Inherited from the parent section (chunk-level page span not computed yet).",
    )
    page_end: int = Field(
        ge=1,
        description="Inherited from the parent section (chunk-level page span not computed yet).",
    )
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    char_count: int = Field(ge=0)
    token_estimate: int = Field(ge=0)
    chunking_strategy: str
    source_section_char_start: int = Field(ge=0)
    source_section_char_end: int = Field(ge=0)


class DocumentChunksArtifact(BaseModel):
    """Sidecar file: ``{document_id}.chunks.json`` derived from ``*.sections.json``."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str
    source_sections_created_at: datetime | None = Field(
        default=None,
        description="``created_at`` from the input ``DocumentSectionsArtifact``, when present.",
    )
    generated_at: datetime
    chunks: list[DocumentChunk]
    chunking_config: ChunkingConfig
