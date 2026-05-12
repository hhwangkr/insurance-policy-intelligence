from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SectionType = Literal[
    "guide",
    "toc",
    "part",
    "article",
    "appendix",
    "legal_reference",
    "glossary",
    "unknown",
]


class DocumentSection(BaseModel):
    """Rule-detected policy section with offsets for citation grounding."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str
    section_id: str
    section_type: SectionType
    title: str
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    start_char_offset: int = Field(ge=0)
    end_char_offset: int = Field(ge=0)
    parent_section_id: str | None = None
    text: str = ""


class DocumentSectionsArtifact(BaseModel):
    """Sidecar file: ``{document_id}.sections.json`` next to processed documents."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str
    source_document_created_at: datetime
    sections: list[DocumentSection]
    created_at: datetime
