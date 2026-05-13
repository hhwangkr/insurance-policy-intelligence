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

RegionType = Literal[
    "cover",
    "toc",
    "guide",
    "summary",
    "policy_body",
    "appendix",
    "legal_reference",
    "glossary",
    "unknown",
]


class PageRegion(BaseModel):
    """Primary layout/intent label for one PDF page (Phase 2D-2: heuristic; later: LLM optional)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str
    page_number: int = Field(ge=1)
    region_type: RegionType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class SectionCandidate(BaseModel):
    """Rule-generated heading candidate before region filtering and final assembly."""

    model_config = ConfigDict(str_strip_whitespace=True)

    candidate_id: str
    document_id: str
    section_type: SectionType
    title: str
    start_page: int = Field(ge=1)
    start_char_offset: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class PolicyUnit(BaseModel):
    """Deterministic product/variant span for multi-unit PDFs (marker line = unit start)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    policy_unit_id: str
    policy_unit_name: str
    variant_name: str
    start_page: int = Field(ge=1)
    start_char_offset: int = Field(ge=0)
    end_page: int = Field(ge=1)
    end_char_offset: int = Field(ge=0)


class DocumentSection(BaseModel):
    """Citation-ready section slice; ``text`` is always a substring of extracted source text."""

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
    assembly_confidence: float | None = Field(
        default=None,
        description="Confidence for inclusion after candidate generation and region checks.",
    )
    assembly_evidence: list[str] = Field(
        default_factory=list,
        description="Deterministic trace for validation (rules + region context).",
    )
    policy_unit_id: str | None = Field(
        default=None,
        description="Policy unit whose [start,end] span contains this section start, if any.",
    )
    policy_unit_name: str | None = Field(
        default=None,
        description="Product name parsed from the variant marker line, if assigned.",
    )
    variant_name: str | None = Field(
        default=None,
        description="Product variant from marker (e.g. 적립형, 거치형, 즉시형), if assigned.",
    )


class DocumentSectionsArtifact(BaseModel):
    """Sidecar file: ``{document_id}.sections.json`` next to processed documents."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str
    source_document_created_at: datetime
    sections: list[DocumentSection]
    page_regions: list[PageRegion] = Field(default_factory=list)
    section_candidates: list[SectionCandidate] = Field(default_factory=list)
    policy_units: list[PolicyUnit] = Field(
        default_factory=list,
        description="Product/variant spans from 적립형·거치형 + 약관 markers; not legal sections.",
    )
    created_at: datetime
