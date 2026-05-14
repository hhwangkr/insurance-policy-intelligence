from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from insurance_ai_retrieval.document_id import parse_document_id
from insurance_ai_shared.models.chunk import DocumentChunk

_INSURER_DISPLAY: dict[str, str] = {
    "kyobolife": "교보생명",
    "samsunglife": "삼성생명",
    "miraeassetlife": "미래에셋생명",
}

_PRODUCT_TYPE_DISPLAY: dict[str, str] = {
    "annuity": "연금보험",
    "variable_annuity": "변액연금보험",
    "cancer": "암보험",
    "whole_life": "종신보험",
}


def display_label_for_insurer(value: str) -> str:
    """User-facing insurer label; unknown slugs pass through unchanged."""
    return _INSURER_DISPLAY.get(value, value)


def display_label_for_product_type(value: str) -> str:
    """User-facing product-type label; unknown slugs pass through unchanged."""
    return _PRODUCT_TYPE_DISPLAY.get(value, value)


def display_label_for_product_name(value: str) -> str:
    """User-facing label for parsed ``product_name`` (no slug table yet)."""
    return value


def apply_display_labels(record: ChunkMetadataRecord) -> ChunkMetadataRecord:
    """Attach display fields derived from canonical insurer / product_type / product_name."""
    insurer = record.insurer
    pt = record.product_type
    pn = record.product_name
    return record.model_copy(
        update={
            "insurer_display_name": display_label_for_insurer(insurer) if insurer else None,
            "product_type_display_name": display_label_for_product_type(pt) if pt else None,
            "product_display_name": display_label_for_product_name(pn) if pn else None,
        },
    )


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
    insurer_display_name: str | None = None
    product_type_display_name: str | None = None
    product_display_name: str | None = None
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
        base = cls(
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
        return apply_display_labels(base)


def enrich_chunk_metadata(record: ChunkMetadataRecord) -> ChunkMetadataRecord:
    """Fill insurer/product fields from ``document_id`` when missing (older indexes)."""
    rec = record
    if not (
        record.insurer is not None
        and record.product_type is not None
        and record.product_name is not None
    ):
        parsed = parse_document_id(record.document_id)
        if parsed is not None:
            updates: dict[str, Any] = {}
            if record.insurer is None:
                updates["insurer"] = parsed.insurer
            if record.product_type is None:
                updates["product_type"] = parsed.product_type
            if record.product_name is None:
                updates["product_name"] = parsed.product_name
            if updates:
                rec = rec.model_copy(update=updates)
    return apply_display_labels(rec)
