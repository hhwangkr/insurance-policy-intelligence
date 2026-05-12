from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low", "unknown"]


class FieldInference(BaseModel):
    """Rule-based inference for a single semantic field."""

    value: str = ""
    confidence: Confidence = "unknown"
    needs_review: bool = True
    evidence: str = Field(default="", description="Short machine-readable provenance hint.")


class StagingManifestEntry(BaseModel):
    """One document row for data/manifests/manual.yaml after inbox staging."""

    document_id: str
    insurer: str
    product_name: str
    product_type: str
    product_slug: str
    document_type: str
    effective_date: str
    source_file: str
    original_filename: str
    source_url: str = ""
    content_hash: str
    collection_method: str = "manual_inbox_rules"
    dataset_split: str = "unassigned"
    language: str = ""
    collected_at: str
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    inference: dict[str, FieldInference] = Field(default_factory=dict)

    def to_yaml_dict(self) -> dict[str, object]:
        """Serialize for PyYAML (nested dicts, no Path objects)."""
        return self.model_dump(mode="python")
