from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from insurance_ai_retrieval.index_engine import SearchFilters


class RetrievalFiltersPayload(BaseModel):
    """JSON body mirror of ``SearchFilters`` (JSON-serializable lists for frozensets)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str | None = None
    insurer: str | None = None
    product_type: str | None = None
    product_name: str | None = None
    policy_unit_name: str | None = None
    variant_name: str | None = None
    include_section_types: list[str] | None = None
    exclude_section_types: list[str] = Field(default_factory=list)
    use_default_section_type_excludes: bool = True

    def to_search_filters(self) -> SearchFilters:
        include = frozenset(self.include_section_types) if self.include_section_types else None
        exclude_raw = self.exclude_section_types
        exclude = frozenset(exclude_raw) if exclude_raw else frozenset()
        use_default = self.use_default_section_type_excludes
        if include is not None:
            use_default = False
        return SearchFilters(
            document_id=self.document_id,
            insurer=self.insurer,
            product_type=self.product_type,
            product_name=self.product_name,
            policy_unit_name=self.policy_unit_name,
            variant_name=self.variant_name,
            include_section_types=include,
            exclude_section_types=exclude,
            use_default_section_type_excludes=use_default,
        )


class RetrievalContextRequest(BaseModel):
    """POST ``/retrieval/context`` body."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str
    index_dir: str
    filters: RetrievalFiltersPayload = Field(default_factory=RetrievalFiltersPayload)
    top_k: int = Field(5, ge=1, le=100)
    dedupe_section: bool = False

    @field_validator("query")
    @classmethod
    def query_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must be non-empty")
        return v

    @field_validator("index_dir")
    @classmethod
    def index_dir_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("index_dir must be non-empty")
        return v

    def resolved_index_dir(self) -> Path:
        return Path(self.index_dir).expanduser()
