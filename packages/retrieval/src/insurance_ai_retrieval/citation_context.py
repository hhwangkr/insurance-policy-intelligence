from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from insurance_ai_retrieval.embedder import PassageEmbedder
from insurance_ai_retrieval.index_engine import SearchFilters, SearchHit, search_local_index
from insurance_ai_retrieval.metadata import enrich_chunk_metadata


def search_filters_to_mapping(filters: SearchFilters | None) -> dict[str, Any]:
    """JSON-friendly snapshot of gates used for this retrieval (for bundle reproducibility)."""
    f = filters or SearchFilters()
    include = sorted(f.include_section_types) if f.include_section_types is not None else None
    exclude = sorted(f.exclude_section_types) if f.exclude_section_types else []
    return {
        "document_id": f.document_id,
        "insurer": f.insurer,
        "product_type": f.product_type,
        "product_name": f.product_name,
        "policy_unit_name": f.policy_unit_name,
        "variant_name": f.variant_name,
        "include_section_types": include,
        "exclude_section_types": exclude,
        "use_default_section_type_excludes": f.use_default_section_type_excludes,
    }


class CitationContextEntry(BaseModel):
    """One ranked passage with a citation handle and retrieval score."""

    model_config = ConfigDict(str_strip_whitespace=True)

    citation_id: str
    chunk_id: str
    section_id: str
    section_title: str
    section_type: str
    document_id: str
    insurer: str | None = None
    product_type: str | None = None
    insurer_display_name: str | None = None
    product_type_display_name: str | None = None
    product_display_name: str | None = None
    policy_unit_name: str | None = None
    variant_name: str | None = None
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    score: float
    text: str


class CitationContextBundle(BaseModel):
    """Query-time retrieval result packaged for downstream RAG / auditing."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str
    filters: dict[str, Any]
    top_k: int
    dedupe_section: bool
    citations: list[CitationContextEntry]


def citation_bundle_from_hits(
    *,
    query: str,
    filters: SearchFilters | None,
    top_k: int,
    dedupe_section: bool,
    hits: list[SearchHit],
) -> CitationContextBundle:
    """Map ranked ``SearchHit`` rows to ``C1``… citation entries (no network / LLM)."""
    entries: list[CitationContextEntry] = []
    for i, hit in enumerate(hits):
        m = enrich_chunk_metadata(hit.metadata)
        entries.append(
            CitationContextEntry(
                citation_id=f"C{i + 1}",
                chunk_id=m.chunk_id,
                section_id=m.section_id,
                section_title=m.section_title,
                section_type=m.section_type,
                document_id=m.document_id,
                insurer=m.insurer,
                product_type=m.product_type,
                insurer_display_name=m.insurer_display_name,
                product_type_display_name=m.product_type_display_name,
                product_display_name=m.product_display_name,
                policy_unit_name=m.policy_unit_name,
                variant_name=m.variant_name,
                page_start=m.page_start,
                page_end=m.page_end,
                char_start=m.char_start,
                char_end=m.char_end,
                score=hit.score,
                text=m.text,
            ),
        )
    return CitationContextBundle(
        query=query,
        filters=search_filters_to_mapping(filters),
        top_k=top_k,
        dedupe_section=dedupe_section,
        citations=entries,
    )


def build_citation_context(
    *,
    index_dir: Path,
    query: str,
    embedder: PassageEmbedder,
    filters: SearchFilters | None = None,
    top_k: int = 5,
    dedupe_section: bool = False,
) -> CitationContextBundle:
    """Embed ``query``, search ``index_dir``, return a citation-ready bundle."""
    hits = search_local_index(
        index_dir=index_dir,
        query=query,
        embedder=embedder,
        top_k=top_k,
        filters=filters,
        dedupe_section=dedupe_section,
    )
    return citation_bundle_from_hits(
        query=query,
        filters=filters,
        top_k=top_k,
        dedupe_section=dedupe_section,
        hits=hits,
    )


def format_citation_bundle_json(bundle: CitationContextBundle) -> str:
    """Pretty JSON for stdout / files (UTF-8, readable Korean)."""
    payload = bundle.model_dump(mode="json")
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
