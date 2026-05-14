"""Local retrieval baselines (Phase 2F: numpy dot-product index over chunk embeddings)."""

from insurance_ai_retrieval.chunks_io import flatten_chunks_sorted, load_chunk_artifacts_from_dir
from insurance_ai_retrieval.citation_context import (
    CitationContextBundle,
    CitationContextEntry,
    build_citation_context,
    citation_bundle_from_hits,
    format_citation_bundle_json,
    search_filters_to_mapping,
)
from insurance_ai_retrieval.e5_text import format_e5_passage, format_e5_query
from insurance_ai_retrieval.embedder import LocalSentenceTransformerEmbedder, PassageEmbedder
from insurance_ai_retrieval.index_engine import (
    IndexConfig,
    SearchFilters,
    SearchHit,
    build_local_index,
    load_embeddings_matrix,
    load_index_config,
    load_metadata_rows,
    search_local_index,
)
from insurance_ai_retrieval.metadata import ChunkMetadataRecord

__all__ = [
    "CitationContextBundle",
    "CitationContextEntry",
    "ChunkMetadataRecord",
    "IndexConfig",
    "LocalSentenceTransformerEmbedder",
    "PassageEmbedder",
    "SearchFilters",
    "SearchHit",
    "build_citation_context",
    "build_local_index",
    "citation_bundle_from_hits",
    "flatten_chunks_sorted",
    "format_citation_bundle_json",
    "format_e5_passage",
    "format_e5_query",
    "load_chunk_artifacts_from_dir",
    "load_embeddings_matrix",
    "load_index_config",
    "load_metadata_rows",
    "search_filters_to_mapping",
    "search_local_index",
]
