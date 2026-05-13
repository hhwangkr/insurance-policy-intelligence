"""Stable assembly_evidence / filter trace tokens for section_detection."""

from __future__ import annotations


def candidate_rule_confidence(confidence: float) -> str:
    return f"candidate_rule_confidence:{confidence:.4f}"


def page_region(region_type: str) -> str:
    return f"page_region:{region_type}"


def page_region_confidence(confidence: float) -> str:
    return f"page_region_confidence:{confidence:.4f}"


def region_evidence(token: str) -> str:
    return f"region_evidence:{token}"


# --- Filters (suppression / gating) ---
FILTER_MISSING_REGION_DEFAULT_KEEP = "filter:missing_region_default_keep"
FILTER_NOT_FRONT_MATTER_REGION_KEEP = "filter:not_front_matter_region_keep"
FILTER_SUPPRESS_STRUCTURE_ON_FRONT_MATTER_PREFIX = "filter:suppress_structure_on_front_matter"
FILTER_KEEP_EXPLICIT_TOC_GUIDE_HEADING = "filter:keep_explicit_toc_guide_heading"
FILTER_SUPPRESS_NON_STRUCTURAL_ON_FRONT_MATTER = "filter:suppress_non_structural_on_front_matter"
FILTER_TOCLIKE_CANDIDATE_BACKSTOP = "filter:toc_like_candidate_backstop"
FILTER_POINTER_REFERENCE_HEADING_SLICE = "filter:pointer_reference_heading_slice"
FILTER_APPENDIX_INSUFFICIENT_SUBSTANTIVE_BODY = "filter:appendix_insufficient_substantive_body"
FILTER_ARTICLE_INLINE_CLAUSE_REFERENCE_HEADING = "filter:article_inline_clause_reference_heading"

# --- Assembly ---
ASSEMBLY_SUBSTRING_SLICE = "assembly:substring_slice"


def candidate_id(candidate_id: str) -> str:
    return f"candidate_id:{candidate_id}"


def suppress_structure_on_front_matter(region_type: str) -> str:
    return f"{FILTER_SUPPRESS_STRUCTURE_ON_FRONT_MATTER_PREFIX}:{region_type}"
