from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from insurance_ai_retrieval.embedder import PassageEmbedder
from insurance_ai_retrieval.index_engine import (
    SearchFilters,
    SearchHit,
    filters_match_hit,
    search_local_index,
)
from insurance_ai_retrieval.metadata import ChunkMetadataRecord


class QueryFilters(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    insurer: str
    product_type: str
    variant_name: str | None = None


class RetrievalQuerySpec(BaseModel):
    """One evaluation case loaded from YAML."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str
    query: str
    filters: QueryFilters
    expected_any_section_titles: list[str] = Field(default_factory=list)
    expected_any_section_types: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    dedupe_section: bool = True

    @model_validator(mode="after")
    def _require_at_least_one_expectation(self) -> RetrievalQuerySpec:
        types = self.expected_any_section_types or []
        if not self.expected_any_section_titles and not types:
            msg = (
                f"query {self.id!r}: set expected_any_section_titles and/or "
                "expected_any_section_types (at least one non-empty)"
            )
            raise ValueError(msg)
        return self

    def to_search_filters(self) -> SearchFilters:
        return SearchFilters(
            insurer=self.filters.insurer,
            product_type=self.filters.product_type,
            variant_name=self.filters.variant_name,
        )


class RetrievalEvalFile(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    queries: list[RetrievalQuerySpec]


def load_retrieval_queries(path: Path) -> list[RetrievalQuerySpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"expected mapping at root in {path}, got {type(raw).__name__}"
        raise ValueError(msg)
    try:
        return RetrievalEvalFile.model_validate(raw).queries
    except ValidationError as exc:
        msg = f"invalid retrieval queries YAML ({path}): {exc}"
        raise ValueError(msg) from exc


def _normalize_title(s: str) -> str:
    return " ".join(s.strip().split())


def _compact_title(s: str) -> str:
    return "".join(s.split())


def title_matches_any(actual: str, expected_titles: list[str]) -> bool:
    """Match expected titles after whitespace normalization, with a compact fallback."""
    if not expected_titles:
        return False
    a_norm = _normalize_title(actual)
    a_compact = _compact_title(actual)
    for exp in expected_titles:
        if _normalize_title(exp) == a_norm:
            return True
        if _compact_title(exp) == a_compact:
            return True
    return False


def expectation_met(metadata: ChunkMetadataRecord, spec: RetrievalQuerySpec) -> bool:
    titles = spec.expected_any_section_titles
    types_ = spec.expected_any_section_types or []
    has_titles = bool(titles)
    has_types = bool(types_)
    if has_titles and has_types:
        return title_matches_any(metadata.section_title, titles) and metadata.section_type in types_
    if has_titles:
        return title_matches_any(metadata.section_title, titles)
    return metadata.section_type in types_


def hit_at_rank_prefix(hits: list[SearchHit], spec: RetrievalQuerySpec, k: int) -> bool:
    if k <= 0:
        return False
    depth = min(k, len(hits))
    if depth == 0:
        return False
    for h in hits[:depth]:
        if expectation_met(h.metadata, spec):
            return True
    return False


def all_hits_match_filters(hits: list[SearchHit], filters: SearchFilters) -> bool:
    for h in hits:
        if not filters_match_hit(h.metadata, filters):
            return False
    return True


@dataclass
class QueryEvalResult:
    query_id: str
    pass_at_1: bool
    pass_at_3: bool
    pass_at_5: bool
    all_filters_matched: bool
    top_section_titles: list[str]
    top_section_types: list[str]
    hit_detail: str


@dataclass
class EvalRunSummary:
    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    per_query: list[QueryEvalResult] = field(default_factory=list)


def evaluate_hits(
    hits: list[SearchHit], spec: RetrievalQuerySpec, *, filters: SearchFilters
) -> QueryEvalResult:
    all_ok = all_hits_match_filters(hits, filters)
    titles = [h.metadata.section_title for h in hits]
    types = [h.metadata.section_type for h in hits]
    p1 = hit_at_rank_prefix(hits, spec, 1)
    p3 = hit_at_rank_prefix(hits, spec, 3)
    p5 = hit_at_rank_prefix(hits, spec, 5)
    if p1:
        detail = "expected matched within rank 1"
    elif p3:
        detail = "expected matched within rank 3 (not rank 1)"
    elif p5:
        detail = "expected matched within rank 5 (not rank 3)"
    else:
        detail = (
            "no expected section title/type found in top results; "
            f"expected titles={spec.expected_any_section_titles!r} "
            f"types={spec.expected_any_section_types!r}"
        )
    return QueryEvalResult(
        query_id=spec.id,
        pass_at_1=p1,
        pass_at_3=p3,
        pass_at_5=p5,
        all_filters_matched=all_ok,
        top_section_titles=titles,
        top_section_types=types,
        hit_detail=detail,
    )


def run_queries_on_index(
    *,
    index_dir: Path,
    embedder: PassageEmbedder,
    queries: list[RetrievalQuerySpec],
) -> EvalRunSummary:
    results: list[QueryEvalResult] = []
    p1 = p3 = p5 = 0
    for spec in queries:
        filters = spec.to_search_filters()
        hits = search_local_index(
            index_dir=index_dir,
            query=spec.query,
            embedder=embedder,
            top_k=spec.top_k,
            filters=filters,
            dedupe_section=spec.dedupe_section,
        )
        r = evaluate_hits(hits, spec, filters=filters)
        results.append(r)
        p1 += int(r.pass_at_1)
        p3 += int(r.pass_at_3)
        p5 += int(r.pass_at_5)
    n = len(queries)
    if n == 0:
        return EvalRunSummary(
            total_queries=0, hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, per_query=[]
        )
    return EvalRunSummary(
        total_queries=n,
        hit_at_1=p1 / n,
        hit_at_3=p3 / n,
        hit_at_5=p5 / n,
        per_query=results,
    )


def render_markdown_report(summary: EvalRunSummary, *, index_dir: Path, queries_path: Path) -> str:
    lines: list[str] = [
        "# Retrieval evaluation",
        "",
        f"- **index_dir:** `{index_dir.as_posix()}`",
        f"- **queries:** `{queries_path.as_posix()}`",
        "",
        "## Summary",
        "",
        f"- **total_queries:** {summary.total_queries}",
        f"- **hit@1:** {summary.hit_at_1:.3f}",
        f"- **hit@3:** {summary.hit_at_3:.3f}",
        f"- **hit@5:** {summary.hit_at_5:.3f}",
        "",
        "## Per query",
        "",
    ]
    for r in summary.per_query:
        overall = "PASS" if r.pass_at_5 and r.all_filters_matched else "FAIL"
        lines.extend(
            [
                f"### `{r.query_id}` — **{overall}**",
                "",
                f"- **hit@1:** {'yes' if r.pass_at_1 else 'no'}",
                f"- **hit@3:** {'yes' if r.pass_at_3 else 'no'}",
                f"- **hit@5:** {'yes' if r.pass_at_5 else 'no'}",
                f"- **all results matched filters:** {'yes' if r.all_filters_matched else 'no'}",
                f"- **note:** {r.hit_detail}",
                "",
                "**Top section titles:**",
                "",
            ]
        )
        for i, t in enumerate(r.top_section_titles, start=1):
            st = r.top_section_types[i - 1] if i - 1 < len(r.top_section_types) else ""
            lines.append(f"{i}. `{t}` ({st})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
