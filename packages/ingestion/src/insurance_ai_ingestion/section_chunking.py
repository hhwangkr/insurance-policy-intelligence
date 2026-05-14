from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

from insurance_ai_shared.models.chunk import (
    SECTION_AWARE_CHUNK_STRATEGY,
    ChunkingConfig,
    DocumentChunk,
    DocumentChunksArtifact,
)
from insurance_ai_shared.models.section import DocumentSectionsArtifact


def _first_line_end_exclusive(text: str) -> int:
    nl = text.find("\n")
    return len(text) if nl < 0 else nl + 1


def _prefer_split_end(text: str, *, pos: int, hard_max: int) -> int:
    """Exclusive end in ``(pos, hard_max]``: blank paragraph, line break, else ``hard_max``."""
    n = len(text)
    if pos >= n or hard_max <= pos:
        return pos
    hard_max = min(hard_max, n)
    min_end = pos + 1
    if pos == 0:
        fle = _first_line_end_exclusive(text)
        if fle < n and fle <= hard_max:
            min_end = max(min_end, fle)

    lo = min_end
    hi = hard_max
    for e in range(hi, lo - 1, -1):
        if e >= 2 and text[e - 2 : e] == "\n\n":
            return e
    for e in range(hi, lo - 1, -1):
        if e >= 1 and text[e - 1] == "\n":
            return e
    return hard_max


def _snap_forward_line_start(text: str, i: int, limit: int) -> int:
    """If ``i`` is mid-line, snap forward to the first full line start ``< limit``."""
    if i <= 0 or i >= limit:
        return i
    line_start = text.rfind("\n", 0, i) + 1
    if line_start == i:
        return i
    nl = text.find("\n", i, limit)
    if nl == -1:
        return i
    return nl + 1


def _overlap_start(text: str, *, end: int, overlap: int, prev_pos: int) -> int:
    """Start index for the next chunk; overlap stays within ``text`` and advances toward ``end``."""
    n = len(text)
    if end >= n:
        return n
    max_overlap = end - prev_pos - 1
    if max_overlap <= 0:
        return min(prev_pos + 1, n - 1) if n > prev_pos + 1 else n
    ov = min(max(overlap, 0), max_overlap)
    nxt = max(prev_pos + 1, end - ov)
    snapped = _snap_forward_line_start(text, nxt, end)
    if snapped < end:
        nxt = snapped
    if nxt >= end:
        nxt = prev_pos + 1
    return min(max(nxt, prev_pos + 1), n - 1) if n > prev_pos + 1 else n


def _split_section_local_text(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[int, int]]:
    """Return half-open local spans ``[start, end)`` covering ``text`` in order, deterministic."""
    n = len(text)
    if n == 0:
        return []
    if n <= max_chars:
        return [(0, n)]

    spans: list[tuple[int, int]] = []
    pos = 0
    while pos < n:
        hard = min(pos + max_chars, n)
        if hard >= n:
            spans.append((pos, n))
            break
        end = _prefer_split_end(text, pos=pos, hard_max=hard)
        if end <= pos:
            end = min(pos + max_chars, n)
        if end <= pos:
            end = min(pos + 1, n)
        spans.append((pos, end))
        if end >= n:
            break
        prev = pos
        pos = _overlap_start(text, end=end, overlap=overlap_chars, prev_pos=prev)
        if pos <= prev:
            pos = min(prev + 1, n)
    return spans


def chunk_document_sections(
    artifact: DocumentSectionsArtifact,
    *,
    config: ChunkingConfig | None = None,
    generated_at: datetime | None = None,
) -> DocumentChunksArtifact:
    """Chunk each section alone.

    ``page_start`` / ``page_end`` copy the section span (not refined per chunk).
    """
    cfg = config or ChunkingConfig()
    if cfg.strategy != SECTION_AWARE_CHUNK_STRATEGY:
        msg = f"unsupported chunking strategy: {cfg.strategy!r}"
        raise ValueError(msg)
    stamp = generated_at if generated_at is not None else datetime.now(UTC)
    chunks: list[DocumentChunk] = []

    for section_index, section in enumerate(artifact.sections):
        spans = _split_section_local_text(
            section.text,
            max_chars=cfg.max_chars,
            overlap_chars=cfg.overlap_chars,
        )
        if not spans:
            continue

        base = section.start_char_offset
        for chunk_index, (ls, le) in enumerate(spans):
            slice_text = section.text[ls:le]
            abs_start = base + ls
            abs_end = base + le
            char_count = len(slice_text)
            chunk_id = f"{artifact.document_id}::chunk::{section_index:04d}::{chunk_index:03d}"
            chunks.append(
                DocumentChunk(
                    document_id=artifact.document_id,
                    chunk_id=chunk_id,
                    section_id=section.section_id,
                    section_type=section.section_type,
                    section_title=section.title,
                    parent_section_id=section.parent_section_id,
                    policy_unit_id=section.policy_unit_id,
                    policy_unit_name=section.policy_unit_name,
                    variant_name=section.variant_name,
                    chunk_index=chunk_index,
                    text=slice_text,
                    page_start=section.start_page,
                    page_end=section.end_page,
                    char_start=abs_start,
                    char_end=abs_end,
                    char_count=char_count,
                    token_estimate=max(0, (char_count + 3) // 4),
                    chunking_strategy=cfg.strategy,
                    source_section_char_start=ls,
                    source_section_char_end=le,
                )
            )

    return DocumentChunksArtifact(
        document_id=artifact.document_id,
        source_sections_created_at=artifact.created_at,
        generated_at=stamp,
        chunks=chunks,
        chunking_config=cfg,
    )


@dataclass(frozen=True)
class ChunkingRunSummary:
    """Lightweight aggregate stats for console reporting."""

    documents_processed: int
    total_chunks: int
    chunks_per_document: dict[str, int]
    average_chars_per_chunk: float
    max_chars_per_chunk: int
    chunks_by_section_type: dict[str, int]
    chunks_by_variant_name: dict[str, int]


def summarize_chunking_run(artifacts: list[DocumentChunksArtifact]) -> ChunkingRunSummary:
    total_chunks = 0
    chunks_per_document: dict[str, int] = {}
    chars_total = 0
    max_chars = 0
    by_type: Counter[str] = Counter()
    by_variant: Counter[str] = Counter()

    for art in artifacts:
        n = len(art.chunks)
        chunks_per_document[art.document_id] = n
        total_chunks += n
        for c in art.chunks:
            chars_total += c.char_count
            max_chars = max(max_chars, c.char_count)
            by_type[c.section_type] += 1
            key = "<null>" if c.variant_name is None else c.variant_name
            by_variant[key] += 1

    avg = (chars_total / total_chunks) if total_chunks else 0.0
    return ChunkingRunSummary(
        documents_processed=len(artifacts),
        total_chunks=total_chunks,
        chunks_per_document=chunks_per_document,
        average_chars_per_chunk=avg,
        max_chars_per_chunk=max_chars,
        chunks_by_section_type=dict(sorted(by_type.items())),
        chunks_by_variant_name=dict(sorted(by_variant.items())),
    )


def format_chunking_summary(summary: ChunkingRunSummary) -> str:
    lines = [
        "chunk_sections summary",
        f"  documents_processed: {summary.documents_processed}",
        f"  total_chunks: {summary.total_chunks}",
        f"  average_chars_per_chunk: {summary.average_chars_per_chunk:.1f}",
        f"  max_chars_per_chunk: {summary.max_chars_per_chunk}",
        "  chunks_per_document:",
    ]
    for doc_id, n in sorted(summary.chunks_per_document.items()):
        lines.append(f"    - {doc_id}: {n}")
    lines.append("  chunks_by_section_type:")
    for k, v in summary.chunks_by_section_type.items():
        lines.append(f"    - {k}: {v}")
    lines.append("  chunks_by_variant_name:")
    for k, v in summary.chunks_by_variant_name.items():
        lines.append(f"    - {k}: {v}")
    return "\n".join(lines) + "\n"


__all__ = [
    "ChunkingRunSummary",
    "chunk_document_sections",
    "format_chunking_summary",
    "summarize_chunking_run",
]
