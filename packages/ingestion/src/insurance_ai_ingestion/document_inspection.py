from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from insurance_ai_shared.models.document import Document

LOW_TEXT_CHAR_THRESHOLD = 50
PREVIEW_MAX_CHARS = 200


@dataclass(frozen=True)
class SuspiciousPage:
    """Page flagged as empty or below the low-text threshold."""

    page_number: int
    reason: str


@dataclass(frozen=True)
class DocumentQualitySummary:
    """Per-document ingestion quality metrics derived from processed JSON."""

    document_id: str
    insurer: str
    product_name: str
    product_type: str
    page_count: int
    total_char_count: int
    average_chars_per_page: float
    min_chars_per_page: int
    max_chars_per_page: int
    empty_page_count: int
    low_text_page_count: int
    first_nonempty_page_number: int | None
    first_nonempty_page_preview: str | None
    suspicious_pages: tuple[SuspiciousPage, ...]


@dataclass(frozen=True)
class AggregateQualitySummary:
    """Roll-up metrics across multiple processed documents."""

    total_documents: int
    total_pages: int
    total_chars: int
    documents_with_empty_pages: int
    documents_with_low_text_pages: int


def _truncate_preview(text: str, *, max_chars: int = PREVIEW_MAX_CHARS) -> str:
    cleaned = text.replace("\r\n", "\n").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def inspect_document(document: Document) -> DocumentQualitySummary:
    """Compute quality metrics for one processed Document."""
    pages = document.pages
    page_count = len(pages)
    counts = [p.char_count for p in pages]
    total_chars = document.total_char_count

    if page_count == 0:
        return DocumentQualitySummary(
            document_id=document.document_id,
            insurer=document.metadata.insurer,
            product_name=document.metadata.product_name,
            product_type=document.metadata.product_type,
            page_count=0,
            total_char_count=total_chars,
            average_chars_per_page=0.0,
            min_chars_per_page=0,
            max_chars_per_page=0,
            empty_page_count=0,
            low_text_page_count=0,
            first_nonempty_page_number=None,
            first_nonempty_page_preview=None,
            suspicious_pages=(),
        )

    empty_page_count = sum(1 for c in counts if c == 0)
    low_text_page_count = sum(1 for c in counts if c < LOW_TEXT_CHAR_THRESHOLD)
    avg = total_chars / page_count if page_count else 0.0
    suspicious: list[SuspiciousPage] = []
    for page in pages:
        if page.char_count == 0:
            suspicious.append(SuspiciousPage(page_number=page.page_number, reason="empty"))
        elif page.char_count < LOW_TEXT_CHAR_THRESHOLD:
            suspicious.append(SuspiciousPage(page_number=page.page_number, reason="low_text"))

    first_nonempty_page_number: int | None = None
    first_nonempty_page_preview: str | None = None
    for page in pages:
        if page.char_count > 0:
            first_nonempty_page_number = page.page_number
            first_nonempty_page_preview = _truncate_preview(page.text)
            break

    return DocumentQualitySummary(
        document_id=document.document_id,
        insurer=document.metadata.insurer,
        product_name=document.metadata.product_name,
        product_type=document.metadata.product_type,
        page_count=page_count,
        total_char_count=total_chars,
        average_chars_per_page=avg,
        min_chars_per_page=min(counts),
        max_chars_per_page=max(counts),
        empty_page_count=empty_page_count,
        low_text_page_count=low_text_page_count,
        first_nonempty_page_number=first_nonempty_page_number,
        first_nonempty_page_preview=first_nonempty_page_preview,
        suspicious_pages=tuple(suspicious),
    )


def aggregate_summaries(summaries: list[DocumentQualitySummary]) -> AggregateQualitySummary:
    """Aggregate per-document summaries."""
    total_documents = len(summaries)
    total_pages = sum(s.page_count for s in summaries)
    total_chars = sum(s.total_char_count for s in summaries)
    documents_with_empty_pages = sum(1 for s in summaries if s.empty_page_count > 0)
    documents_with_low_text_pages = sum(1 for s in summaries if s.low_text_page_count > 0)
    return AggregateQualitySummary(
        total_documents=total_documents,
        total_pages=total_pages,
        total_chars=total_chars,
        documents_with_empty_pages=documents_with_empty_pages,
        documents_with_low_text_pages=documents_with_low_text_pages,
    )


def load_documents_from_dir(input_dir: Path) -> list[Document]:
    """Load all ``*.json`` files as Document models (sorted by path for stable ordering)."""
    if not input_dir.is_dir():
        msg = f"input directory does not exist or is not a directory: {input_dir}"
        raise FileNotFoundError(msg)
    paths = sorted(input_dir.glob("*.json"))
    documents: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        documents.append(Document.model_validate_json(text))
    return documents


def format_console_report(
    summaries: list[DocumentQualitySummary],
    aggregate: AggregateQualitySummary,
) -> str:
    """Human-readable multi-line summary for stdout."""
    lines: list[str] = []
    lines.append("=== Ingestion quality (processed Document JSON) ===")
    lines.append("")
    lines.append(
        f"Aggregate: documents={aggregate.total_documents} pages={aggregate.total_pages} "
        f"chars={aggregate.total_chars} "
        f"docs_with_empty_pages={aggregate.documents_with_empty_pages} "
        f"docs_with_low_text_pages={aggregate.documents_with_low_text_pages}"
    )
    lines.append("")
    for s in summaries:
        lines.append(f"--- {s.document_id} ---")
        lines.append(f"  insurer: {s.insurer}")
        lines.append(f"  product_name: {s.product_name}")
        lines.append(f"  product_type: {s.product_type}")
        lines.append(f"  page_count: {s.page_count}")
        lines.append(f"  total_char_count: {s.total_char_count}")
        lines.append(f"  average_chars_per_page: {s.average_chars_per_page:.2f}")
        lines.append(f"  min_chars_per_page: {s.min_chars_per_page}")
        lines.append(f"  max_chars_per_page: {s.max_chars_per_page}")
        lines.append(f"  empty_page_count: {s.empty_page_count}")
        lines.append(
            f"  low_text_page_count (<{LOW_TEXT_CHAR_THRESHOLD} chars): {s.low_text_page_count}"
        )
        lines.append(f"  first_nonempty_page_number: {s.first_nonempty_page_number}")
        preview = s.first_nonempty_page_preview
        if preview is None:
            lines.append("  first_nonempty_page_preview: (none)")
        else:
            single_line = preview.replace("\n", " ")
            lines.append(f"  first_nonempty_page_preview: {single_line}")
        if s.suspicious_pages:
            flagged = ", ".join(f"p{p.page_number}({p.reason})" for p in s.suspicious_pages[:20])
            extra = ""
            if len(s.suspicious_pages) > 20:
                extra = f" … (+{len(s.suspicious_pages) - 20} more)"
            lines.append(f"  suspicious_pages: {flagged}{extra}")
        else:
            lines.append("  suspicious_pages: (none)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_markdown_report(
    summaries: list[DocumentQualitySummary],
    aggregate: AggregateQualitySummary,
) -> str:
    """Markdown report suitable for writing to disk."""
    lines: list[str] = []
    lines.append("# Ingestion quality report")
    lines.append("")
    lines.append(
        "Generated by `insurance_ai_ingestion.inspect_documents`. Regenerate after ingestion:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append(
        "uv run python -m insurance_ai_ingestion.inspect_documents \\\n"
        "  --input-dir data/processed/documents \\\n"
        "  --report-path data/processed/reports/ingestion_quality.md"
    )
    lines.append("```")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Total documents | {aggregate.total_documents} |")
    lines.append(f"| Total pages | {aggregate.total_pages} |")
    lines.append(f"| Total characters | {aggregate.total_chars} |")
    lines.append(f"| Documents with empty pages | {aggregate.documents_with_empty_pages} |")
    low_hdr = f"Documents with low-text pages (<{LOW_TEXT_CHAR_THRESHOLD} chars)"
    lines.append(f"| {low_hdr} | {aggregate.documents_with_low_text_pages} |")
    lines.append("")
    lines.append("## Per document")
    lines.append("")
    for s in summaries:
        lines.append(f"### `{s.document_id}`")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| insurer | {s.insurer} |")
        safe_product = s.product_name.replace("|", "\\|")
        lines.append(f"| product_name | {safe_product} |")
        lines.append(f"| product_type | {s.product_type} |")
        lines.append(f"| page_count | {s.page_count} |")
        lines.append(f"| total_char_count | {s.total_char_count} |")
        lines.append(f"| average_chars_per_page | {s.average_chars_per_page:.2f} |")
        lines.append(f"| min_chars_per_page | {s.min_chars_per_page} |")
        lines.append(f"| max_chars_per_page | {s.max_chars_per_page} |")
        lines.append(f"| empty_page_count | {s.empty_page_count} |")
        lines.append(
            f"| low_text_page_count (<{LOW_TEXT_CHAR_THRESHOLD} chars) | {s.low_text_page_count} |"
        )
        lines.append(f"| first_nonempty_page_number | {s.first_nonempty_page_number} |")
        preview_cell = (
            s.first_nonempty_page_preview.replace("\n", " ").replace("|", "\\|")
            if s.first_nonempty_page_preview
            else "(none)"
        )
        lines.append(f"| first_nonempty_page_preview | {preview_cell} |")
        if s.suspicious_pages:
            bad = ", ".join(f"p{p.page_number} ({p.reason})" for p in s.suspicious_pages[:50])
            if len(s.suspicious_pages) > 50:
                bad += f" … (+{len(s.suspicious_pages) - 50} more)"
            lines.append(f"| suspicious_pages | {bad} |")
        else:
            lines.append("| suspicious_pages | (none) |")
        lines.append("")
    return "\n".join(lines) + "\n"
