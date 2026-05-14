from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from insurance_ai_shared.models.chunk import DocumentChunk, DocumentChunksArtifact


def load_chunk_artifacts(*, input_dir: Path) -> list[DocumentChunksArtifact]:
    paths = sorted(input_dir.glob("*.chunks.json"))
    out: list[DocumentChunksArtifact] = []
    for path in paths:
        out.append(DocumentChunksArtifact.model_validate_json(path.read_text(encoding="utf-8")))
    return out


@dataclass(frozen=True)
class LargestChunkRow:
    chunk_id: str
    document_id: str
    section_title: str
    section_type: str
    variant_name: str | None
    char_count: int


@dataclass(frozen=True)
class ChunkQualityInspection:
    total_documents: int
    total_chunks: int
    chunks_per_document: dict[str, int]
    average_char_count: float
    max_char_count: int
    min_char_count: int
    chunks_over_max_chars: int
    chunks_by_section_type: dict[str, int]
    chunks_by_variant_name: dict[str, int]
    chunks_with_null_variant_name: int
    chunks_with_missing_section_id: int
    chunks_with_missing_policy_unit_when_variant: int
    chunks_with_chunk_index_gt_0: int
    top_20_largest: tuple[LargestChunkRow, ...]
    chunks_by_chunking_strategy: dict[str, int]
    missing_section_type: int
    missing_text: int
    duplicate_chunk_ids: int
    chunk_index_negative: int
    char_count_len_mismatches: int


def _flatten_chunks(artifacts: Iterable[DocumentChunksArtifact]) -> list[tuple[str, DocumentChunk]]:
    rows: list[tuple[str, DocumentChunk]] = []
    for art in artifacts:
        for ch in art.chunks:
            rows.append((art.document_id, ch))
    return rows


def inspect_chunk_artifacts(artifacts: list[DocumentChunksArtifact]) -> ChunkQualityInspection:
    """Aggregate metrics and validation counters across all chunk artifacts."""
    total_documents = len(artifacts)
    flat = _flatten_chunks(artifacts)
    total_chunks = len(flat)

    chunks_per_document: dict[str, int] = Counter()
    by_section: Counter[str] = Counter()
    by_variant: Counter[str] = Counter()
    by_strategy: Counter[str] = Counter()
    null_variant = 0
    missing_section_id = 0
    missing_policy_when_variant = 0
    chunk_index_gt_0 = 0
    missing_section_type = 0
    missing_text = 0
    chunk_index_negative = 0
    char_mismatch = 0
    over_max = 0

    max_cfg_by_doc: dict[str, int] = {
        art.document_id: art.chunking_config.max_chars for art in artifacts
    }

    for doc_id, ch in flat:
        chunks_per_document[doc_id] += 1
        by_section[ch.section_type] += 1
        vkey = "<null>" if ch.variant_name is None else ch.variant_name
        by_variant[vkey] += 1
        by_strategy[ch.chunking_strategy] += 1

        if ch.variant_name is None:
            null_variant += 1
        if not (ch.section_id or "").strip():
            missing_section_id += 1
        if ch.variant_name is not None and ch.policy_unit_id is None:
            missing_policy_when_variant += 1
        if ch.chunk_index > 0:
            chunk_index_gt_0 += 1
        if not ch.section_type.strip():
            missing_section_type += 1
        if ch.text == "":
            missing_text += 1
        if ch.chunk_index < 0:
            chunk_index_negative += 1
        if len(ch.text) != ch.char_count:
            char_mismatch += 1

        cfg_max = max_cfg_by_doc.get(doc_id, 2200)
        if ch.char_count > cfg_max:
            over_max += 1

    all_chunk_ids = [ch.chunk_id for _, ch in flat]
    duplicate_rows = len(all_chunk_ids) - len(set(all_chunk_ids))

    if total_chunks == 0:
        avg = 0.0
        mx = mn = 0
    else:
        char_counts = [c.char_count for _, c in flat]
        avg = sum(char_counts) / total_chunks
        mx = max(char_counts)
        mn = min(char_counts)

    sorted_by_size = sorted(flat, key=lambda t: t[1].char_count, reverse=True)
    top20: list[LargestChunkRow] = []
    for doc_id, ch in sorted_by_size[:20]:
        top20.append(
            LargestChunkRow(
                chunk_id=ch.chunk_id,
                document_id=doc_id,
                section_title=ch.section_title,
                section_type=ch.section_type,
                variant_name=ch.variant_name,
                char_count=ch.char_count,
            )
        )

    return ChunkQualityInspection(
        total_documents=total_documents,
        total_chunks=total_chunks,
        chunks_per_document=dict(sorted(chunks_per_document.items())),
        average_char_count=avg,
        max_char_count=mx,
        min_char_count=mn,
        chunks_over_max_chars=over_max,
        chunks_by_section_type=dict(sorted(by_section.items())),
        chunks_by_variant_name=dict(sorted(by_variant.items())),
        chunks_with_null_variant_name=null_variant,
        chunks_with_missing_section_id=missing_section_id,
        chunks_with_missing_policy_unit_when_variant=missing_policy_when_variant,
        chunks_with_chunk_index_gt_0=chunk_index_gt_0,
        top_20_largest=tuple(top20),
        chunks_by_chunking_strategy=dict(sorted(by_strategy.items())),
        missing_section_type=missing_section_type,
        missing_text=missing_text,
        duplicate_chunk_ids=duplicate_rows,
        chunk_index_negative=chunk_index_negative,
        char_count_len_mismatches=char_mismatch,
    )


def format_console_report(inspection: ChunkQualityInspection) -> str:
    lines = [
        "inspect_chunks summary",
        f"  total_documents: {inspection.total_documents}",
        f"  total_chunks: {inspection.total_chunks}",
        f"  average_char_count: {inspection.average_char_count:.1f}",
        f"  max_char_count: {inspection.max_char_count}",
        f"  min_char_count: {inspection.min_char_count}",
        f"  chunks_over_max_chars: {inspection.chunks_over_max_chars}",
        f"  chunks_with_null_variant_name: {inspection.chunks_with_null_variant_name}",
        f"  chunks_with_missing_section_id: {inspection.chunks_with_missing_section_id}",
        f"  chunks_with_missing_policy_unit_when_variant: "
        f"{inspection.chunks_with_missing_policy_unit_when_variant}",
        f"  chunks_with_chunk_index_gt_0: {inspection.chunks_with_chunk_index_gt_0}",
        f"  duplicate_chunk_id_rows: {inspection.duplicate_chunk_ids}",
        f"  chunk_index_negative: {inspection.chunk_index_negative}",
        f"  char_count_len_mismatches: {inspection.char_count_len_mismatches}",
        f"  missing_section_type: {inspection.missing_section_type}",
        f"  missing_text: {inspection.missing_text}",
        "  chunks_per_document:",
    ]
    for doc_id, n in inspection.chunks_per_document.items():
        lines.append(f"    - {doc_id}: {n}")
    lines.append("  chunks_by_section_type:")
    for k, v in inspection.chunks_by_section_type.items():
        lines.append(f"    - {k}: {v}")
    lines.append("  chunks_by_variant_name:")
    for k, v in inspection.chunks_by_variant_name.items():
        lines.append(f"    - {k}: {v}")
    lines.append("  chunks_by_chunking_strategy:")
    for k, v in inspection.chunks_by_chunking_strategy.items():
        lines.append(f"    - {k}: {v}")
    lines.append("  top_20_largest (chunk_id | doc | section_title | type | variant | chars):")
    for row in inspection.top_20_largest:
        vn = "<null>" if row.variant_name is None else row.variant_name
        lines.append(
            f"    - {row.chunk_id} | {row.document_id} | {row.section_title[:60]} | "
            f"{row.section_type} | {vn} | {row.char_count}"
        )
    return "\n".join(lines) + "\n"


def write_markdown_report(*, inspection: ChunkQualityInspection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Chunk quality report",
        "",
        "## Metrics",
        "",
        f"- **total_documents**: {inspection.total_documents}",
        f"- **total_chunks**: {inspection.total_chunks}",
        f"- **average_char_count**: {inspection.average_char_count:.1f}",
        f"- **max_char_count**: {inspection.max_char_count}",
        f"- **min_char_count**: {inspection.min_char_count}",
        f"- **chunks_over_max_chars**: {inspection.chunks_over_max_chars}",
        f"- **chunks_with_null_variant_name**: {inspection.chunks_with_null_variant_name}",
        f"- **chunks_with_missing_section_id**: {inspection.chunks_with_missing_section_id}",
        f"- **chunks_with_missing_policy_unit_when_variant**: "
        f"{inspection.chunks_with_missing_policy_unit_when_variant}",
        f"- **chunks_with_chunk_index_gt_0**: {inspection.chunks_with_chunk_index_gt_0}",
        f"- **duplicate_chunk_id_rows**: {inspection.duplicate_chunk_ids}",
        f"- **chunk_index_negative**: {inspection.chunk_index_negative}",
        f"- **char_count_len_mismatches**: {inspection.char_count_len_mismatches}",
        f"- **missing_section_type**: {inspection.missing_section_type}",
        f"- **missing_text**: {inspection.missing_text}",
        "",
        "### chunks_per_document",
        "",
        "| document_id | chunks |",
        "|---|---:|",
    ]
    for doc_id, n in inspection.chunks_per_document.items():
        lines.append(f"| {doc_id} | {n} |")
    lines.extend(
        [
            "",
            "### chunks_by_section_type",
            "",
            "| section_type | chunks |",
            "|---|---:|",
        ]
    )
    for k, v in inspection.chunks_by_section_type.items():
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "### chunks_by_variant_name",
            "",
            "| variant_name | chunks |",
            "|---|---:|",
        ]
    )
    for k, v in inspection.chunks_by_variant_name.items():
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "### chunks_by_chunking_strategy",
            "",
            "| chunking_strategy | chunks |",
            "|---|---:|",
        ]
    )
    for k, v in inspection.chunks_by_chunking_strategy.items():
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "## Top 20 largest chunks",
            "",
            "| chunk_id | document_id | section_title | section_type | variant_name | char_count |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for row in inspection.top_20_largest:
        title = row.section_title.replace("|", "\\|")
        vn = "" if row.variant_name is None else row.variant_name.replace("|", "\\|")
        lines.append(
            f"| {row.chunk_id} | {row.document_id} | {title} | {row.section_type} | {vn} | "
            f"{row.char_count} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configure_stdout_utf8() -> None:
    enc = getattr(sys.stdout, "encoding", None) or ""
    if enc.casefold() == "utf-8".casefold():
        return
    reconf = getattr(sys.stdout, "reconfigure", None)
    if callable(reconf):
        try:
            reconf(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect ``*.chunks.json`` for retrieval-oriented quality metrics.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing ``*.chunks.json``.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path for a UTF-8 markdown report (parent dirs created).",
    )
    args = parser.parse_args(argv)

    _configure_stdout_utf8()

    try:
        artifacts = load_chunk_artifacts(input_dir=args.input_dir)
    except (FileNotFoundError, OSError, UnicodeError, ValidationError, ValueError) as exc:
        print(f"inspect_chunks: error: {exc}", file=sys.stderr)
        return 1

    inspection = inspect_chunk_artifacts(artifacts)
    print(format_console_report(inspection), end="")

    if args.report_path is not None:
        write_markdown_report(inspection=inspection, path=args.report_path)
        print(f"wrote report: {args.report_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
