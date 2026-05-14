from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from insurance_ai_ingestion.section_chunking import (
    chunk_document_sections,
    format_chunking_summary,
    summarize_chunking_run,
)
from insurance_ai_shared.models.chunk import ChunkingConfig, DocumentChunksArtifact
from insurance_ai_shared.models.section import DocumentSectionsArtifact
from insurance_ai_shared.stdio_utf8 import configure_stdout_utf8

__all__ = [
    "chunks_json_path",
    "configure_stdout_utf8",
    "load_sections_artifacts",
    "main",
]


def chunks_json_path(*, output_dir: Path, document_id: str) -> Path:
    safe_id = document_id.strip()
    if not safe_id:
        msg = "document_id must be non-empty"
        raise ValueError(msg)
    return output_dir / f"{safe_id}.chunks.json"


def load_sections_artifacts(*, input_dir: Path) -> list[DocumentSectionsArtifact]:
    paths = sorted(input_dir.glob("*.sections.json"))
    artifacts: list[DocumentSectionsArtifact] = []
    for path in paths:
        artifacts.append(
            DocumentSectionsArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        )
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic section-aware chunking: read ``*.sections.json`` and write "
            "``*.chunks.json`` (one chunk stream per section; no cross-section merges)."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing ``*.sections.json`` (e.g. data/processed/sections).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for ``*.chunks.json`` (e.g. data/processed/chunks).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Maximum characters per chunk within a section (default: 2200).",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=None,
        help="Overlap between consecutive chunks within the same section (default: 250).",
    )
    args = parser.parse_args(argv)
    configure_stdout_utf8()

    try:
        sections_artifacts = load_sections_artifacts(input_dir=args.input_dir)
    except (FileNotFoundError, OSError, UnicodeError, ValidationError, ValueError) as exc:
        print(f"chunk_sections: error: {exc}", file=sys.stderr)
        return 1

    cfg = ChunkingConfig()
    if args.max_chars is not None:
        cfg = cfg.model_copy(update={"max_chars": args.max_chars})
    if args.overlap_chars is not None:
        cfg = cfg.model_copy(update={"overlap_chars": args.overlap_chars})

    out_artifacts: list[DocumentChunksArtifact] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for artifact in sections_artifacts:
        chunked = chunk_document_sections(artifact, config=cfg)
        out_artifacts.append(chunked)
        path = chunks_json_path(output_dir=args.output_dir, document_id=artifact.document_id)
        path.write_text(chunked.model_dump_json(indent=2), encoding="utf-8")
        print(path)

    summary = summarize_chunking_run(out_artifacts)
    print(format_chunking_summary(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
