from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from insurance_ai_ingestion.document_builder import build_document
from insurance_ai_ingestion.export import export_document_json
from insurance_ai_ingestion.hashing import sha256_hex_file
from insurance_ai_ingestion.manifest import load_manifest_documents, resolve_source_pdf
from insurance_ai_ingestion.pdf_extract import extract_document_pages
from insurance_ai_shared.models.document import DocumentMetadata


def _normalize_content_hash(value: str) -> str:
    cleaned = value.strip().lower()
    if len(cleaned) != 64:
        msg = "content_hash must be a 64-character SHA-256 hex digest"
        raise ValueError(msg)
    return cleaned


def _validate_pdf_exists_and_hash(metadata: DocumentMetadata, *, repo_root: Path) -> Path:
    pdf_path = resolve_source_pdf(repo_root=repo_root, source_file=metadata.source_file)
    if not pdf_path.is_file():
        msg = f"source_file not found: {metadata.source_file} (resolved: {pdf_path})"
        raise FileNotFoundError(msg)
    expected = _normalize_content_hash(metadata.content_hash)
    actual = sha256_hex_file(pdf_path)
    if actual != expected:
        msg = f"content_hash mismatch for {metadata.document_id}: manifest={expected} file={actual}"
        raise ValueError(msg)
    return pdf_path


def ingest_manifest(*, manifest_path: Path, output_dir: Path, repo_root: Path) -> list[Path]:
    """Load manifest, validate PDFs, extract text, write one JSON per document_id."""
    rows = load_manifest_documents(manifest_path)
    written: list[Path] = []
    for metadata in rows:
        pdf_path = _validate_pdf_exists_and_hash(metadata, repo_root=repo_root)
        pages = extract_document_pages(pdf_path=pdf_path, document_id=metadata.document_id)
        document = build_document(metadata=metadata, pages=pages)
        written.append(export_document_json(document=document, output_dir=output_dir))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest staged PDFs from a YAML manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest YAML (e.g. data/manifests/manual.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for processed JSON files (e.g. data/processed/documents).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for resolving manifest source_file paths (default: cwd).",
    )
    args = parser.parse_args(argv)

    try:
        paths = ingest_manifest(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            repo_root=args.repo_root.resolve(),
        )
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ingest_manifest: error: {exc}", file=sys.stderr)
        return 1

    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
