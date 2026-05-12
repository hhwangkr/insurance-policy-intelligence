from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from staging_lib import (
    build_document_id,
    build_storage_basename,
    copy_staged_pdf,
    format_manifest_entry_yaml,
    manual_pdf_relative_path,
    resolve_manual_source_pdf,
    sha256_hex_file,
    validate_effective_date,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_tags(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a PDF from data/inbox/manual/ (or another path) into data/raw/manual/ "
            "using a normalized hash-based filename, and print a manifest YAML snippet."
        ),
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help=(
            "PDF to stage: repo-relative path (preferred), e.g. data/inbox/manual/PolicyBook.pdf, "
            "or an absolute path."
        ),
    )
    parser.add_argument("--insurer", required=True, help="Insurer label (normalized for filename).")
    parser.add_argument(
        "--product-name", required=True, dest="product_name", help="Human-readable product name."
    )
    parser.add_argument(
        "--product-type", required=True, dest="product_type", help="Product type segment."
    )
    parser.add_argument(
        "--product-slug", required=True, dest="product_slug", help="Product slug segment."
    )
    parser.add_argument(
        "--document-type", required=True, dest="document_type", help="Document type segment."
    )
    parser.add_argument(
        "--effective-date",
        required=True,
        dest="effective_date",
        help="ISO effective date YYYY-MM-DD (used in filename).",
    )
    parser.add_argument(
        "--dataset-split", required=True, dest="dataset_split", help="Dataset split label."
    )
    parser.add_argument(
        "--source-url", default="", dest="source_url", help="Optional provenance URL."
    )
    parser.add_argument("--language", default="", help="Optional language code or label.")
    parser.add_argument("--notes", default="", help="Optional freeform notes.")
    parser.add_argument("--tags", default="", help="Optional comma-separated tags.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute hash and print manifest template without copying.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    source = resolve_manual_source_pdf(repo_root=repo_root, source=args.source)
    if not source.is_file():
        print(f"source is not a file: {source}")
        return 2
    if source.suffix.lower() != ".pdf":
        print("source file must have a .pdf extension")
        return 2

    content_hash = sha256_hex_file(source)
    effective_date = validate_effective_date(args.effective_date)
    basename = build_storage_basename(
        insurer=args.insurer,
        product_type=args.product_type,
        product_slug=args.product_slug,
        document_type=args.document_type,
        effective_date=effective_date,
        content_hash_hex=content_hash,
    )
    document_id = build_document_id(basename)
    destination = repo_root / "data" / "raw" / "manual" / basename
    relative_path = manual_pdf_relative_path(basename)
    collected_at = datetime.now(tz=UTC).isoformat()
    tags = _parse_tags(args.tags)

    if not args.dry_run:
        copy_staged_pdf(source=source, destination=destination, content_hash=content_hash)

    manifest_yaml = format_manifest_entry_yaml(
        document_id=document_id,
        insurer=args.insurer.strip(),
        product_name=args.product_name.strip(),
        product_type=args.product_type.strip(),
        product_slug=args.product_slug.strip(),
        document_type=args.document_type.strip(),
        effective_date=effective_date,
        source_file=relative_path,
        original_filename=source.name,
        source_url=args.source_url.strip(),
        content_hash=content_hash,
        collection_method="manual",
        dataset_split=args.dataset_split.strip(),
        language=args.language.strip(),
        collected_at=collected_at,
        tags=tags,
        notes=args.notes,
    )

    print("# Manifest entry template (paste under `documents:` in a manifest file)")
    print(manifest_yaml, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
