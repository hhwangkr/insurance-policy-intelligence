from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from metadata_models import FieldInference, StagingManifestEntry
from rule_extraction import (
    extract_pdf_head_text,
    infer_dataset_split,
    infer_document_type,
    infer_effective_date,
    infer_insurer,
    infer_language,
    infer_product_name,
    infer_product_slug,
    infer_product_type,
)
from staging_lib import (
    build_document_id,
    build_storage_basename,
    copy_staged_pdf,
    manual_pdf_relative_path,
    sha256_hex_file,
    validate_effective_date,
)


def build_staging_manifest_entry(
    *,
    pdf_path: Path,
    repo_root: Path,
    max_pages: int,
) -> tuple[StagingManifestEntry | None, str]:
    """Build a manifest entry from inbox PDF + rule extraction. Empty reason means success."""
    filename = pdf_path.name
    try:
        head_text = extract_pdf_head_text(pdf_path, max_pages=max_pages)
    except Exception as exc:  # noqa: BLE001 - surface PyMuPDF failures clearly
        return None, f"pdf_read_failed:{exc.__class__.__name__}"

    insurer = infer_insurer(text=head_text, filename=filename)
    document_type = infer_document_type(text=head_text, filename=filename)
    effective = infer_effective_date(text=head_text, filename=filename)
    product_name = infer_product_name(text=head_text, filename=filename)
    product_type = infer_product_type(
        text=head_text,
        filename=filename,
        product_name=product_name.value,
    )
    language = infer_language(text=head_text)
    dataset_split = infer_dataset_split()

    if not insurer.value.strip():
        return None, "insurer_not_inferred"
    if not document_type.value.strip():
        return None, "document_type_not_inferred"
    if not product_type.value.strip():
        return None, "product_type_not_inferred"
    if not effective.value.strip():
        return None, "effective_date_not_inferred"

    if not product_name.value.strip():
        product_name = FieldInference(
            value=Path(filename).stem,
            confidence="low",
            needs_review=True,
            evidence="fallback:filename_stem",
        )
        product_type = infer_product_type(
            text=head_text,
            filename=filename,
            product_name=product_name.value,
        )
        if not product_type.value.strip():
            return None, "product_type_not_inferred"

    slug = infer_product_slug(
        product_name=product_name.value,
        product_type=product_type.value,
    )

    try:
        effective_iso = validate_effective_date(effective.value)
    except ValueError:
        return None, "effective_date_invalid"

    content_hash = sha256_hex_file(pdf_path)
    basename = build_storage_basename(
        insurer=insurer.value,
        product_type=product_type.value,
        product_slug=slug.value,
        document_type=document_type.value,
        effective_date=effective_iso,
        content_hash_hex=content_hash,
    )
    document_id = build_document_id(basename)
    source_file = manual_pdf_relative_path(basename)
    collected_at = datetime.now(tz=UTC).isoformat()

    inference: dict[str, FieldInference] = {
        "insurer": insurer,
        "document_type": document_type,
        "product_type": product_type,
        "effective_date": effective,
        "product_name": product_name,
        "product_slug": slug,
        "language": language,
        "dataset_split": dataset_split,
    }

    review_fields = sorted(name for name, field in inference.items() if field.needs_review)
    notes = f"rule_extract_needs_review:{','.join(review_fields)}" if review_fields else ""

    entry = StagingManifestEntry(
        document_id=document_id,
        insurer=insurer.value,
        product_name=product_name.value,
        product_type=product_type.value,
        product_slug=slug.value,
        document_type=document_type.value,
        effective_date=effective_iso,
        source_file=source_file,
        original_filename=filename,
        content_hash=content_hash,
        collection_method="manual_inbox_rules",
        dataset_split=dataset_split.value,
        language=language.value,
        collected_at=collected_at,
        tags=[],
        notes=notes,
        inference=inference,
    )
    return entry, ""


def format_inbox_staging_preview_block(
    *,
    original_filename: str,
    entry: StagingManifestEntry | None,
    error: str,
) -> str:
    """Human-readable, fixed-width-table-free preview (safe for long Hangul paths)."""
    lines: list[str] = ["---", f"original_filename: {original_filename!r}"]
    if entry is None:
        lines.append("status: skipped")
        lines.append(f"error: {error}")
        return "\n".join(lines) + "\n"

    review_fields = sorted(name for name, field in entry.inference.items() if field.needs_review)
    lines.append("status: ok")
    lines.append(f"document_id: {entry.document_id!r}")
    lines.append(f"insurer: {entry.insurer!r}")
    lines.append(f"product_name: {entry.product_name!r}")
    lines.append(f"product_type: {entry.product_type!r}")
    lines.append(f"product_slug: {entry.product_slug!r}")
    lines.append(f"document_type: {entry.document_type!r}")
    lines.append(f"effective_date: {entry.effective_date!r}")
    lines.append(f"source_file: {entry.source_file!r}")
    lines.append("needs_review_fields:")
    if review_fields:
        for name in review_fields:
            lines.append(f"  - {name}")
    else:
        lines.append("  []")
    lines.append("inference:")
    for name in sorted(entry.inference.keys()):
        field = entry.inference[name]
        lines.append(f"  {name}:")
        lines.append(f"    value: {field.value!r}")
        lines.append(f"    confidence: {field.confidence!r}")
        lines.append(f"    needs_review: {field.needs_review}")
        lines.append(f"    evidence: {field.evidence!r}")
    if entry.notes:
        lines.append(f"notes: {entry.notes!r}")
    return "\n".join(lines) + "\n"


def run_inbox_staging(*, repo_root: Path, apply: bool, max_pages: int) -> int:
    inbox = repo_root / "data" / "inbox" / "manual"
    if not inbox.is_dir():
        print(
            "data/inbox/manual/ is missing; create it before running this command.", file=sys.stderr
        )
        return 2

    pdfs = sorted(p for p in inbox.glob("*.pdf") if p.is_file())
    if not pdfs:
        print("No *.pdf files found under data/inbox/manual/.")
        return 0

    entries: list[StagingManifestEntry] = []
    failures: list[tuple[str, str]] = []

    print("Inferred metadata (rule-based; verify needs_review_fields before committing).")

    for pdf in pdfs:
        entry, err = build_staging_manifest_entry(
            pdf_path=pdf, repo_root=repo_root, max_pages=max_pages
        )
        if entry is None:
            failures.append((pdf.name, err))
            print(
                format_inbox_staging_preview_block(
                    original_filename=pdf.name,
                    entry=None,
                    error=err,
                ),
                end="",
            )
            continue

        if apply:
            destination = repo_root / entry.source_file
            try:
                copy_staged_pdf(
                    source=pdf,
                    destination=destination,
                    content_hash=entry.content_hash,
                )
            except FileExistsError:
                failures.append((pdf.name, "destination_conflict"))
                print(
                    format_inbox_staging_preview_block(
                        original_filename=pdf.name,
                        entry=None,
                        error="destination_conflict",
                    ),
                    end="",
                )
                continue
            entries.append(entry)

        print(
            format_inbox_staging_preview_block(
                original_filename=pdf.name,
                entry=entry,
                error="",
            ),
            end="",
        )

    if not apply:
        print(
            "Preview only: no files copied and manual.yaml not written. "
            "Re-run with --apply after verification.",
        )

    for name, err in failures:
        print(f"SKIPPED {name}: {err}", file=sys.stderr)

    manifest_path = repo_root / "data" / "manifests" / "manual.yaml"
    if apply:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_by": "stage_manual_inbox",
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "documents": [entry.to_yaml_dict() for entry in entries],
        }
        manifest_path.write_text(
            yaml.safe_dump(
                payload,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        print("Wrote manifest: data/manifests/manual.yaml")

    if failures:
        return 3
    return 0
