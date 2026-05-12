from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

_SHA256_HEX_LEN = 64
_EFFECTIVE_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_SEGMENT = re.compile(r"[^a-z0-9]+")


def sha256_hex_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file's raw bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash8_from_digest_hex(digest_hex: str) -> str:
    """Return the first 8 hex characters of a full SHA-256 hex digest."""
    normalized = digest_hex.strip().lower()
    if len(normalized) < _SHA256_HEX_LEN:
        msg = f"expected a {_SHA256_HEX_LEN}-char hex digest, got {len(normalized)}"
        raise ValueError(msg)
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError("digest must be 64 lowercase hex characters")
    return normalized[:8]


def normalize_key_segment(value: str) -> str:
    """Normalize a filename segment: lowercase, collapse non-alphanumeric to single underscores."""
    lowered = value.strip().lower()
    collapsed = _SEGMENT.sub("_", lowered)
    trimmed = collapsed.strip("_")
    if not trimmed:
        msg = "segment is empty after normalization"
        raise ValueError(msg)
    return trimmed


def validate_effective_date(value: str) -> str:
    """Validate and return ISO effective date YYYY-MM-DD (manifest and inputs)."""
    candidate = value.strip()
    if not _EFFECTIVE_DATE.fullmatch(candidate):
        msg = "effective_date must be ISO YYYY-MM-DD"
        raise ValueError(msg)
    return candidate


def effective_date_storage_segment(iso_date: str) -> str:
    """Return YYYYMMDD filename segment from a validated ISO date."""
    iso = validate_effective_date(iso_date)
    return iso.replace("-", "")


def build_storage_basename(
    *,
    insurer: str,
    product_type: str,
    product_slug: str,
    document_type: str,
    effective_date: str,
    content_hash_hex: str,
) -> str:
    """Build `{segments}.pdf` under data/raw/manual conventions.

    ``effective_date`` must be ISO ``YYYY-MM-DD``; the on-disk segment uses ``YYYYMMDD``.
    """
    parts = [
        normalize_key_segment(insurer),
        normalize_key_segment(product_type),
        normalize_key_segment(product_slug),
        normalize_key_segment(document_type),
        effective_date_storage_segment(effective_date),
        hash8_from_digest_hex(content_hash_hex),
    ]
    return "_".join(parts) + ".pdf"


def build_document_id(storage_basename: str) -> str:
    """Document id matches the storage basename without extension."""
    if not storage_basename.lower().endswith(".pdf"):
        msg = "storage basename must end with .pdf"
        raise ValueError(msg)
    return storage_basename[:-4]


def manual_pdf_relative_path(storage_basename: str) -> str:
    """Repository-relative path for a staged manual PDF."""
    return f"data/raw/manual/{storage_basename}"


def resolve_manual_source_pdf(*, repo_root: Path, source: Path) -> Path:
    """Resolve a PDF path: absolute paths as-is; repo-relative paths against repo_root."""
    expanded = source.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (repo_root / expanded).resolve()


def copy_staged_pdf(*, source: Path, destination: Path, content_hash: str) -> None:
    """Copy a PDF into raw/manual, skipping if an identical file already exists."""
    if destination.exists():
        existing_hash = sha256_hex_file(destination)
        if existing_hash == content_hash:
            return
        msg = f"destination exists with different content: {destination}"
        raise FileExistsError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def yaml_double_quoted_scalar(value: str) -> str:
    """Emit a YAML double-quoted scalar (minimal escaping)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def format_tags_yaml(tags: list[str]) -> str:
    """Format tags as a YAML flow sequence."""
    if not tags:
        return "tags: []"
    inner = ", ".join(yaml_double_quoted_scalar(tag) for tag in tags)
    return f"tags: [{inner}]"


def format_manifest_entry_yaml(
    *,
    document_id: str,
    insurer: str,
    product_name: str,
    product_type: str,
    product_slug: str,
    document_type: str,
    effective_date: str,
    source_file: str,
    original_filename: str,
    source_url: str,
    content_hash: str,
    collection_method: str,
    dataset_split: str,
    language: str,
    collected_at: str,
    tags: list[str],
    notes: str,
) -> str:
    """Single manifest document as a YAML list item (stdout-friendly)."""
    lines = [
        f"- document_id: {yaml_double_quoted_scalar(document_id)}",
        f"  insurer: {yaml_double_quoted_scalar(insurer)}",
        f"  product_name: {yaml_double_quoted_scalar(product_name)}",
        f"  product_type: {yaml_double_quoted_scalar(product_type)}",
        f"  product_slug: {yaml_double_quoted_scalar(product_slug)}",
        f"  document_type: {yaml_double_quoted_scalar(document_type)}",
        f"  effective_date: {yaml_double_quoted_scalar(effective_date)}",
        f"  source_file: {yaml_double_quoted_scalar(source_file)}",
        f"  original_filename: {yaml_double_quoted_scalar(original_filename)}",
        f"  source_url: {yaml_double_quoted_scalar(source_url)}",
        f"  content_hash: {yaml_double_quoted_scalar(content_hash)}",
        f"  collection_method: {yaml_double_quoted_scalar(collection_method)}",
        f"  dataset_split: {yaml_double_quoted_scalar(dataset_split)}",
        f"  language: {yaml_double_quoted_scalar(language)}",
        f"  collected_at: {yaml_double_quoted_scalar(collected_at)}",
        f"  {format_tags_yaml(tags)}",
        f"  notes: {yaml_double_quoted_scalar(notes)}",
    ]
    return "\n".join(lines) + "\n"
