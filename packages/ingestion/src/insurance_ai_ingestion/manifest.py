from __future__ import annotations

from pathlib import Path

import yaml

from insurance_ai_shared.models.document import DocumentMetadata


def load_manifest_documents(path: Path) -> list[DocumentMetadata]:
    """Load and validate manifest rows as DocumentMetadata (staging-only fields ignored)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, dict):
        msg = "manifest root must be a mapping"
        raise ValueError(msg)
    documents_any = raw.get("documents")
    if documents_any is None:
        return []
    if not isinstance(documents_any, list):
        msg = "documents must be a list"
        raise ValueError(msg)
    return [DocumentMetadata.model_validate(row) for row in documents_any]


def resolve_source_pdf(*, repo_root: Path, source_file: str) -> Path:
    """Resolve a repo-relative manifest path to an absolute Path."""
    relative = Path(source_file)
    if relative.is_absolute():
        msg = "source_file must be repo-relative, not absolute"
        raise ValueError(msg)
    return (repo_root / relative).resolve()
