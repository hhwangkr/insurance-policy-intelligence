from __future__ import annotations

from pathlib import Path

from insurance_ai_shared.models.document import Document


def document_json_path(*, output_dir: Path, document_id: str) -> Path:
    """Return the target JSON path for a document_id (one file per document)."""
    safe_id = document_id.strip()
    if not safe_id:
        msg = "document_id must be non-empty"
        raise ValueError(msg)
    return output_dir / f"{safe_id}.json"


def export_document_json(*, document: Document, output_dir: Path) -> Path:
    """Write a single processed document JSON file; create parent directories if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = document_json_path(output_dir=output_dir, document_id=document.document_id)
    path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    return path
