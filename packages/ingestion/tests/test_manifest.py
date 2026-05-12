from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from insurance_ai_ingestion.manifest import load_manifest_documents, resolve_source_pdf


def _minimal_row(*, document_id: str, source_file: str, content_hash: str) -> dict[str, object]:
    return {
        "document_id": document_id,
        "insurer": "acme",
        "product_name": "Example Product",
        "product_type": "auto",
        "product_slug": "example",
        "document_type": "policy",
        "effective_date": "2026-01-01",
        "source_file": source_file,
        "original_filename": "in.pdf",
        "source_url": "",
        "content_hash": content_hash,
        "collection_method": "manual",
        "dataset_split": "train",
        "language": "en",
        "tags": [],
        "notes": "",
    }


def test_load_manifest_documents_empty_documents_key(tmp_path: Path) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump({"documents": []}, sort_keys=False),
        encoding="utf-8",
    )
    assert load_manifest_documents(manifest) == []


def test_load_manifest_documents_ignores_staging_only_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "m.yaml"
    row = _minimal_row(
        document_id="doc_one",
        source_file="data/raw/manual/x.pdf",
        content_hash="a" * 64,
    )
    row["collected_at"] = "2026-01-01T00:00:00Z"
    row["inference"] = {"insurer": {"value": "x", "confidence": "low"}}
    manifest.write_text(
        yaml.safe_dump({"documents": [row], "generated_by": "test"}, sort_keys=False),
        encoding="utf-8",
    )
    loaded = load_manifest_documents(manifest)
    assert len(loaded) == 1
    assert loaded[0].document_id == "doc_one"


def test_load_manifest_documents_rejects_non_mapping_root(tmp_path: Path) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_manifest_documents(manifest)


def test_resolve_source_pdf_builds_path(tmp_path: Path) -> None:
    pdf = tmp_path / "data" / "raw" / "manual" / "f.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4\n")
    resolved = resolve_source_pdf(repo_root=tmp_path, source_file="data/raw/manual/f.pdf")
    assert resolved == pdf.resolve()


def test_resolve_source_pdf_rejects_absolute_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repo-relative"):
        resolve_source_pdf(repo_root=tmp_path, source_file=str(tmp_path / "x.pdf"))
