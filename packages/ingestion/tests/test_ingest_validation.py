from __future__ import annotations

from pathlib import Path

import fitz
import pytest
import yaml

from insurance_ai_ingestion.hashing import sha256_hex_file
from insurance_ai_ingestion.ingest_manifest import ingest_manifest


def _write_minimal_pdf(path: Path) -> None:
    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), "fixture text")
        doc.save(path)
    finally:
        doc.close()


def _manifest_row(
    *,
    document_id: str,
    source_file: str,
    content_hash: str,
) -> dict[str, object]:
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


def test_ingest_manifest_missing_source_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    pdf_dir = repo / "data" / "raw" / "manual"
    pdf_dir.mkdir(parents=True)
    pdf_path = pdf_dir / "missing.pdf"
    manifest_path = repo / "manual.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "documents": [
                    _manifest_row(
                        document_id="doc_missing",
                        source_file="data/raw/manual/missing.pdf",
                        content_hash="a" * 64,
                    )
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert not pdf_path.exists()
    out = repo / "out"
    with pytest.raises(FileNotFoundError, match="source_file not found"):
        ingest_manifest(manifest_path=manifest_path, output_dir=out, repo_root=repo)


def test_ingest_manifest_content_hash_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    pdf_dir = repo / "data" / "raw" / "manual"
    pdf_dir.mkdir(parents=True)
    pdf_path = pdf_dir / "one.pdf"
    _write_minimal_pdf(pdf_path)
    manifest_path = repo / "manual.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "documents": [
                    _manifest_row(
                        document_id="doc_bad_hash",
                        source_file="data/raw/manual/one.pdf",
                        content_hash="b" * 64,
                    )
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out = repo / "out"
    with pytest.raises(ValueError, match="content_hash mismatch"):
        ingest_manifest(manifest_path=manifest_path, output_dir=out, repo_root=repo)


def test_ingest_manifest_end_to_end_single_pdf(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    pdf_dir = repo / "data" / "raw" / "manual"
    pdf_dir.mkdir(parents=True)
    pdf_path = pdf_dir / "one.pdf"
    _write_minimal_pdf(pdf_path)
    digest = sha256_hex_file(pdf_path)
    manifest_path = repo / "manual.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "documents": [
                    _manifest_row(
                        document_id="doc_ok",
                        source_file="data/raw/manual/one.pdf",
                        content_hash=digest,
                    )
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out = repo / "processed"
    paths = ingest_manifest(manifest_path=manifest_path, output_dir=out, repo_root=repo)
    assert len(paths) == 1
    assert paths[0].name == "doc_ok.json"
    payload = paths[0].read_text(encoding="utf-8")
    assert "fixture text" in payload
    assert '"page_number": 1' in payload
