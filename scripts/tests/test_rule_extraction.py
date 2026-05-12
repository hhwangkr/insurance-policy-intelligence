from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from metadata_models import FieldInference, StagingManifestEntry
from rule_extraction import (
    infer_document_type,
    infer_effective_date,
    infer_insurer,
    infer_product_name,
    infer_product_type,
    parse_effective_date_candidates,
)
from staging_lib import build_storage_basename, sha256_hex_file, validate_effective_date


def test_infer_product_name_skips_generic_cover_lines() -> None:
    text = "보험약관\n목 차\n고객권리안내문\n약관 이용 Guide Book\n삼성 인터넷암보험\n"
    field = infer_product_name(text=text, filename="cover.pdf")
    assert field.value.strip() == "삼성 인터넷암보험"
    assert field.evidence.startswith("pdf_text:keyword:")


def test_infer_product_name_selects_samsung_internet_cancer() -> None:
    text = "보험약관\n삼성 인터넷암보험\n"
    field = infer_product_name(text=text, filename="x.pdf")
    assert "삼성 인터넷암보험" in field.value
    assert "암보험" in field.evidence


def test_infer_product_name_selects_kyobo_integrated_cancer() -> None:
    text = "약관 이용 Guide Book\n교보간편통합암보험\n"
    field = infer_product_name(text=text, filename="x.pdf")
    assert "교보간편통합암보험" in field.value
    assert "통합암보험" in field.evidence


def test_storage_basename_uses_yyyymmdd_manifest_iso_preserved(tmp_path: Path) -> None:
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"x")
    digest = sha256_hex_file(pdf)
    iso = "2026-01-01"
    basename = build_storage_basename(
        insurer="kyobolife",
        product_type="annuity",
        product_slug="pension",
        document_type="policy_terms",
        effective_date=iso,
        content_hash_hex=digest,
    )
    assert "20260101" in basename
    assert "2026-01-01" not in basename
    assert validate_effective_date(iso) == "2026-01-01"


def test_infer_insurer_samsunglife_high() -> None:
    text = "삼성생명 보험 약관"
    field = infer_insurer(text=text, filename="x.pdf")
    assert field.value == "samsunglife"
    assert field.confidence == "high"
    assert field.needs_review is False


def test_infer_insurer_kyobo_from_filename() -> None:
    text = ""
    field = infer_insurer(text=text, filename="교보생명_약관.pdf")
    assert field.value == "kyobolife"
    assert field.confidence == "high"


def test_infer_product_type_annuity_from_korean() -> None:
    hay = "개인연금저축 상품"
    field = infer_product_type(text=hay, filename="a.pdf")
    assert field.value == "annuity"
    assert field.confidence == "high"


def test_infer_document_type_policy_terms() -> None:
    field = infer_document_type(text="", filename="공시용_통합약관.pdf")
    assert field.value == "policy_terms"
    assert field.needs_review is True


def test_parse_effective_date_yymmdd_filename() -> None:
    cands = parse_effective_date_candidates(text="", filename="prefix_260101_suffix.pdf")
    assert any(iso == "2026-01-01" for iso, _ev, _c in cands)


def test_infer_effective_date_prefers_iso() -> None:
    field = infer_effective_date(text="기준일 2025-06-01", filename="260101_x.pdf")
    assert field.value == "2025-06-01"


def test_build_storage_basename_with_inferred_segments(tmp_path: Path) -> None:
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"abc")
    digest = sha256_hex_file(pdf)
    name = build_storage_basename(
        insurer="kyobolife",
        product_type="annuity",
        product_slug="personal_pension",
        document_type="policy_terms",
        effective_date="2026-01-01",
        content_hash_hex=digest,
    )
    assert name.endswith(".pdf")
    assert digest[:8] in name
    assert "20260101" in name
    assert "2026-01-01" not in name


def test_manifest_entry_inference_flags() -> None:
    inference = {
        "insurer": FieldInference(
            value="kyobolife",
            confidence="high",
            needs_review=False,
            evidence="test",
        ),
        "dataset_split": FieldInference(
            value="unassigned",
            confidence="low",
            needs_review=True,
            evidence="default",
        ),
    }
    entry = StagingManifestEntry(
        document_id="doc",
        insurer="kyobolife",
        product_name="Test",
        product_type="annuity",
        product_slug="test",
        document_type="policy_terms",
        effective_date="2026-01-01",
        source_file="data/raw/manual/doc.pdf",
        original_filename="in.pdf",
        content_hash="0" * 64,
        collected_at="2026-01-01T00:00:00+00:00",
        inference=inference,
    )
    dumped = entry.to_yaml_dict()
    inf_map = cast(dict[str, Any], dumped["inference"])
    ds = cast(dict[str, Any], inf_map["dataset_split"])
    assert ds["needs_review"] is True
    ins = cast(dict[str, Any], inf_map["insurer"])
    assert ins["needs_review"] is False


def test_build_staging_manifest_entry_smoke(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    import fitz

    from inbox_staging import build_staging_manifest_entry

    inbox = tmp_path / "data" / "inbox" / "manual"
    inbox.mkdir(parents=True)
    pdf_path = inbox / "1000341_개인연금저축교보로연금보험_260101 공시용 통합약관.pdf"
    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), "교보생명 개인연금보험 보험약관")
        doc.save(pdf_path)
    finally:
        doc.close()

    entry, err = build_staging_manifest_entry(pdf_path=pdf_path, repo_root=tmp_path, max_pages=3)
    assert err == ""
    assert entry is not None
    assert entry.insurer == "kyobolife"
    assert entry.product_type == "annuity"
    assert entry.document_type == "policy_terms"
    assert entry.effective_date == "2026-01-01"
    assert "20260101" in entry.source_file
    assert entry.original_filename == pdf_path.name
    assert entry.source_file.startswith("data/raw/manual/")
    assert entry.content_hash == sha256_hex_file(pdf_path)
    assert any(field.needs_review for field in entry.inference.values())
