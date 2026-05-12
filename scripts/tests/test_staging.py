from __future__ import annotations

from pathlib import Path

import pytest

from staging_lib import (
    build_document_id,
    build_storage_basename,
    hash8_from_digest_hex,
    manual_pdf_relative_path,
    normalize_key_segment,
    sha256_hex_file,
    validate_effective_date,
    yaml_double_quoted_scalar,
)


def test_normalize_key_segment_basic() -> None:
    assert normalize_key_segment("Example Mutual") == "example_mutual"
    assert normalize_key_segment("  auto  ") == "auto"


def test_normalize_key_segment_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty after normalization"):
        normalize_key_segment("   !!!   ")


def test_validate_effective_date() -> None:
    assert validate_effective_date(" 2024-01-15 ") == "2024-01-15"


def test_validate_effective_date_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_effective_date("15-01-2024")


def test_hash8_from_digest_hex() -> None:
    digest = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert hash8_from_digest_hex(digest) == "2cf24dba"


def test_hash8_from_digest_hex_accepts_uppercase_input() -> None:
    digest = "2CF24DBA5FB0A30E26E83B2AC5B9E29E1B161E5C1FA7425E73043362938B9824"
    assert hash8_from_digest_hex(digest) == "2cf24dba"


def test_hash8_from_digest_hex_rejects_short_digest() -> None:
    with pytest.raises(ValueError, match="64"):
        hash8_from_digest_hex("abcd")


def test_hash8_from_digest_hex_rejects_non_hex() -> None:
    bad = "g" * 64
    with pytest.raises(ValueError, match="hex"):
        hash8_from_digest_hex(bad)


def test_build_storage_basename(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"hello")
    digest = sha256_hex_file(pdf_path)
    basename = build_storage_basename(
        insurer="Acme Co.",
        product_type="Auto",
        product_slug="sample_auto",
        document_type="policy",
        effective_date="2024-01-15",
        content_hash_hex=digest,
    )
    assert basename.endswith(".pdf")
    assert digest[:8] in basename
    assert basename.startswith("acme_co_auto_sample_auto_policy_2024-01-15_")


def test_build_document_id_and_relative_path() -> None:
    basename = "acme_auto_sample_policy_2024-01-15_deadbeef.pdf"
    assert build_document_id(basename) == "acme_auto_sample_policy_2024-01-15_deadbeef"
    assert manual_pdf_relative_path(basename) == f"data/raw/manual/{basename}"


def test_yaml_double_quoted_scalar_escapes() -> None:
    assert yaml_double_quoted_scalar('say "hi"') == '"say \\"hi\\""'
    assert yaml_double_quoted_scalar("a\nb") == '"a\\nb"'
    assert yaml_double_quoted_scalar("a\\b") == '"a\\\\b"'
