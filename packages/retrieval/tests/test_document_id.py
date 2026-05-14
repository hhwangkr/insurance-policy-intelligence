from __future__ import annotations

import pytest

from insurance_ai_retrieval.document_id import parse_document_id


@pytest.mark.parametrize(
    (
        "document_id",
        "insurer",
        "product_type",
        "slug_tail",
        "compact_date",
    ),
    [
        (
            "kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62",
            "kyobolife",
            "annuity",
            "kyobo_ro_annuity_insurance",
            "20260101",
        ),
        (
            "samsunglife_cancer_internet_cancer_insurance_policy_terms_20260101_39af0c18",
            "samsunglife",
            "cancer",
            "internet_cancer_insurance",
            "20260101",
        ),
        (
            "samsunglife_whole_life_balance_whole_life_insurance_policy_terms_20260301_1699395d",
            "samsunglife",
            "whole_life",
            "balance_whole_life_insurance",
            "20260301",
        ),
        (
            "miraeassetlife_variable_annuity_variable_annuity_insurance_policy_terms_20260401_76283b26",
            "miraeassetlife",
            "variable_annuity",
            "variable_annuity_insurance",
            "20260401",
        ),
    ],
)
def test_parse_document_id_known_manifest_ids(
    document_id: str,
    insurer: str,
    product_type: str,
    slug_tail: str,
    compact_date: str,
) -> None:
    parsed = parse_document_id(document_id)
    assert parsed is not None
    assert parsed.insurer == insurer
    assert parsed.product_type == product_type
    assert parsed.product_slug == slug_tail
    assert parsed.product_name == slug_tail.replace("_", " ")
    assert parsed.effective_date == compact_date
    assert parsed.content_hash == document_id.rsplit("_", 1)[-1]


def test_parse_document_id_unknown_layout_returns_none() -> None:
    assert parse_document_id("doc_a") is None
    assert parse_document_id("a_b_c_d_e_f") is None
