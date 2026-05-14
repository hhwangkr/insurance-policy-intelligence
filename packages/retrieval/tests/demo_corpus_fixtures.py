"""Normalized ``document_id`` and slug constants for retrieval unit tests.

These values mirror the **staged manual demo corpus** (see ``data/manifests``). They exist so
tests can exercise **metadata parsing, enrichment, and filter wiring** with realistic strings
without scattering literals across files. They are **not** benchmark assertions that a given
PDF revision must remain in the repository, and they do **not** replace curated hit
expectations in ``data/eval/*.yaml``.
"""

from __future__ import annotations

INSURER_KYOBO = "kyobolife"
INSURER_SAMSUNG = "samsunglife"

PRODUCT_TYPE_ANNUITY = "annuity"
PRODUCT_TYPE_CANCER = "cancer"

KYOBO_ANNUITY_DOCUMENT_ID = (
    "kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62"
)
SAMSUNG_CANCER_DOCUMENT_ID = (
    "samsunglife_cancer_internet_cancer_insurance_policy_terms_20260101_39af0c18"
)
SAMSUNG_WHOLE_LIFE_DOCUMENT_ID = (
    "samsunglife_whole_life_balance_whole_life_insurance_policy_terms_20260301_1699395d"
)
MIRAE_VARIABLE_ANNUITY_DOCUMENT_ID = (
    "miraeassetlife_variable_annuity_variable_annuity_insurance_policy_terms_20260401_76283b26"
)

# Synthetic id matching the normalized filename pattern (used with tmp chunk artifacts).
KYOBO_ANNUITY_SYNTHETIC_DOCUMENT_ID = "kyobolife_annuity_x_policy_terms_20260101_ab12cd34"
