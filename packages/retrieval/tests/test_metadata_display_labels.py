from __future__ import annotations

from demo_corpus_fixtures import KYOBO_ANNUITY_SYNTHETIC_DOCUMENT_ID

from insurance_ai_retrieval.metadata import (
    ChunkMetadataRecord,
    apply_display_labels,
    display_label_for_insurer,
    display_label_for_product_type,
    enrich_chunk_metadata,
)


def test_display_label_insurer_known_and_unknown() -> None:
    assert display_label_for_insurer("kyobolife") == "교보생명"
    assert display_label_for_insurer("samsunglife") == "삼성생명"
    assert display_label_for_insurer("miraeassetlife") == "미래에셋생명"
    assert display_label_for_insurer("unknown_insurer") == "unknown_insurer"


def test_display_label_product_type_known_and_unknown() -> None:
    assert display_label_for_product_type("annuity") == "연금보험"
    assert display_label_for_product_type("variable_annuity") == "변액연금보험"
    assert display_label_for_product_type("cancer") == "암보험"
    assert display_label_for_product_type("whole_life") == "종신보험"
    assert display_label_for_product_type("unknown_pt") == "unknown_pt"


def test_enrich_chunk_metadata_sets_display_names() -> None:
    raw = ChunkMetadataRecord(
        chunk_id="c",
        section_id="s",
        section_title="t",
        section_type="article",
        document_id=KYOBO_ANNUITY_SYNTHETIC_DOCUMENT_ID,
        insurer=None,
        product_type=None,
        product_name=None,
        policy_unit_id=None,
        policy_unit_name=None,
        variant_name=None,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
        text="x",
    )
    out = enrich_chunk_metadata(raw)
    assert out.insurer == "kyobolife"
    assert out.product_type == "annuity"
    assert out.insurer_display_name == "교보생명"
    assert out.product_type_display_name == "연금보험"
    assert out.product_display_name == "x"


def test_apply_display_labels_on_complete_record() -> None:
    rec = ChunkMetadataRecord(
        chunk_id="c",
        section_id="s",
        section_title="t",
        section_type="article",
        document_id=KYOBO_ANNUITY_SYNTHETIC_DOCUMENT_ID,
        insurer="samsunglife",
        product_type="cancer",
        product_name="internet cancer insurance",
        policy_unit_id=None,
        policy_unit_name=None,
        variant_name=None,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
        text="x",
    )
    out = apply_display_labels(rec)
    assert out.insurer_display_name == "삼성생명"
    assert out.product_type_display_name == "암보험"
    assert out.product_display_name == "internet cancer insurance"
