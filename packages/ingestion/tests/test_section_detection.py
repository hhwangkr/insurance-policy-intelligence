from __future__ import annotations

from datetime import UTC, date, datetime

from insurance_ai_ingestion.section_detection import (
    assemble_document_sections,
    build_sections_artifact,
    collect_section_candidates,
    detect_sections,
    run_section_detection,
)
from insurance_ai_shared.models.document import Document, DocumentMetadata, DocumentPage
from insurance_ai_shared.models.section import PageRegion


def _meta(document_id: str) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        insurer="test_insurer",
        product_name="테스트상품",
        product_type="annuity",
        product_slug="test_product",
        document_type="policy_terms",
        effective_date=date(2026, 1, 1),
        source_file="data/raw/manual/x.pdf",
        original_filename="x.pdf",
        content_hash="a" * 64,
        collection_method="test",
        dataset_split="unassigned",
        language="ko",
    )


def _page(document_id: str, n: int, text: str) -> DocumentPage:
    return DocumentPage(
        document_id=document_id,
        page_number=n,
        text=text,
        char_count=len(text),
        extraction_method="pymupdf",
    )


def test_detects_gwan_and_article_parent_and_pages() -> None:
    doc_id = "doc_sections_sample"
    pages = [
        _page(doc_id, 1, "\n".join(["표지 내용", "[ 목 차 ]", "일반 안내"])),
        _page(doc_id, 2, ""),
        _page(
            doc_id,
            3,
            "\n".join(
                [
                    "제1관 목적 및 용어의 정의",
                    "본 관에서는 목적을 설명합니다.",
                    "제1조 (목적)",
                    "이 계약의 목적은 ...",
                    "제2조 (용어의 정의)",
                    "용어는 다음과 같습니다.",
                ]
            ),
        ),
        _page(doc_id, 4, "본문 계속"),
        _page(
            doc_id,
            5,
            "\n".join(
                [
                    "제2관 보험금의 지급",
                    "보험금 지급에 관한 사항",
                    "제3조 (보험금의 지급사유)",
                    "지급사유 조항",
                    "제5조 (보험금을 지급하지 않는 사유)",
                    "면책 조항",
                ]
            ),
        ),
        _page(doc_id, 6, "\n".join(["제32조 (해약환급금)", "환급 규정"])),
    ]
    created = datetime(2026, 5, 12, tzinfo=UTC)
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=len(pages),
        total_char_count=sum(p.char_count for p in pages),
        created_at=created,
    )
    sections = detect_sections(doc)

    kinds = [s.section_type for s in sections]
    assert "toc" in kinds
    assert "part" in kinds
    assert "article" in kinds

    toc = next(s for s in sections if s.section_type == "toc")
    assert toc.start_page == 1
    assert toc.section_id == f"{doc_id}::sec::0000"

    first_part = next(s for s in sections if s.title.startswith("제1관"))
    assert first_part.section_type == "part"
    assert first_part.start_page == 3

    second_part = next(s for s in sections if "제2관" in s.title)
    assert second_part.start_page == 5

    art1 = next(s for s in sections if s.title.startswith("제1조"))
    assert art1.section_type == "article"
    assert art1.parent_section_id == first_part.section_id

    art3 = next(s for s in sections if "제3조" in s.title)
    assert art3.section_type == "article"
    assert art3.parent_section_id == second_part.section_id

    sec32 = next(s for s in sections if "제32조" in s.title)
    assert sec32.start_page == 6
    assert sec32.end_page >= 6


def test_appendix_legal_and_glossary_headings() -> None:
    """Policy-body page with substantive tails after each heading (not front-matter guide)."""
    doc_id = "doc_appendix_sample"
    filler_a = "별표 본문 설명입니다." * 12
    filler_b = "특별약관 관련 본문입니다." * 12
    filler_l = "민법 및 보험업법 등 관련 법령 조항을 인용합니다." * 6
    filler_g = "보험 용어에 대한 해설 문단입니다." * 10
    pages = [
        _page(
            doc_id,
            1,
            "\n".join(
                [
                    "제1관 목적 및 용어의 정의",
                    "본 문서는 약관 본문 영역으로 분류되도록 충분히 길게 작성합니다." * 30,
                    "( 별표 1 ) 보험금 지급기준표",
                    filler_a,
                    "별표 2 특별약관",
                    filler_b,
                    "약관에서 인용한 법·규정",
                    filler_l,
                    "보험용어 해설",
                    filler_g,
                ]
            ),
        ),
    ]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(pages[0].text),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    kinds = {s.section_type for s in sections}
    assert "appendix" in kinds
    assert "legal_reference" in kinds
    assert "glossary" in kinds


def test_toc_appendix_pointer_not_emitted_as_section() -> None:
    doc_id = "doc_toc_appendix_pointer"
    pad = "." * 40
    toc_block = "\n".join(
        [
            "[ 목 차 ]",
            f"( 별표1 ) 보험금지급기준표{pad} 48",
            f"( 별표2 ) 재해분류표{pad} 50",
        ]
    )
    body = "\n".join(
        [
            "제1관 목적 및 용어의 정의",
            "실제 약관 본문입니다." * 40,
            "제1조 (목적)",
            "계약의 목적은 ...",
        ]
    )
    pages = [_page(doc_id, 1, toc_block), _page(doc_id, 2, body)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=2,
        total_char_count=sum(p.char_count for p in pages),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    toc_appendix = [s for s in sections if s.section_type == "appendix" and s.start_page == 1]
    assert toc_appendix == []


def test_guide_legal_pointer_not_emitted() -> None:
    doc_id = "doc_guide_legal_pointer"
    guide_page = "\n".join(
        [
            "고객 안내",
            "관련법규 168p",
            "관련법규 항목을 활용하시면 편리합니다.",
        ]
    )
    body = "\n".join(
        [
            "제1관 목적 및 용어의 정의",
            "실제 약관 본문입니다." * 40,
            "제1조 (목적)",
            "목적 조항 본문입니다.",
        ]
    )
    pages = [_page(doc_id, 1, guide_page), _page(doc_id, 2, body)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=2,
        total_char_count=sum(p.char_count for p in pages),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    legal_on_p1 = [s for s in sections if s.section_type == "legal_reference" and s.start_page == 1]
    assert legal_on_p1 == []


def test_real_appendix_with_substantive_body_emitted() -> None:
    doc_id = "doc_real_appendix"
    body_tail = "별표 세부 기준 본문입니다." * 15
    text = "\n".join(
        [
            "제1관 목적 및 용어의 정의",
            "들여쓰기 없는 긴 본문 영역입니다." * 35,
            "( 별표 1 ) 보험금 지급기준표",
            body_tail,
        ]
    )
    pages = [_page(doc_id, 1, text)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(text),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    apx = [s for s in sections if s.section_type == "appendix"]
    assert len(apx) == 1
    assert "별표" in apx[0].title


def test_section_ids_are_deterministic_across_runs() -> None:
    doc_id = "doc_deterministic"
    body = "\n".join(["제1관 목적 및 용어의 정의", "내용", "제1조 (목적)", "조항"])
    pages = [_page(doc_id, 1, body)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    first = [s.section_id for s in detect_sections(doc)]
    second = [s.section_id for s in detect_sections(doc)]
    assert first == second


def test_empty_pages_do_not_break_detection() -> None:
    doc_id = "doc_empty_pages"
    pages = [
        _page(doc_id, 1, ""),
        _page(doc_id, 2, "   \n  \n"),
        _page(doc_id, 3, "제1관 목적 및 용어의 정의\n내용"),
        _page(doc_id, 4, ""),
    ]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=4,
        total_char_count=sum(p.char_count for p in pages),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    assert len(sections) >= 1
    assert any(s.section_type == "part" for s in sections)


def test_build_sections_artifact_includes_regions_and_candidates() -> None:
    doc_id = "doc_artifact_enriched"
    pages = [_page(doc_id, 1, "제1조 (목적)\n내용")]
    created = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(pages[0].text),
        created_at=created,
    )
    art = build_sections_artifact(document=doc)
    assert art.document_id == doc_id
    assert len(art.page_regions) == 1
    assert art.page_regions[0].page_number == 1
    assert len(art.section_candidates) >= 1


def test_toc_like_page_suppresses_article_outline_candidates() -> None:
    """Many dotted TOC lines + 조 outlines should not become article sections."""
    doc_id = "doc_toc_noise"
    pad = "." * 48
    dotted_toc = "\n".join(
        [
            "[ 목 차 ]",
            f"미래에셋생명 변액연금보험 무배당{pad} 3",
            f"제1조 (목적) {pad} 10",
            f"제2조 (용어의 정의) {pad} 11",
            f"제3조 (보험금의 지급사유) {pad} 12",
            f"제5조 (보험금을 지급하지 않는 사유) {pad} 13",
        ]
    )
    body = "\n".join(
        [
            "제1관 목적 및 용어의 정의",
            "실제 약관 본문이 시작됩니다.",
            "제1조 (목적)",
            "계약 목적 조항 본문입니다.",
        ]
    )
    pages = [
        _page(doc_id, 1, dotted_toc),
        _page(doc_id, 2, body),
    ]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=2,
        total_char_count=sum(p.char_count for p in pages),
        created_at=datetime.now(UTC),
    )
    candidates = collect_section_candidates(doc)
    article_like = [c for c in candidates if c.section_type == "article"]
    assert len(article_like) >= 5

    sections, _, regions = run_section_detection(doc)
    p1_region = next(r for r in regions if r.page_number == 1)
    assert p1_region.region_type == "toc"

    kept_articles_p1 = [s for s in sections if s.section_type == "article" and s.start_page == 1]
    assert kept_articles_p1 == []

    kept_parts_p2 = [s for s in sections if s.section_type == "part" and s.start_page == 2]
    assert len(kept_parts_p2) == 1


def test_toc_like_backstop_suppresses_kyobo_style_article_pointer() -> None:
    doc_id = "doc_kyobo_article_pointer"
    body = "제29조(계약자의임의해지)\n. 39"
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=[_page(doc_id, 1, body)],
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    assert [s for s in sections if s.section_type == "article"] == []


def test_toc_like_backstop_suppresses_article_pointer_variants() -> None:
    for body in (
        "제1조(목적)\n. 57",
        "제15조 (청약의 철회)\n25p",
    ):
        doc_id = "doc_ptr_variant"
        doc = Document(
            document_id=doc_id,
            metadata=_meta(doc_id),
            pages=[_page(doc_id, 1, body)],
            page_count=1,
            total_char_count=len(body),
            created_at=datetime.now(UTC),
        )
        sections = detect_sections(doc)
        assert [s for s in sections if s.section_type == "article"] == []


def test_toc_like_backstop_suppresses_appendix_pointer_two_lines() -> None:
    doc_id = "doc_apx_ptr"
    body = "( 별표1 ) 보험금지급기준표\n. 48"
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=[_page(doc_id, 1, body)],
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    assert [s for s in sections if s.section_type == "appendix"] == []


def test_toc_like_backstop_suppresses_compact_legal_pointer() -> None:
    doc_id = "doc_legal_ptr"
    body = "약관에서인용된법령\n. 168"
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=[_page(doc_id, 1, body)],
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    assert [s for s in sections if s.section_type == "legal_reference"] == []


def test_kyobo_style_legal_glossary_toc_ladder_suppressed() -> None:
    doc_id = "doc_kyobo_legal_toc_ladder"
    body = "\n".join(
        [
            "약관에서인용된법령",
            ". 168",
            "보험용어해설",
            ". 367",
            "",
            "Ⅰ. 보험약관 가이드",
            "안내 문단입니다.",
        ]
    )
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=[_page(doc_id, 1, body)],
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    assert [s for s in sections if s.section_type == "legal_reference"] == []


def test_legal_inline_page_pointer_line_suppressed() -> None:
    doc_id = "doc_legal_inline"
    body = "약관에서 인용한 법·규정 168p\n다음 안내는 요약입니다."
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=[_page(doc_id, 1, body)],
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    assert [s for s in sections if s.section_type == "legal_reference"] == []


def test_legal_pointer_then_guide_suppressed() -> None:
    doc_id = "doc_legal_then_guide"
    body = "\n".join(
        [
            "약관에서 인용된 법령",
            ". 168",
            "보험약관 가이드",
            "이어지는 안내입니다.",
        ]
    )
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=[_page(doc_id, 1, body)],
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    assert [s for s in sections if s.section_type == "legal_reference"] == []


def test_substantive_legal_reference_emitted_with_law_corpus() -> None:
    doc_id = "doc_real_legal"
    body = "\n".join(
        [
            "약관에서 인용한 법·규정",
            "이 약관은 민법 제103조, 보험업법 제95조 등 관계 법령을 따릅니다.",
            "금융소비자보호법에 따른 설명의무를 이행합니다.",
        ]
    )
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=[_page(doc_id, 1, body)],
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    legal = [s for s in sections if s.section_type == "legal_reference"]
    assert len(legal) == 1
    assert "민법" in legal[0].text or "보험업법" in legal[0].text


def test_real_body_article_not_suppressed_when_followed_by_clause_text() -> None:
    doc_id = "doc_real_article_body"
    clause = "이 보험계약의 목적은 피보험자의 생존 또는 사망 시 보험금을 지급하는 데 있습니다."
    body = "제1조(목적)\n" + clause * 2
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=[_page(doc_id, 1, body)],
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    arts = [s for s in sections if s.section_type == "article"]
    assert len(arts) == 1
    assert arts[0].title.startswith("제1조")


def test_real_appendix_with_substantive_table_body_still_emitted() -> None:
    doc_id = "doc_substantive_appendix"
    filler = "지급 구분 및 산출 예시에 대한 상세 설명입니다." * 8
    body = "\n".join(
        [
            "제1관 목적 및 용어의 정의",
            "실제 본문 영역입니다." * 60,
            "( 별표 1 ) 보험금 지급기준표",
            filler,
        ]
    )
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=[_page(doc_id, 1, body)],
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    apx = [s for s in sections if s.section_type == "appendix"]
    assert len(apx) >= 1


def test_compact_part_heading_detected_as_part() -> None:
    doc_id = "doc_compact_gwan"
    body = "\n".join(
        [
            "제2관보험금의지급",
            "이 관에서는 보험금 지급을 규정합니다.",
            "제10조 (보험금)",
            "회사는 보험금을 지급합니다.",
        ]
    )
    pages = [_page(doc_id, 1, body)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    parts = [s for s in sections if s.section_type == "part"]
    assert any("제2관" in p.title and "보험금" in p.title for p in parts)


def test_part_heading_closes_article_series_compact_gwan() -> None:
    doc_id = "doc_gwan_splits_articles"
    pad = "약관 본문입니다. 회사는 피보험자를 보호하기 위해 노력합니다." * 12
    body = "\n".join(
        [
            "제1관 목적 및 용어의 정의",
            pad,
            "제35조 (배당금의지급)",
            "이 계약은 배당금 지급에 관한 사항을 규정합니다." * 2,
            "제7관분쟁의조정등",
            "제36조 (분쟁의 조정)",
            "회사는 분쟁 조정 절차를 안내합니다." * 2,
            "제37조 (관할법원)",
            "계약자는 관할 법원에 소를 제기할 수 있습니다." * 2,
            "제38조 (소멸시효)",
            "보험금 청구권은 소멸시효가 적용됩니다." * 2,
        ]
    )
    pages = [_page(doc_id, 1, body)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    arts = [s for s in sections if s.section_type == "article"]
    titles = [a.title for a in arts]
    assert any("제35조" in t for t in titles)
    assert any("제36조" in t for t in titles)
    assert any("제37조" in t for t in titles)
    assert any("제38조" in t for t in titles)

    body35 = next(s.text for s in arts if "제35조" in s.title)
    assert "제36조" not in body35
    assert "제37조" not in body35
    assert "제38조" not in body35


def test_paren_byeolpyo3_and_plain_compact_appendix_sections_split() -> None:
    doc_id = "doc_byeolpyo_split"
    filler2 = "재해 분류 표 및 세부 기준 설명입니다." * 20
    filler3 = "적립 이율 산출 방식과 세부 예시입니다." * 20
    body = "\n".join(
        [
            "( 별표 2 ) 재해분류표",
            filler2,
            "( 별표3 ) 보험금을 지급할 때의 적립이율 계산",
            filler3,
            "별표4보험금지급기준표",
            "별표 세부 배열입니다." * 20,
        ]
    )
    pages = [_page(doc_id, 1, body)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    apx = [s for s in sections if s.section_type == "appendix"]
    titles = [a.title for a in apx]
    assert any("별표 2" in t or "별표2" in t for t in titles)
    assert any("별표 3" in t or "별표3" in t for t in titles)
    assert any("별표 4" in t or "별표4" in t for t in titles)
    assert len(apx) >= 3


def test_appendix_region_suppresses_table_style_article_pins() -> None:
    doc_id = "doc_appendix_article_pins"
    filler = "표 내용 및 금액 산출 예시입니다." * 25
    body = "\n".join(
        [
            "( 별표 3 ) 보험금 지급기준표",
            filler,
            "제22조(계약의소멸)",
            "셀 참조 주석입니다.",
            "제33조제1항",
            "별도 각주입니다.",
            "제7조제2항",
            "조항 인용 표기입니다.",
        ]
    )
    pages = [_page(doc_id, 1, body)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    sections = detect_sections(doc)
    arts = [s for s in sections if s.section_type == "article"]
    assert arts == []


def test_toc_region_override_emits_part_and_closes_article() -> None:
    """Misclassified TOC pages still emit real 관/조 headings when lines are not TOC-row shaped."""
    doc_id = "doc_toc_override_structure"
    body = "\n".join(
        [
            "제35조 (배당금의지급)",
            "이 계약은 배당금 규정입니다." * 6,
            "제7관 분쟁의 조정 등",
            "제36조 (분쟁의 조정)",
            "회사는 분쟁 조정 절차를 따릅니다." * 6,
        ]
    )
    pages = [_page(doc_id, 1, body)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    cands = collect_section_candidates(doc)
    regions = [
        PageRegion(
            document_id=doc_id,
            page_number=1,
            region_type="toc",
            confidence=0.9,
            evidence=["test:forced_toc"],
        ),
    ]
    sections = assemble_document_sections(document=doc, candidates=cands, page_regions=regions)
    titles = [s.title for s in sections]
    assert any("제7관" in t for t in titles)
    assert any("제36조" in t for t in titles)
    sec35 = next(s for s in sections if s.section_type == "article" and "제35조" in s.title)
    assert "제7관" not in sec35.text
    assert "제36조" not in sec35.text


def test_appendix_substantive_gate_skips_suppressed_table_article_pins() -> None:
    """Appendix body gate ignores appendix-region 조 pins that assembly drops later."""
    doc_id = "doc_appendix_gate_skip_pins"
    filler = "적립이율 및 지급 이자에 관한 상세 설명입니다." * 20
    body = "\n".join(
        [
            "( 별표 3 )",
            "(제7조제2항 관련)",
            filler,
            "제22조 (계약의",
            "소멸) 표기",
            "( 별표 4 ) 기타",
            filler,
        ]
    )
    pages = [_page(doc_id, 1, body)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    cands = collect_section_candidates(doc)
    regions = [
        PageRegion(
            document_id=doc_id,
            page_number=1,
            region_type="appendix",
            confidence=0.9,
            evidence=["test:forced_appendix"],
        ),
    ]
    sections = assemble_document_sections(document=doc, candidates=cands, page_regions=regions)
    apx = [s for s in sections if s.section_type == "appendix"]
    assert len(apx) >= 2
    apx3 = next(
        s
        for s in apx
        if ("별표 3" in s.title or "별표3" in s.title.replace(" ", "")) and "별표 4" not in s.title
    )
    assert "( 별표 4 )" not in apx3.text
    assert len(apx3.text) > 400


def test_same_page_candidates_emitted_in_char_offset_order() -> None:
    doc_id = "doc_same_page_offsets"
    body = "\n".join(
        [
            "제1조 (목적)",
            "이 계약은 목적입니다." * 8,
            "제2조 (정의)",
            "용어 정의 본문입니다." * 8,
        ]
    )
    pages = [_page(doc_id, 1, body)]
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(body),
        created_at=datetime.now(UTC),
    )
    cands = collect_section_candidates(doc)
    regions = [
        PageRegion(
            document_id=doc_id,
            page_number=1,
            region_type="toc",
            confidence=0.9,
            evidence=["test:forced_toc"],
        ),
    ]
    sections = assemble_document_sections(document=doc, candidates=cands, page_regions=regions)
    arts = [s for s in sections if s.section_type == "article"]
    offs = [s.start_char_offset for s in arts]
    assert offs == sorted(offs)
    assert len(arts) == 2


def test_build_sections_artifact_preserves_document_created_at() -> None:
    doc_id = "doc_artifact"
    pages = [_page(doc_id, 1, "제1조 (목적)\n내용")]
    created = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    doc = Document(
        document_id=doc_id,
        metadata=_meta(doc_id),
        pages=pages,
        page_count=1,
        total_char_count=len(pages[0].text),
        created_at=created,
    )
    art = build_sections_artifact(document=doc)
    assert art.document_id == doc_id
    assert art.source_document_created_at == created
    assert len(art.sections) >= 1
