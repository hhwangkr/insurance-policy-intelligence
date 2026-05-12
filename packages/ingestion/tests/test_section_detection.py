from __future__ import annotations

from datetime import UTC, date, datetime

from insurance_ai_ingestion.section_detection import (
    build_sections_artifact,
    collect_section_candidates,
    detect_sections,
    run_section_detection,
)
from insurance_ai_shared.models.document import Document, DocumentMetadata, DocumentPage


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
