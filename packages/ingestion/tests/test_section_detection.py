from __future__ import annotations

from datetime import UTC, date, datetime

from insurance_ai_ingestion.section_detection import build_sections_artifact, detect_sections
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
    doc_id = "doc_appendix_sample"
    pages = [
        _page(
            doc_id,
            1,
            "\n".join(
                [
                    "( 별표 1 ) 보험금 지급기준표",
                    "내용",
                    "별표 2 특별약관",
                    "내용2",
                    "약관에서 인용한 법·규정",
                    "법령 목록",
                    "보험용어 해설",
                    "용어 설명",
                    "고객권리안내문",
                    "권리 안내",
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
    assert "guide" in kinds


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
