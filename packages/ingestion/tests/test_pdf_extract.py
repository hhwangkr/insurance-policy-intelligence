from __future__ import annotations

from pathlib import Path

import fitz

from insurance_ai_ingestion.pdf_extract import extract_document_pages


def _write_single_page_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), text)
        doc.save(path)
    finally:
        doc.close()


def test_extract_document_pages_one_based_and_text(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    _write_single_page_pdf(pdf, "Hello\nWorld")
    pages = extract_document_pages(pdf_path=pdf, document_id="doc_a")
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].document_id == "doc_a"
    assert "Hello" in pages[0].text
    assert pages[0].char_count == len(pages[0].text)
    assert pages[0].extraction_method == "pymupdf"


def test_extract_document_pages_multiple_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "multi.pdf"
    doc = fitz.open()
    try:
        for label in ("AAA", "BBB"):
            page = doc.new_page()
            page.insert_text((72, 72), label)
        doc.save(pdf)
    finally:
        doc.close()

    pages = extract_document_pages(pdf_path=pdf, document_id="doc_b")
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert "AAA" in pages[0].text
    assert "BBB" in pages[1].text
