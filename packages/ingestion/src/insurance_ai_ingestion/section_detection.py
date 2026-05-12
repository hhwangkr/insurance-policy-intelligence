from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from insurance_ai_shared.models.document import Document
from insurance_ai_shared.models.section import (
    DocumentSection,
    DocumentSectionsArtifact,
    SectionType,
)


@dataclass(frozen=True)
class _Marker:
    """Internal heading marker before IDs/parents are assigned."""

    section_type: SectionType
    title: str
    start_char: int
    start_page: int


_TOC_LINE = re.compile(r"^[\s\[［【（(]*목\s*차[\s\]］】）)]*$")
_APPENDIX_PAREN = re.compile(r"^\s*\(\s*별표\s*(\d+)\s*\)\s*(.*)$")
_APPENDIX_PLAIN = re.compile(r"^\s*별표\s*(\d+)\s+(.+)$")
_PART = re.compile(r"^\s*(제\s*\d+\s*관)\s+(.+)$")
_ARTICLE = re.compile(r"^\s*(제\s*\d+\s*조)\s*(.*)$")


def _strip_heading_title(raw: str) -> str:
    t = raw.strip()
    return t if len(t) <= 500 else t[:497] + "..."


def _classify_line(line: str) -> tuple[SectionType, str] | None:
    """Return section type and display title for the first matching heading on this line."""
    s = line.strip()
    if not s:
        return None

    if _TOC_LINE.match(s) or s in {"목차", "목 차"}:
        return "toc", _strip_heading_title(s)

    if s.startswith("고객권리안내문"):
        return "guide", _strip_heading_title(s)

    m_ap = _APPENDIX_PAREN.match(line)
    if m_ap:
        rest = (m_ap.group(2) or "").strip()
        title = f"( 별표 {m_ap.group(1)} ) {rest}".strip()
        return "appendix", _strip_heading_title(title if title else f"별표 {m_ap.group(1)}")

    m_ap2 = _APPENDIX_PLAIN.match(line)
    if m_ap2:
        rest = (m_ap2.group(2) or "").strip()
        title = f"별표 {m_ap2.group(1)} {rest}".strip()
        return "appendix", _strip_heading_title(title)

    m_part = _PART.match(line)
    if m_part:
        title = f"{m_part.group(1).replace(' ', '')} {m_part.group(2)}".strip()
        title = re.sub(r"\s+", " ", title)
        return "part", _strip_heading_title(title)

    m_art = _ARTICLE.match(line)
    if m_art:
        body = (m_art.group(2) or "").strip()
        title = f"{m_art.group(1).replace(' ', '')} {body}".strip()
        title = re.sub(r"\s+", " ", title)
        return "article", _strip_heading_title(title)

    if "약관에서 인용한" in s and "규정" in s:
        return "legal_reference", _strip_heading_title(s)

    if s.startswith("보험용어 해설"):
        return "glossary", _strip_heading_title(s)

    return None


def _split_page_lines(text: str, page_base_offset: int) -> list[tuple[int, str]]:
    """Return (global_start_index, line_without_newline) for each line in page text."""
    if not text:
        return []
    lines: list[tuple[int, str]] = []
    start = 0
    while True:
        nl = text.find("\n", start)
        if nl == -1:
            lines.append((page_base_offset + start, text[start:]))
            break
        lines.append((page_base_offset + start, text[start:nl]))
        start = nl + 1
    return lines


def _collect_markers(document: Document) -> list[_Marker]:
    pages = sorted(document.pages, key=lambda p: p.page_number)
    texts = [p.text for p in pages]
    page_numbers = [p.page_number for p in pages]
    if not pages:
        return []

    starts: list[int] = []
    pos = 0
    for t in texts:
        starts.append(pos)
        pos += len(t) + 1

    markers: list[_Marker] = []
    for idx, page in enumerate(pages):
        base = starts[idx]
        for line_start, line in _split_page_lines(page.text, base):
            hit = _classify_line(line)
            if hit is None:
                continue
            kind, title = hit
            # Heading anchor: first non-whitespace char of the logical line
            stripped = line.lstrip()
            lead = len(line) - len(stripped)
            anchor = line_start + lead
            markers.append(
                _Marker(
                    section_type=kind,
                    title=title,
                    start_char=anchor,
                    start_page=page_numbers[idx],
                )
            )

    markers.sort(key=lambda m: (m.start_char, m.start_page, m.title, m.section_type))
    deduped: list[_Marker] = []
    for m in markers:
        if deduped and deduped[-1].start_char == m.start_char:
            continue
        deduped.append(m)
    return deduped


def _page_for_offset(
    offset: int,
    *,
    page_starts: list[int],
    page_numbers: list[int],
    total_len: int,
) -> int:
    if total_len <= 0:
        return page_numbers[0] if page_numbers else 1
    if offset < 0:
        offset = 0
    if offset >= total_len:
        offset = total_len - 1
    for i in range(len(page_starts) - 1, -1, -1):
        if page_starts[i] <= offset:
            return page_numbers[i]
    return page_numbers[0]


def detect_sections(document: Document) -> list[DocumentSection]:
    """Deterministic rule-based section boundaries for Korean policy PDF text."""
    pages = sorted(document.pages, key=lambda p: p.page_number)
    texts = [p.text for p in pages]
    page_numbers = [p.page_number for p in pages]
    full_text = "\n".join(texts)
    total_len = len(full_text)

    markers = _collect_markers(document)
    if not markers:
        return []

    starts: list[int] = []
    pos = 0
    for t in texts:
        starts.append(pos)
        pos += len(t) + 1

    last_part_id: str | None = None
    sections: list[DocumentSection] = []

    for i, m in enumerate(markers):
        section_id = f"{document.document_id}::sec::{i:04d}"
        next_start = markers[i + 1].start_char if i + 1 < len(markers) else total_len
        body = full_text[m.start_char : next_start]

        parent_id: str | None = None
        if m.section_type == "part":
            last_part_id = section_id
        elif m.section_type == "article":
            parent_id = last_part_id

        end_char = next_start
        end_page = _page_for_offset(
            end_char - 1 if end_char > m.start_char else m.start_char,
            page_starts=starts,
            page_numbers=page_numbers,
            total_len=total_len,
        )

        sections.append(
            DocumentSection(
                document_id=document.document_id,
                section_id=section_id,
                section_type=m.section_type,
                title=m.title,
                start_page=m.start_page,
                end_page=max(m.start_page, end_page),
                start_char_offset=m.start_char,
                end_char_offset=end_char,
                parent_section_id=parent_id,
                text=body,
            )
        )

    return sections


def build_sections_artifact(
    *,
    document: Document,
    created_at: datetime | None = None,
) -> DocumentSectionsArtifact:
    stamp = created_at if created_at is not None else datetime.now(UTC)
    return DocumentSectionsArtifact(
        document_id=document.document_id,
        source_document_created_at=document.created_at,
        sections=detect_sections(document),
        created_at=stamp,
    )
