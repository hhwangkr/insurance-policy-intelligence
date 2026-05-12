from __future__ import annotations

import re
from datetime import UTC, datetime

from insurance_ai_ingestion.region_classification import (
    HeuristicRegionClassifier,
    RegionClassifier,
)
from insurance_ai_shared.models.document import Document
from insurance_ai_shared.models.section import (
    DocumentSection,
    DocumentSectionsArtifact,
    PageRegion,
    RegionType,
    SectionCandidate,
    SectionType,
)

_RULE_MATCH_CONFIDENCE = 0.82

_SUPPRESS_STRUCTURE_IN_REGIONS: frozenset[RegionType] = frozenset(
    {"toc", "guide", "cover", "summary"},
)

_STRUCTURE_SUPPRESS_TYPES: frozenset[SectionType] = frozenset(
    {"article", "part", "appendix", "legal_reference", "glossary"},
)

_MIN_APPENDIX_BODY_CHARS = 100
_MIN_TOCLIKE_TAIL_SUBSTANTIVE_CHARS = 35

_LEGAL_CORPUS_HINTS: tuple[str, ...] = (
    "민법",
    "상법",
    "보험업법",
    "금융소비자보호법",
    "조세특례제한법",
    "소비자기본법",
    "개인정보 보호법",
    "전자금융거래법",
    "보험업법 시행령",
    "약관의 규제에 관한 법률",
)

_LEGAL_GUIDE_BLOCK_PHRASES: tuple[str, ...] = (
    "보험약관 가이드",
    "보험약관 요약서",
    "약관을 쉽게 이용",
    "관련법규 168p",
)

_TOC_LINE = re.compile(r"^[\s\[［【（(]*목\s*차[\s\]］】）)]*$")
_APPENDIX_PAREN = re.compile(r"^\s*[\(（]\s*별표\s*(\d+)\s*[\)）]\s*(.*)$")
_APPENDIX_PLAIN = re.compile(r"^\s*별표\s*(\d+)\s*(.+)$")
_PART = re.compile(r"^\s*(제\s*\d+\s*관)\s*(.+)$")
_ARTICLE = re.compile(r"^\s*(제\s*\d+\s*조)\s*(.*)$")

_ARTICLE_APPENDIX_PROSE_OPENERS: tuple[str, ...] = (
    "이 계약은",
    "회사는",
    "계약자는",
    "보험수익자는",
    "피보험자는",
)

_PAGE_POINTER_TAIL = re.compile(
    r"(?:[\.\．]\s*\d{1,4}\s*|\d{1,4}\s*p\s*)$",
    re.IGNORECASE,
)
_LEGAL_GUIDE_POINTER = re.compile(
    r"^\s*관련법규\s*\d+\s*p?\s*$",
    re.IGNORECASE,
)


def _strip_heading_title(raw: str) -> str:
    t = raw.strip()
    return t if len(t) <= 500 else t[:497] + "..."


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _article_suppressed_in_appendix_region(
    candidate: SectionCandidate,
    full_text: str,
    end_exclusive: int,
) -> tuple[bool, str]:
    """Drop table-style 조 pins inside appendix/table pages (not policy-body articles)."""
    if candidate.section_type != "article":
        return False, ""
    chunk = full_text[candidate.start_char_offset : end_exclusive]
    lines = chunk.splitlines()
    if not lines:
        return False, ""
    head = lines[0].strip()
    body = "\n".join(lines[1:]).strip()
    early_body = body[:1400]
    compact_head = _collapse_ws(head)

    if re.search(r"제\s*\d+\s*조\s*제\s*\d+\s*항", compact_head):
        return True, "filter:appendix_article_clause_pin_reference"

    if "(" in head or "（" in head:
        if ")" not in head and "）" not in head:
            return True, "filter:appendix_article_incomplete_paren_heading"

    if re.match(r"^\s*제\s*\d+\s*조[（(]", head):
        if re.search(r"[）)]", head) and len(head) <= 78:
            return True, "filter:appendix_article_compact_paren_row"

    if any(op in early_body for op in _ARTICLE_APPENDIX_PROSE_OPENERS):
        return False, ""

    if len(body) >= 220:
        return False, ""

    return True, "filter:appendix_article_missing_clause_prose"


def _legal_heading_line(s: str) -> bool:
    """True only for standalone legal-index headings (not guide pointers like '관련법규 168p')."""
    t = s.strip()
    if not t:
        return False
    if _LEGAL_GUIDE_POINTER.match(t):
        return False
    if re.match(r"^\s*관련법규\s*\d", t):
        return False
    compact = re.sub(r"\s+", "", t)
    if re.match(r"^약관에서인용(?:한|된)법", compact):
        return True
    if re.match(r"^\s*약관에서\s*인용(?:한|된)\s*법", t):
        return True
    if re.match(r"^\s*관련\s*법규\s*$", t) or re.match(r"^\s*관련\s*법규\s*[:\：]", t):
        return True
    return False


def _appendix_paren_line_is_page_pointer(line: str) -> bool:
    """TOC-style appendix row: heading ends with dot/page number on the same line."""
    s = line.strip()
    if not _APPENDIX_PAREN.match(line):
        return False
    if len(s) <= 120 and _PAGE_POINTER_TAIL.search(s):
        return True
    return False


def _appendix_plain_line_is_page_pointer(line: str) -> bool:
    s = line.strip()
    if not _APPENDIX_PLAIN.match(line):
        return False
    if len(s) <= 120 and _PAGE_POINTER_TAIL.search(s):
        return True
    return False


def _tail_lines_are_only_page_references(tail: str) -> bool:
    """True when every non-empty line is a lone page pointer (TOC row continuation)."""
    if not tail.strip():
        return False
    for ln in tail.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"[\.\．]\s*\d{1,4}", s):
            continue
        if re.fullmatch(r"\d{1,4}\s*p?", s, flags=re.IGNORECASE):
            continue
        return False
    return True


def _non_pointer_tail_chars(text: str) -> int:
    """Rough count of substantive characters after dropping page-number-only lines."""
    kept: list[str] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if re.fullmatch(r"[\.\．]\s*\d{1,4}", ln):
            continue
        if re.fullmatch(r"\d{1,4}\s*p?", ln, flags=re.IGNORECASE):
            continue
        kept.append(ln)
    return sum(len(x) for x in kept)


def _appendix_slice_has_substantive_body(full_text: str, start: int, end_exclusive: int) -> bool:
    chunk = full_text[start:end_exclusive]
    lines = chunk.splitlines()
    if not lines:
        return False
    tail = "\n".join(lines[1:]).strip()
    if _non_pointer_tail_chars(tail) >= _MIN_APPENDIX_BODY_CHARS:
        return True
    return False


def _legal_chunk_has_substantive_law_content(chunk: str) -> bool:
    """True when text after the heading looks like law citations, not guide/TOC filler."""
    lines = chunk.splitlines()
    if len(lines) < 2:
        return False
    body = "\n".join(lines[1:]).strip()
    if len(body) < 40:
        return False
    if any(hint in body for hint in _LEGAL_CORPUS_HINTS):
        return True
    if re.search(r"[「【][^」』\n]{2,80}[」』]", body):
        return True
    if re.search(
        r"제\s*\d+\s*조\s*[（(]?\s*(?:민법|상법|보험업법|금융소비자보호법)",
        body,
    ):
        return True
    if re.search(r"(?:법률|법령|시행령)\s*(?:제\s*\d+\s*조|제\s*\d+\s*호)", body):
        return True
    return False


def _legal_reference_slice_emit_allowed(
    full_text: str, start: int, end_exclusive: int
) -> tuple[bool, str]:
    """Suppress legal_reference slices that are TOC pointers or guide front matter."""
    chunk = full_text[start:end_exclusive]
    if not chunk.strip():
        return False, "filter:legal_blocked_empty_chunk"

    early = chunk[:2200]
    for phrase in _LEGAL_GUIDE_BLOCK_PHRASES:
        if phrase in early:
            return False, f"filter:legal_blocked_guide_phrase:{phrase}"

    compact_early = re.sub(r"\s+", "", chunk[:1100])
    if "약관에서인용된법령" in compact_early or "약관에서인용한법" in compact_early:
        if re.search(r"[\.\．]\s*168|168\s*p", chunk[:520], flags=re.IGNORECASE):
            if "보험용어해설" in compact_early or "보험용어 해설" in chunk[:900]:
                return False, "filter:legal_blocked_pointer_glossary_toc_ladder"

    head_line = chunk.splitlines()[0].strip()
    compact_head = re.sub(r"\s+", "", head_line)
    if re.search(r"(?:법·규정|법령)\s*\d{1,4}\s*p?\s*$", head_line, flags=re.IGNORECASE):
        return False, "filter:legal_blocked_heading_same_line_page_pointer"
    if re.search(r"(?:법·규정|법령)\s*\d{1,4}\s*p?\s*$", compact_head, flags=re.IGNORECASE):
        return False, "filter:legal_blocked_heading_compact_page_pointer"

    if not _legal_chunk_has_substantive_law_content(chunk):
        return False, "filter:legal_blocked_missing_substantive_law_corpus"

    return True, ""


def _classify_line(line: str) -> tuple[SectionType, str, str] | None:
    """Return section type, display title, and evidence token for the matched rule."""
    s = line.strip()
    if not s:
        return None

    if _TOC_LINE.match(s) or s in {"목차", "목 차"}:
        return "toc", _strip_heading_title(s), "pattern:toc_heading"

    if s.startswith("고객권리안내문"):
        return "guide", _strip_heading_title(s), "pattern:customer_rights_heading"

    m_ap = _APPENDIX_PAREN.match(line)
    if m_ap:
        if _appendix_paren_line_is_page_pointer(line):
            return None
        rest = (m_ap.group(2) or "").strip()
        title = f"( 별표 {m_ap.group(1)} ) {rest}".strip()
        return (
            "appendix",
            _strip_heading_title(title if title else f"별표 {m_ap.group(1)}"),
            "pattern:appendix_paren",
        )

    m_ap2 = _APPENDIX_PLAIN.match(line)
    if m_ap2:
        if _appendix_plain_line_is_page_pointer(line):
            return None
        rest = (m_ap2.group(2) or "").strip()
        title = f"별표 {m_ap2.group(1)} {rest}".strip()
        return "appendix", _strip_heading_title(title), "pattern:appendix_plain"

    m_part = _PART.match(line)
    if m_part:
        title = f"{m_part.group(1).replace(' ', '')} {m_part.group(2)}".strip()
        title = re.sub(r"\s+", " ", title)
        return "part", _strip_heading_title(title), "pattern:gwan_line"

    m_art = _ARTICLE.match(line)
    if m_art:
        body = (m_art.group(2) or "").strip()
        title = f"{m_art.group(1).replace(' ', '')} {body}".strip()
        title = re.sub(r"\s+", " ", title)
        return "article", _strip_heading_title(title), "pattern:article_line"

    if _legal_heading_line(s):
        return "legal_reference", _strip_heading_title(s), "pattern:legal_reference_heading"

    if s.startswith("보험용어 해설"):
        return "glossary", _strip_heading_title(s), "pattern:glossary_heading"

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


def collect_section_candidates(document: Document) -> list[SectionCandidate]:
    """Deterministic regex scan: all heading candidates with stable IDs (Phase 2D core)."""
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

    candidates: list[SectionCandidate] = []
    for idx, page in enumerate(pages):
        base = starts[idx]
        for line_start, line in _split_page_lines(page.text, base):
            hit = _classify_line(line)
            if hit is None:
                continue
            kind, title, ev = hit
            stripped = line.lstrip()
            lead = len(line) - len(stripped)
            anchor = line_start + lead
            cid = f"{document.document_id}::cand::{anchor:010d}"
            candidates.append(
                SectionCandidate(
                    candidate_id=cid,
                    document_id=document.document_id,
                    section_type=kind,
                    title=title,
                    start_page=page_numbers[idx],
                    start_char_offset=anchor,
                    confidence=_RULE_MATCH_CONFIDENCE,
                    evidence=[ev, "generator:regex_rules_v1"],
                )
            )

    candidates.sort(
        key=lambda c: (c.start_char_offset, c.start_page, c.title, c.section_type),
    )
    deduped: list[SectionCandidate] = []
    for c in candidates:
        if deduped and deduped[-1].start_char_offset == c.start_char_offset:
            continue
        deduped.append(c)
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


def _regions_by_page(regions: list[PageRegion]) -> dict[int, PageRegion]:
    return {r.page_number: r for r in regions}


def _candidate_is_toc_like_reference_row(
    candidate: SectionCandidate,
    full_text: str,
    end_exclusive: int,
) -> bool:
    """TOC row / pointer backstop: suppress regardless of page_region when slice is pointer-only."""
    chunk = full_text[candidate.start_char_offset : end_exclusive]
    lines = chunk.splitlines()
    if not lines:
        return False
    head = lines[0].strip()
    tail = "\n".join(lines[1:]).strip()

    if candidate.section_type == "article":
        if not re.match(r"^\s*제\s*\d+\s*조", head):
            return False
        if _non_pointer_tail_chars(tail) >= _MIN_TOCLIKE_TAIL_SUBSTANTIVE_CHARS:
            return False
        return _tail_lines_are_only_page_references(tail)

    if candidate.section_type == "appendix":
        if not (_APPENDIX_PAREN.match(lines[0]) or _APPENDIX_PLAIN.match(lines[0])):
            return False
        if _non_pointer_tail_chars(tail) >= _MIN_TOCLIKE_TAIL_SUBSTANTIVE_CHARS:
            return False
        return _tail_lines_are_only_page_references(tail)

    if candidate.section_type == "legal_reference":
        compact_head = re.sub(r"\s+", "", head)
        legal_ok = _legal_heading_line(head) or bool(
            re.match(r"^약관에서인용(?:한|된)법", compact_head),
        )
        if not legal_ok:
            return False
        if _non_pointer_tail_chars(tail) >= _MIN_TOCLIKE_TAIL_SUBSTANTIVE_CHARS:
            return False
        return _tail_lines_are_only_page_references(tail)

    return False


def _explicit_front_matter_heading(candidate: SectionCandidate) -> bool:
    """Keep 목차 / 고객권리안내문 anchors on front-matter pages."""
    ev = candidate.evidence
    return any(x.startswith("pattern:toc_heading") for x in ev) or any(
        x.startswith("pattern:customer_rights_heading") for x in ev
    )


def _should_emit_candidate(
    candidate: SectionCandidate,
    region: PageRegion | None,
) -> tuple[bool, list[str]]:
    """Suppress structural headings on cover/toc/guide/summary; keep coarse toc/guide anchors."""
    trace: list[str] = [f"candidate_rule_confidence:{candidate.confidence:.4f}"]
    if region is not None:
        trace.append(f"page_region:{region.region_type}")
        trace.append(f"page_region_confidence:{region.confidence:.4f}")
        trace.extend(f"region_evidence:{e}" for e in region.evidence)

    if region is None:
        trace.append("filter:missing_region_default_keep")
        return True, trace

    if region.region_type not in _SUPPRESS_STRUCTURE_IN_REGIONS:
        trace.append("filter:not_front_matter_region_keep")
        return True, trace

    if candidate.section_type in _STRUCTURE_SUPPRESS_TYPES:
        trace.append(f"filter:suppress_structure_on_front_matter:{region.region_type}")
        return False, trace

    if candidate.section_type in ("toc", "guide") and _explicit_front_matter_heading(candidate):
        trace.append("filter:keep_explicit_toc_guide_heading")
        return True, trace

    trace.append("filter:suppress_non_structural_on_front_matter")
    return False, trace


def _assembly_confidence(
    candidate: SectionCandidate, region: PageRegion | None, *, emit: bool
) -> float:
    if not emit:
        return max(0.0, candidate.confidence - 0.5)
    base = candidate.confidence
    if candidate.section_type in ("part", "article") and region is not None:
        if region.region_type == "policy_body":
            return min(1.0, base + 0.12)
        if region.region_type == "unknown":
            return min(1.0, base + 0.02)
    return base


def assemble_document_sections(
    *,
    document: Document,
    candidates: list[SectionCandidate],
    page_regions: list[PageRegion],
) -> list[DocumentSection]:
    """Slice final sections strictly from extracted text (no LLM-generated body text)."""
    pages = sorted(document.pages, key=lambda p: p.page_number)
    texts = [p.text for p in pages]
    page_numbers = [p.page_number for p in pages]
    full_text = "\n".join(texts)
    total_len = len(full_text)

    starts: list[int] = []
    pos = 0
    for t in texts:
        starts.append(pos)
        pos += len(t) + 1

    by_page = _regions_by_page(page_regions)

    accepted: list[tuple[SectionCandidate, list[str], float]] = []
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c.start_char_offset, c.start_page, c.title, c.section_type),
    )
    next_offset_by_index: dict[int, int] = {}
    for i, _c in enumerate(sorted_cands):
        next_offset_by_index[i] = (
            sorted_cands[i + 1].start_char_offset if i + 1 < len(sorted_cands) else total_len
        )

    for i, c in enumerate(sorted_cands):
        region = by_page.get(c.start_page)
        next_start = next_offset_by_index[i]

        emit: bool
        trace: list[str]
        if _candidate_is_toc_like_reference_row(c, full_text, next_start):
            trace = [
                f"candidate_rule_confidence:{c.confidence:.4f}",
                "filter:toc_like_candidate_backstop",
            ]
            if region is not None:
                trace.append(f"page_region:{region.region_type}")
                trace.append(f"page_region_confidence:{region.confidence:.4f}")
                trace.extend(f"region_evidence:{e}" for e in region.evidence)
            emit = False
        else:
            emit, trace = _should_emit_candidate(c, region)

        if emit and c.section_type == "appendix":
            if not _appendix_slice_has_substantive_body(full_text, c.start_char_offset, next_start):
                trace = trace + ["filter:appendix_insufficient_substantive_body"]
                emit = False

        if emit and c.section_type == "legal_reference":
            legal_ok, legal_reason = _legal_reference_slice_emit_allowed(
                full_text,
                c.start_char_offset,
                next_start,
            )
            if not legal_ok:
                trace = trace + [legal_reason]
                emit = False

        if (
            emit
            and c.section_type == "article"
            and region is not None
            and region.region_type == "appendix"
        ):
            apx_art_sup, apx_art_reason = _article_suppressed_in_appendix_region(
                c,
                full_text,
                next_start,
            )
            if apx_art_sup:
                trace = trace + [apx_art_reason]
                emit = False

        conf = _assembly_confidence(c, region, emit=emit)
        if emit:
            accepted.append((c, trace, conf))

    if not accepted:
        return []

    last_part_id: str | None = None
    sections: list[DocumentSection] = []

    for i, (c, trace, asm_conf) in enumerate(accepted):
        section_id = f"{document.document_id}::sec::{i:04d}"
        next_start = accepted[i + 1][0].start_char_offset if i + 1 < len(accepted) else total_len
        body = full_text[c.start_char_offset : next_start]

        parent_id: str | None = None
        if c.section_type == "part":
            last_part_id = section_id
        elif c.section_type == "article":
            parent_id = last_part_id

        end_char = next_start
        end_page = _page_for_offset(
            end_char - 1 if end_char > c.start_char_offset else c.start_char_offset,
            page_starts=starts,
            page_numbers=page_numbers,
            total_len=total_len,
        )

        asm_evidence = trace + [
            "assembly:substring_slice",
            f"candidate_id:{c.candidate_id}",
        ]

        sections.append(
            DocumentSection(
                document_id=document.document_id,
                section_id=section_id,
                section_type=c.section_type,
                title=c.title,
                start_page=c.start_page,
                end_page=max(c.start_page, end_page),
                start_char_offset=c.start_char_offset,
                end_char_offset=end_char,
                parent_section_id=parent_id,
                text=body,
                assembly_confidence=min(1.0, max(0.0, asm_conf)),
                assembly_evidence=asm_evidence,
            )
        )

    return sections


def run_section_detection(
    document: Document,
    *,
    region_classifier: RegionClassifier | None = None,
) -> tuple[list[DocumentSection], list[SectionCandidate], list[PageRegion]]:
    """Hybrid pipeline foundation: regex candidates + region labels + deterministic assembly."""
    classifier = region_classifier or HeuristicRegionClassifier()
    candidates = collect_section_candidates(document)
    regions = classifier.classify_document(document)
    sections = assemble_document_sections(
        document=document,
        candidates=candidates,
        page_regions=regions,
    )
    return sections, candidates, regions


def detect_sections(document: Document) -> list[DocumentSection]:
    """Public API: final sections after rule candidates and heuristic region filtering."""
    sections, _, _ = run_section_detection(document)
    return sections


def build_sections_artifact(
    *,
    document: Document,
    created_at: datetime | None = None,
    region_classifier: RegionClassifier | None = None,
) -> DocumentSectionsArtifact:
    stamp = created_at if created_at is not None else datetime.now(UTC)
    sections, candidates, regions = run_section_detection(
        document,
        region_classifier=region_classifier,
    )
    return DocumentSectionsArtifact(
        document_id=document.document_id,
        source_document_created_at=document.created_at,
        sections=sections,
        page_regions=regions,
        section_candidates=candidates,
        created_at=stamp,
    )
