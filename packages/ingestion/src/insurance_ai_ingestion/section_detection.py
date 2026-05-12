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


_TOC_LINE = re.compile(r"^[\s\[［【（(]*목\s*차[\s\]］】）)]*$")
_APPENDIX_PAREN = re.compile(r"^\s*\(\s*별표\s*(\d+)\s*\)\s*(.*)$")
_APPENDIX_PLAIN = re.compile(r"^\s*별표\s*(\d+)\s+(.+)$")
_PART = re.compile(r"^\s*(제\s*\d+\s*관)\s+(.+)$")
_ARTICLE = re.compile(r"^\s*(제\s*\d+\s*조)\s*(.*)$")


def _strip_heading_title(raw: str) -> str:
    t = raw.strip()
    return t if len(t) <= 500 else t[:497] + "..."


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
        rest = (m_ap.group(2) or "").strip()
        title = f"( 별표 {m_ap.group(1)} ) {rest}".strip()
        return (
            "appendix",
            _strip_heading_title(title if title else f"별표 {m_ap.group(1)}"),
            "pattern:appendix_paren",
        )

    m_ap2 = _APPENDIX_PLAIN.match(line)
    if m_ap2:
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

    if "약관에서 인용한" in s and "규정" in s:
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


def _should_emit_candidate(
    candidate: SectionCandidate,
    region: PageRegion | None,
) -> tuple[bool, list[str]]:
    """Drop noisy structure headings on cover/toc/guide/summary pages."""
    trace: list[str] = [f"candidate_rule_confidence:{candidate.confidence:.4f}"]
    if region is not None:
        trace.append(f"page_region:{region.region_type}")
        trace.append(f"page_region_confidence:{region.confidence:.4f}")
        trace.extend(f"region_evidence:{e}" for e in region.evidence)

    if candidate.section_type not in ("part", "article"):
        trace.append("filter:structure_heading_not_suppressed")
        return True, trace

    if region is None:
        trace.append("filter:missing_region_default_keep")
        return True, trace

    if region.region_type in _SUPPRESS_STRUCTURE_IN_REGIONS:
        trace.append(f"filter:suppress_structure_in_region:{region.region_type}")
        return False, trace

    trace.append("filter:structure_heading_keep")
    return True, trace


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
    for c in candidates:
        region = by_page.get(c.start_page)
        emit, trace = _should_emit_candidate(c, region)
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
