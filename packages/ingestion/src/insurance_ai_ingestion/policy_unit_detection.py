"""Deterministic product variant (policy unit) spans from integrated PDF markers."""

from __future__ import annotations

import re

from insurance_ai_ingestion.section_detection_patterns import (
    GLOBAL_TAIL_HEADING_RE,
    POLICY_VARIANT_GROUP,
    TAIL_MATTER_SUBSTRINGS,
)
from insurance_ai_ingestion.section_text_layout import (
    document_full_text_index,
    page_for_offset,
)
from insurance_ai_shared.models.document import Document
from insurance_ai_shared.models.section import DocumentSection, PolicyUnit


def iter_policy_unit_markers(full_text: str) -> list[tuple[int, str, str]]:
    """Return (start_char, policy_unit_name, variant) for each variant + 약관 marker block."""
    out: list[tuple[int, str, str]] = []
    pos = 0
    hi = len(full_text)
    while pos < hi:
        nl = full_text.find("\n", pos)
        line_end = nl if nl != -1 and nl < hi else hi
        stripped = full_text[pos:line_end].strip()
        if stripped.startswith("제") and re.match(r"^제\s*\d+\s*(?:조|관)\b", stripped):
            pos = line_end + 1 if nl != -1 else hi
            continue
        if re.match(r"^\s*[\(（]\s*별표", stripped):
            pos = line_end + 1 if nl != -1 else hi
            continue
        m_one = re.match(
            rf"^(.+)\(({POLICY_VARIANT_GROUP})\)\s*(?:약\s*관|약관)\s*$",
            stripped,
        )
        if m_one:
            out.append((pos, m_one.group(1).strip(), m_one.group(2)))
            pos = line_end + 1 if nl != -1 else hi
            continue
        m_two = re.match(rf"^(.+)\(({POLICY_VARIANT_GROUP})\)\s*$", stripped)
        if m_two and nl != -1 and nl + 1 < hi:
            nl2 = full_text.find("\n", nl + 1)
            line2_end = nl2 if nl2 != -1 and nl2 < hi else hi
            line2 = full_text[nl + 1 : line2_end].strip()
            if (
                re.match(r"^약\s*관\s*$", line2)
                or re.match(r"^약관\s*$", line2)
                or re.match(r"^제\s*1\s*관", line2)
            ):
                out.append((pos, m_two.group(1).strip(), m_two.group(2)))
        pos = line_end + 1 if nl != -1 else hi
    return out


def _dedupe_consecutive_same_variant_markers(
    hits: list[tuple[int, str, str]],
) -> list[tuple[int, str, str]]:
    """Drop repeated (name, variant) hits so marker page + body page become one unit."""
    out: list[tuple[int, str, str]] = []
    for off, name, var in hits:
        if out and out[-1][1] == name and out[-1][2] == var:
            continue
        out.append((off, name, var))
    return out


def policy_unit_marker_hits(full_text: str) -> list[tuple[int, str, str]]:
    return _dedupe_consecutive_same_variant_markers(iter_policy_unit_markers(full_text))


def first_global_tail_matter_offset(full_text: str, search_from: int) -> int | None:
    """Earliest tail-matter start at/after ``search_from`` (regex line + substring anchors)."""
    if search_from < 0:
        search_from = 0
    hi = len(full_text)
    if search_from >= hi:
        return None
    candidates: list[int] = []
    m = GLOBAL_TAIL_HEADING_RE.search(full_text, search_from)
    if m:
        candidates.append(m.start())
    for needle in TAIL_MATTER_SUBSTRINGS:
        j = full_text.find(needle, search_from)
        if j != -1:
            candidates.append(j)
    return min(candidates) if candidates else None


def global_policy_tail_cutoff(full_text: str, policy_units: list[PolicyUnit]) -> int | None:
    if not policy_units:
        return None
    return first_global_tail_matter_offset(full_text, policy_units[-1].start_char_offset)


def first_product_unit_boundary_offset(
    markers: list[tuple[int, str, str]],
    lo: int,
    hi: int,
) -> int | None:
    """First policy-unit marker offset in [lo, hi), or None."""
    for off, _, _ in markers:
        if lo <= off < hi:
            return off
    return None


def _trim_last_policy_unit_at_tail_headings(
    full_text: str,
    units: list[PolicyUnit],
    *,
    page_starts: list[int],
    page_numbers: list[int],
    total_len: int,
) -> None:
    """Shorten the last unit so shared legal/glossary tail is outside any variant span."""
    if not units:
        return
    last = units[-1]
    cut = first_global_tail_matter_offset(full_text, last.start_char_offset)
    if cut is None:
        return
    new_end = cut - 1
    if new_end < last.start_char_offset or new_end > total_len - 1:
        return
    units[-1] = last.model_copy(
        update={
            "end_char_offset": new_end,
            "end_page": page_for_offset(
                new_end,
                page_starts=page_starts,
                page_numbers=page_numbers,
                total_len=total_len,
            ),
        },
    )


def discover_policy_units(document: Document) -> list[PolicyUnit]:
    """Scan full extracted text for variant + 약관 markers; spans run until the next marker."""
    full_text, page_starts, page_numbers, total_len = document_full_text_index(document)
    hits = policy_unit_marker_hits(full_text)
    if not hits:
        return []
    out: list[PolicyUnit] = []
    for i, (off, name, var) in enumerate(hits):
        end_off = hits[i + 1][0] - 1 if i + 1 < len(hits) else total_len - 1
        end_off = max(off, min(end_off, total_len - 1))
        sid = f"{document.document_id}::pu::{i:04d}"
        out.append(
            PolicyUnit(
                policy_unit_id=sid,
                policy_unit_name=name,
                variant_name=var,
                start_page=page_for_offset(
                    off,
                    page_starts=page_starts,
                    page_numbers=page_numbers,
                    total_len=total_len,
                ),
                start_char_offset=off,
                end_page=page_for_offset(
                    end_off,
                    page_starts=page_starts,
                    page_numbers=page_numbers,
                    total_len=total_len,
                ),
                end_char_offset=end_off,
            )
        )
    _trim_last_policy_unit_at_tail_headings(
        full_text,
        out,
        page_starts=page_starts,
        page_numbers=page_numbers,
        total_len=total_len,
    )
    return out


def attach_policy_unit_metadata(
    sections: list[DocumentSection],
    policy_units: list[PolicyUnit],
    *,
    global_tail_cutoff: int | None = None,
) -> list[DocumentSection]:
    """Assign policy unit when section start lies in [unit.start, unit.end] (disjoint spans)."""
    if not policy_units:
        return sections
    out: list[DocumentSection] = []
    for sec in sections:
        if global_tail_cutoff is not None and sec.start_char_offset >= global_tail_cutoff:
            out.append(sec)
            continue
        chosen: PolicyUnit | None = None
        for u in policy_units:
            if u.start_char_offset <= sec.start_char_offset <= u.end_char_offset:
                chosen = u
                break
        if chosen is None:
            out.append(sec)
            continue
        out.append(
            sec.model_copy(
                update={
                    "policy_unit_id": chosen.policy_unit_id,
                    "policy_unit_name": chosen.policy_unit_name,
                    "variant_name": chosen.variant_name,
                },
            ),
        )
    return out
