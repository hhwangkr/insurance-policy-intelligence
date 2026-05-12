from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from insurance_ai_shared.models.document import Document
from insurance_ai_shared.models.section import PageRegion, RegionType


def _appendix_heading_line_count(text: str) -> tuple[int, list[str]]:
    """Count appendix/table heading lines (not inline 별표 mentions inside clauses)."""
    evidence: list[str] = []
    n = 0
    for line in text.splitlines():
        if re.match(r"^\s*[\(（]\s*별표\s*\d+\s*[\)）]", line):
            n += 1
            evidence.append("heading_line:byeolpyo_paren")
        elif re.match(r"^\s*별표\s+\d+\s+\S", line):
            n += 1
            evidence.append("heading_line:byeolpyo_plain")
        elif re.match(r"^\s*별표\s*\d+(?:\s+\S|\S)", line):
            n += 1
            evidence.append("heading_line:byeolpyo_plain_compact")
    return n, evidence


def _legal_heading_present(text: str) -> bool:
    """Boost legal_reference region only when a real legal-index heading appears."""
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^\s*약관에서\s*인용(?:한|된)\s*법", s):
            return True
        compact = re.sub(r"\s+", "", s)
        if re.match(r"^약관에서인용(?:한|된)법", compact):
            return True
        if re.match(r"^\s*관련\s*법규\s*$", s) or re.match(r"^\s*관련\s*법규\s*[:\：]", s):
            return True
    return False


def _article_outline_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*제\s*\d+\s*조\b", line.strip()))


def _toc_style_page_ref_row_count(text: str) -> int:
    """Lines that look like TOC rows: short heading + trailing page ref."""
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or len(s) > 140:
            continue
        if not re.search(r"(?:[\.\．]\s*\d{1,4}|\d{1,4}\s*p)\s*$", s, flags=re.IGNORECASE):
            continue
        if re.search(r"제\s*\d+\s*조", s) or re.search(r"별표", s) or "약관" in s:
            n += 1
    return n


def _substantive_policy_body_heuristic(stripped: str) -> bool:
    """True when the page likely contains real clause paragraphs, not only TOC rows."""
    lines = stripped.splitlines()
    long_substantive = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if re.match(r"^\s*제\s*\d+\s*조\b", s) and len(s) < 85:
            continue
        if re.fullmatch(r"[\.\．]\s*\d{1,4}", s) or re.fullmatch(
            r"\d{1,4}\s*p?", s, flags=re.IGNORECASE
        ):
            continue
        if len(s) >= 110:
            long_substantive += 1
    if long_substantive >= 2:
        return True
    if len(stripped) > 9500:
        return True
    if re.search(r"제\s*\d+\s*관\b", stripped) and len(stripped) > 4500:
        return True
    return False


def _apply_toc_priority_over_special_regions(
    scores: dict[RegionType, float],
    evidence: list[str],
    *,
    toc_article_lines: int,
    toc_page_ref_rows: int,
    stripped: str,
) -> None:
    """Prefer TOC over appendix/legal/glossary when outline-heavy without clause text."""
    substantive = _substantive_policy_body_heuristic(stripped)
    strong_outline = toc_article_lines >= 5
    strong_toc_rows = toc_page_ref_rows >= 6
    combined_toc_signal = strong_outline or (toc_article_lines >= 3 and toc_page_ref_rows >= 5)

    if substantive:
        return

    if strong_outline:
        scores["toc"] += min(40.0 + 4.5 * float(toc_article_lines), 220.0)
        evidence.append(f"signal:toc_outline_priority_boost:{toc_article_lines}")

    if combined_toc_signal or strong_toc_rows:
        scores["toc"] += 55.0
        evidence.append(
            f"signal:toc_reference_row_priority:{toc_page_ref_rows}:{toc_article_lines}"
        )
        scores["appendix"] = min(scores["appendix"], 12.0)
        scores["legal_reference"] = min(scores["legal_reference"], 12.0)
        scores["glossary"] = min(scores["glossary"], 12.0)


def _score_region(page_number: int, text: str) -> tuple[RegionType, float, list[str]]:
    """Deterministic page-level region label from shallow text signals."""
    evidence: list[str] = []
    raw = text or ""
    stripped = raw.strip()

    if not stripped:
        return "unknown", 0.35, ["empty_page"]

    scores: dict[RegionType, float] = {
        "cover": 0.0,
        "toc": 0.0,
        "guide": 0.0,
        "summary": 0.0,
        "policy_body": 0.0,
        "appendix": 0.0,
        "legal_reference": 0.0,
        "glossary": 0.0,
        "unknown": 0.0,
    }

    lowered = stripped.lower()

    if "고객권리안내문" in stripped or "guide book" in lowered or "약관 이용" in lowered:
        scores["guide"] += 85.0
        evidence.append("keyword:guide_or_customer_rights")

    if _legal_heading_present(stripped):
        scores["legal_reference"] += 80.0
        evidence.append("keyword:legal_reference_heading")

    if stripped.startswith("보험용어 해설") or ("보험용어" in stripped and "해설" in stripped):
        scores["glossary"] += 78.0
        evidence.append("keyword:glossary_heading")

    dot_leader_lines = sum(1 for line in stripped.splitlines() if re.search(r"\.{4,}", line))
    if dot_leader_lines >= 2:
        bump = min(60.0 + 8.0 * float(dot_leader_lines), 95.0)
        scores["toc"] += bump
        evidence.append(f"signal:dot_leader_lines:{dot_leader_lines}")

    if re.search(r"목\s*차", stripped):
        scores["toc"] += 55.0
        evidence.append("keyword:toc_marker")

    toc_article_lines = _article_outline_line_count(stripped)
    if toc_article_lines >= 4 and len(stripped) < 6000:
        scores["toc"] += 45.0
        evidence.append(f"signal:many_article_outline_lines:{toc_article_lines}")

    toc_page_ref_rows = _toc_style_page_ref_row_count(stripped)
    if toc_page_ref_rows >= 4:
        scores["toc"] += min(30.0 + 6.0 * float(toc_page_ref_rows), 95.0)
        evidence.append(f"signal:toc_page_reference_rows:{toc_page_ref_rows}")

    apx_heads, apx_evs = _appendix_heading_line_count(stripped)
    if apx_heads >= 1:
        scores["appendix"] += min(38.0 + 14.0 * float(apx_heads), 88.0)
        evidence.extend(apx_evs[:5])

    _apply_toc_priority_over_special_regions(
        scores,
        evidence,
        toc_article_lines=toc_article_lines,
        toc_page_ref_rows=toc_page_ref_rows,
        stripped=stripped,
    )

    if "요약" in stripped or "알아두세요" in stripped or "주요내용" in stripped:
        scores["summary"] += 48.0
        evidence.append("keyword:summary_marker")

    if (
        page_number <= 2
        and len(stripped) < 1600
        and ("보험약관" in stripped or "개정약관" in stripped or "공시" in stripped)
    ):
        scores["cover"] += 52.0
        evidence.append("heuristic:early_short_cover_like")

    if re.search(r"제\s*\d+\s*관\b", stripped):
        scores["policy_body"] += 62.0
        evidence.append("keyword:gwan_heading")

    if len(stripped) > 3200:
        scores["policy_body"] += 28.0
        evidence.append("heuristic:long_page")

    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]
    if best_score <= 0.0:
        return "unknown", 0.35, (evidence or []) + ["fallback:unknown"]

    sorted_scores = sorted(scores.values(), reverse=True)
    top = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    confidence = float(top / (top + second + 1e-6))
    confidence = max(0.25, min(confidence, 1.0))

    return best, confidence, evidence


@runtime_checkable
class RegionClassifier(Protocol):
    """Pluggable document-region labeling (deterministic or future LLM-backed)."""

    def classify_document(self, document: Document) -> list[PageRegion]:
        """Return one primary region label per PDF page (1-based page numbers)."""
        ...


class HeuristicRegionClassifier:
    """Conservative deterministic region tagging for Phase 2D-2."""

    def classify_document(self, document: Document) -> list[PageRegion]:
        pages = sorted(document.pages, key=lambda p: p.page_number)
        out: list[PageRegion] = []
        for page in pages:
            region_type, confidence, evidence = _score_region(page.page_number, page.text)
            out.append(
                PageRegion(
                    document_id=document.document_id,
                    page_number=page.page_number,
                    region_type=region_type,
                    confidence=confidence,
                    evidence=sorted(set(evidence)),
                )
            )
        return out
