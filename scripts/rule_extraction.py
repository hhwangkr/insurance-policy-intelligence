from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Literal, cast

from metadata_models import FieldInference
from staging_lib import normalize_key_segment

_KOREAN_DATE = re.compile(r"(?P<y>\d{4})년\s*(?P<m>\d{1,2})월\s*(?P<d>\d{1,2})일")
_ISO = re.compile(r"(?P<y>20\d{2})-(?P<m>\d{2})-(?P<d>\d{2})")
_DOT = re.compile(r"(?P<y>20\d{2})\.(?P<m>\d{2})\.(?P<d>\d{2})")
_COMPACT8 = re.compile(r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})")


def extract_pdf_head_text(path: Path, *, max_pages: int = 5) -> str:
    """Extract plain text from the first `max_pages` pages (PyMuPDF)."""
    import fitz

    doc = fitz.open(path)
    try:
        chunks: list[str] = []
        limit = min(max_pages, len(doc))
        for i in range(limit):
            page = doc[i]
            chunks.append(page.get_text("text"))
        return "\n".join(chunks)
    finally:
        doc.close()


def _iso_from_ymd(year: int, month: int, day: int) -> str | None:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    try:
        date(year, month, day)
    except ValueError:
        return None
    return f"{year}-{month:02d}-{day:02d}"


def _yy_to_full_year(yy: int) -> int:
    """Map two-digit year to four-digit (00-69 -> 2000-2069, else 1900+y)."""
    if 0 <= yy <= 69:
        return 2000 + yy
    return 1900 + yy


def parse_effective_date_candidates(*, text: str, filename: str) -> list[tuple[str, str, str]]:
    """Return ordered list of (iso_date, evidence, confidence) with best candidates first."""
    scored: dict[str, tuple[int, int, str, str]] = {}
    rank_map = {"high": 0, "medium": 1, "low": 2, "unknown": 3}

    def upsert(iso: str, evidence: str, confidence: str) -> None:
        rank = rank_map.get(confidence, 3)
        prefers_filename = 0 if "filename" in evidence else 1
        key = (rank, prefers_filename)
        current = scored.get(iso)
        if current is None or key < (current[0], current[1]):
            scored[iso] = (rank, prefers_filename, evidence, confidence)

    hay_name = filename
    hay_all = f"{filename}\n{text}"

    for match in _ISO.finditer(hay_all):
        iso = _iso_from_ymd(int(match.group("y")), int(match.group("m")), int(match.group("d")))
        if iso:
            where = "filename" if match.group(0) in hay_name else "text"
            upsert(iso, f"pattern:YYYY-MM-DD:{where}", "high")

    for match in _DOT.finditer(hay_all):
        iso = _iso_from_ymd(int(match.group("y")), int(match.group("m")), int(match.group("d")))
        if iso:
            where = "filename" if match.group(0) in hay_name else "text"
            upsert(iso, f"pattern:YYYY.MM.DD:{where}", "high")

    for match in _COMPACT8.finditer(hay_all):
        iso = _iso_from_ymd(int(match.group("y")), int(match.group("m")), int(match.group("d")))
        if iso:
            where = "filename" if match.group(0) in hay_name else "text"
            upsert(iso, f"pattern:YYYYMMDD:{where}", "high")

    for match in _KOREAN_DATE.finditer(text):
        iso = _iso_from_ymd(int(match.group("y")), int(match.group("m")), int(match.group("d")))
        if iso:
            upsert(iso, "pattern:korean_yyyymmdd", "medium")

    for token in re.split(r"[^0-9]+", hay_name):
        if len(token) == 6 and token.isdigit():
            yy, mm, dd = int(token[:2]), int(token[2:4]), int(token[4:6])
            year = _yy_to_full_year(yy)
            iso = _iso_from_ymd(year, mm, dd)
            if iso:
                upsert(iso, "pattern:YYMMDD:filename_token", "medium")

    ordered = sorted(
        scored.items(),
        key=lambda item: (item[1][0], item[1][1], item[0]),
    )
    return [(iso, ev, conf) for iso, (_r, _p, ev, conf) in ordered]


def infer_effective_date(*, text: str, filename: str) -> FieldInference:
    candidates = parse_effective_date_candidates(text=text, filename=filename)
    if not candidates:
        return FieldInference(
            value="",
            confidence="unknown",
            needs_review=True,
            evidence="none",
        )
    iso, evidence, confidence_raw = candidates[0]
    confidence = cast(
        Literal["high", "medium", "low", "unknown"],
        confidence_raw if confidence_raw in ("high", "medium", "low", "unknown") else "unknown",
    )
    return FieldInference(
        value=iso,
        confidence=confidence,
        needs_review=confidence != "high",
        evidence=evidence,
    )


def infer_insurer(*, text: str, filename: str) -> FieldInference:
    hay = f"{filename}\n{text}"
    if "삼성생명" in hay:
        return FieldInference(
            value="samsunglife",
            confidence="high",
            needs_review=False,
            evidence="keyword:삼성생명",
        )
    if "삼성" in hay:
        return FieldInference(
            value="samsunglife",
            confidence="medium",
            needs_review=True,
            evidence="keyword:삼성",
        )
    if "교보생명" in hay:
        return FieldInference(
            value="kyobolife",
            confidence="high",
            needs_review=False,
            evidence="keyword:교보생명",
        )
    if "교보" in hay:
        return FieldInference(
            value="kyobolife",
            confidence="medium",
            needs_review=True,
            evidence="keyword:교보",
        )
    if "미래에셋생명" in hay:
        return FieldInference(
            value="miraeassetlife",
            confidence="high",
            needs_review=False,
            evidence="keyword:미래에셋생명",
        )
    if "미래에셋" in hay:
        return FieldInference(
            value="miraeassetlife",
            confidence="medium",
            needs_review=True,
            evidence="keyword:미래에셋",
        )
    return FieldInference(value="", confidence="unknown", needs_review=True, evidence="none")


def infer_document_type(*, text: str, filename: str) -> FieldInference:
    hay = f"{filename}\n{text}"
    if "보험약관" in hay:
        return FieldInference(
            value="policy_terms",
            confidence="high",
            needs_review=False,
            evidence="keyword:보험약관",
        )
    if "약관" in hay:
        return FieldInference(
            value="policy_terms",
            confidence="medium",
            needs_review=True,
            evidence="keyword:약관",
        )
    return FieldInference(value="", confidence="unknown", needs_review=True, evidence="none")


def infer_product_type(*, text: str, filename: str) -> FieldInference:
    hay = f"{filename}\n{text}"
    if "변액연금" in hay:
        return FieldInference(
            value="variable_annuity",
            confidence="high",
            needs_review=False,
            evidence="keyword:변액연금",
        )
    if "개인연금" in hay:
        return FieldInference(
            value="annuity",
            confidence="high",
            needs_review=False,
            evidence="keyword:개인연금",
        )
    if "연금보험" in hay:
        return FieldInference(
            value="annuity",
            confidence="medium",
            needs_review=True,
            evidence="keyword:연금보험",
        )
    if "연금" in hay and ("보험" in hay or "저축" in hay):
        return FieldInference(
            value="annuity",
            confidence="medium",
            needs_review=True,
            evidence="keyword:연금+보험/저축",
        )
    if "종신" in hay:
        return FieldInference(
            value="whole_life",
            confidence="medium",
            needs_review=True,
            evidence="keyword:종신",
        )
    if re.search(r"암", hay):
        return FieldInference(
            value="cancer",
            confidence="low",
            needs_review=True,
            evidence="keyword:암",
        )
    return FieldInference(value="", confidence="unknown", needs_review=True, evidence="none")


def infer_product_name(*, text: str, filename: str) -> FieldInference:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:12]:
        if len(line) >= 6:
            return FieldInference(
                value=line[:200],
                confidence="medium",
                needs_review=True,
                evidence="pdf_text:first_substantial_line",
            )
    stem = Path(filename).stem
    if stem:
        return FieldInference(
            value=stem[:200],
            confidence="low",
            needs_review=True,
            evidence="filename:stem_fallback",
        )
    return FieldInference(value="", confidence="unknown", needs_review=True, evidence="none")


def infer_product_slug(*, product_name: str) -> FieldInference:
    try:
        slug = normalize_key_segment(product_name)
    except ValueError:
        slug = "unknown_product"
        return FieldInference(
            value=slug,
            confidence="low",
            needs_review=True,
            evidence="slug:fallback_unknown_product",
        )
    return FieldInference(
        value=slug,
        confidence="medium",
        needs_review=True,
        evidence="slug:normalized_product_name",
    )


def infer_language(*, text: str) -> FieldInference:
    if re.search(r"[\uac00-\ud7a3]", text):
        return FieldInference(
            value="ko",
            confidence="medium",
            needs_review=True,
            evidence="heuristic:hangul_present",
        )
    return FieldInference(value="", confidence="unknown", needs_review=True, evidence="none")


def infer_dataset_split() -> FieldInference:
    return FieldInference(
        value="unassigned",
        confidence="low",
        needs_review=True,
        evidence="default:unassigned",
    )
