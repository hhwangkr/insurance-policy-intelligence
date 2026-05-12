from __future__ import annotations

import hashlib
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
_YYMM_PARENS = re.compile(r"\((?P<yy>\d{2})(?P<mm>\d{2})\)")


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

    for hay_yymm in (hay_name, hay_all):
        for match in _YYMM_PARENS.finditer(hay_yymm):
            yy = int(match.group("yy"))
            mm = int(match.group("mm"))
            year = _yy_to_full_year(yy)
            iso = _iso_from_ymd(year, mm, 1)
            if iso:
                where = "filename" if match.group(0) in hay_name else "text"
                upsert(
                    iso,
                    f"pattern:YYMM_parens_inferred_day_01:{where}",
                    "low",
                )

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
    needs_review = confidence != "high" or "inferred_day_01" in evidence
    return FieldInference(
        value=iso,
        confidence=confidence,
        needs_review=needs_review,
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


def _collapse_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _is_savings_pension_disclaimer_line(line: str) -> bool:
    collapsed = _collapse_ws(line)
    if "저축(연금) 목적" in collapsed:
        return True
    if (
        "본 상품은 보장성보험으로" in collapsed
        and "저축(연금) 목적에는 적합하지 않습니다" in collapsed
    ):
        return True
    return False


def _filter_lines_excluding_samsung_disclaimer(text: str) -> str:
    """Remove generic Samsung-style savings/pension disclaimer lines from free text."""
    kept: list[str] = []
    for raw in text.splitlines():
        if _is_savings_pension_disclaimer_line(raw):
            continue
        kept.append(raw)
    return "\n".join(kept)


def _classify_product_type_from_hay(hay: str) -> tuple[str, str] | None:
    """Return (product_type, evidence_suffix) using ordered keyword priority."""
    if "변액연금보험" in hay or "변액연금" in hay:
        return "variable_annuity", "변액연금"
    if "종신보험" in hay or "밸런스종신보험" in hay or "밸런스종신" in hay:
        return "whole_life", "종신"
    if (
        "간편통합암보험" in hay
        or "통합암보험" in hay
        or "인터넷암보험" in hay
        or "암치료보험" in hay
        or "암보험" in hay
    ):
        return "cancer", "cancer_product_keywords"
    if "개인연금" in hay or "연금보험" in hay:
        return "annuity", "연금_product_keywords"
    return None


def infer_product_type(*, text: str, filename: str, product_name: str) -> FieldInference:
    """Infer product type from the chosen product title + filename, not generic disclaimer pages."""
    filtered_text = _filter_lines_excluding_samsung_disclaimer(text)
    name = product_name.strip()
    hay_primary = "\n".join(part for part in (name, filename) if part)
    hay_secondary = "\n".join(part for part in (filtered_text, filename) if part)

    primary = _classify_product_type_from_hay(hay_primary)
    if primary is not None:
        ptype, hint = primary
        conf: Literal["high", "medium", "low", "unknown"] = "high" if name else "medium"
        return FieldInference(
            value=ptype,
            confidence=conf,
            needs_review=conf != "high",
            evidence=f"product_signal:{hint}:primary",
        )

    secondary = _classify_product_type_from_hay(hay_secondary)
    if secondary is not None:
        ptype, hint = secondary
        return FieldInference(
            value=ptype,
            confidence="medium",
            needs_review=True,
            evidence=f"product_signal:{hint}:filtered_text",
        )

    return FieldInference(value="", confidence="unknown", needs_review=True, evidence="none")


_PRODUCT_NAME_KEYWORDS: tuple[str, ...] = (
    "간편통합암보험",
    "통합암보험",
    "인터넷암보험",
    "암치료보험",
    "변액연금보험",
    "밸런스종신보험",
    "교보로연금보험",
    "개인연금",
    "연금보험",
    "종신보험",
    "암보험",
)


def _is_generic_product_name_line(line: str) -> bool:
    """True for common cover / ToC / boilerplate lines that are not product titles."""
    collapsed = _collapse_ws(line)
    lowered = collapsed.lower()
    if _is_savings_pension_disclaimer_line(line):
        return True
    if collapsed in {"보험약관", "목 차", "목차", "고객권리안내문"}:
        return True
    if "고객권리안내문" in collapsed:
        return True
    if "약관 이용 guide book" in lowered:
        return True
    if collapsed.startswith("본 상품은 보장성보험으로"):
        return True
    if collapsed.startswith("목차"):
        return True
    if collapsed.startswith("목 차"):
        return True
    return False


def infer_product_name(*, text: str, filename: str) -> FieldInference:
    """Pick a product title from head text; skip boilerplate; prefer known product keywords."""
    lines = [_collapse_ws(ln) for ln in text.splitlines() if _collapse_ws(ln)]
    scoped = lines[:40]

    for keyword in _PRODUCT_NAME_KEYWORDS:
        for line in scoped:
            if keyword not in line:
                continue
            if _is_generic_product_name_line(line):
                continue
            return FieldInference(
                value=line[:200],
                confidence="high",
                needs_review=False,
                evidence=f"pdf_text:keyword:{keyword}",
            )

    for line in scoped:
        if len(line) < 6:
            continue
        if _is_generic_product_name_line(line):
            continue
        return FieldInference(
            value=line[:200],
            confidence="medium",
            needs_review=True,
            evidence="pdf_text:first_non_generic_line",
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


_SLUG_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("간편통합암보험", "easy_integrated_cancer_insurance"),
    ("통합암보험", "integrated_cancer_insurance"),
    ("인터넷암보험", "internet_cancer_insurance"),
    ("암치료보험", "cancer_treatment_insurance"),
    ("변액연금보험", "variable_annuity_insurance"),
    ("밸런스종신보험", "balance_whole_life_insurance"),
    ("교보로연금보험", "kyobo_ro_annuity_insurance"),
    ("개인연금", "personal_pension"),
    ("연금보험", "annuity_insurance"),
    ("종신보험", "whole_life_insurance"),
    ("암보험", "cancer_insurance"),
)


def infer_product_slug(*, product_name: str, product_type: str) -> FieldInference:
    """Deterministic English slug from known Korean tokens, ASCII, or type-scoped hash."""
    collapsed = _collapse_ws(product_name)
    for keyword, slug in _SLUG_KEYWORDS:
        if keyword in collapsed:
            return FieldInference(
                value=slug,
                confidence="high",
                needs_review=False,
                evidence=f"slug:keyword:{keyword}",
            )

    latin = re.sub(r"[^A-Za-z0-9]+", " ", collapsed).strip()
    if latin:
        try:
            slug = normalize_key_segment(latin)
        except ValueError:
            pass
        else:
            return FieldInference(
                value=slug,
                confidence="medium",
                needs_review=True,
                evidence="slug:ascii_normalized",
            )

    digest = hashlib.sha256(collapsed.encode("utf-8")).hexdigest()[:8]
    slug = f"{product_type}_{digest}"
    return FieldInference(
        value=slug,
        confidence="low",
        needs_review=True,
        evidence="slug:hash8_fallback",
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
