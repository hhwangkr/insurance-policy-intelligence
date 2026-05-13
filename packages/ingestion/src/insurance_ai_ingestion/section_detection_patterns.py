"""Shared compiled regexes and heading markers for section + region heuristics."""

from __future__ import annotations

import re

RULE_MATCH_CONFIDENCE = 0.82

_TOC_LINE = re.compile(r"^[\s\[［【（(]*목\s*차[\s\]］】）)]*$")
_APPENDIX_PAREN = re.compile(r"^\s*[\(（]\s*별표\s*(\d+)\s*[\)）]\s*(.*)$")
_APPENDIX_PLAIN = re.compile(r"^\s*별표\s*(\d+)\s*(.+)$")
_PART = re.compile(r"^\s*(제\s*\d+\s*관)\s*(.+)$")
_ARTICLE = re.compile(r"^\s*(제\s*\d+\s*조)\s*(.*)$")

_PAGE_POINTER_TAIL = re.compile(
    r"(?:[\.\．]\s*\d{1,4}\s*|\d{1,4}\s*p\s*)$",
    re.IGNORECASE,
)
_LEGAL_GUIDE_POINTER = re.compile(
    r"^\s*관련법규\s*\d+\s*p?\s*$",
    re.IGNORECASE,
)

# Region scoring: appendix heading lines (prefix forms; see region_classification).
BYEOLPYO_PAREN_HEAD_PREFIX = re.compile(r"^\s*[\(（]\s*별표\s*\d+\s*[\)）]")
BYEOLPYO_PLAIN_TWO_TOKEN = re.compile(r"^\s*별표\s+\d+\s+\S")
BYEOLPYO_PLAIN_COMPACT = re.compile(r"^\s*별표\s*\d+(?:\s+\S|\S)")

# Legal / tail headings (spaced and collapsed).
LEGAL_CITED_LAW_LINE = re.compile(r"^\s*약관에서\s*인용(?:한|된)\s*법")
LEGAL_CITED_LAW_COMPACT = re.compile(r"^약관에서인용(?:한|된)법")
RELATED_LAW_HEADING = re.compile(r"^\s*관련\s*법규\s*$")
RELATED_LAW_HEADING_PREFIX = re.compile(r"^\s*관련\s*법규\s*[:\：]")

# Policy units (Kyobo-style integrated variants).
POLICY_VARIANT_GROUP = "적립형|거치형|즉시형"

_GLOBAL_TAIL_HEADING_RE = re.compile(
    r"(?m)^\s*(?:"
    r"약관에서\s*인용(?:한|된)\s*법(?:[\s·\.]*규정|\s*령)?"
    r"|약관에서\s*인용(?:한|된)\s*법령"
    r"|보험용어\s*해설"
    r"|보험용어해설"
    r"|관련\s*법규"
    r"|관련법규"
    r")(?:\s|$|[.:：])",
)

TAIL_MATTER_SUBSTRINGS: tuple[str, ...] = (
    "약관에서인용된법령",
    "약관에서인용한법·규정",
    "약관에서 인용한 법·규정",
    "약관에서 인용된 법령",
    "보험용어해설",
    "보험용어 해설",
    "관련법규",
)

# Public names (aliases) for heading matchers used by section_detection.
TOC_LINE = _TOC_LINE
APPENDIX_PAREN = _APPENDIX_PAREN
APPENDIX_PLAIN = _APPENDIX_PLAIN
PART = _PART
ARTICLE = _ARTICLE
PAGE_POINTER_TAIL = _PAGE_POINTER_TAIL
LEGAL_GUIDE_POINTER = _LEGAL_GUIDE_POINTER
GLOBAL_TAIL_HEADING_RE = _GLOBAL_TAIL_HEADING_RE

__all__ = [
    "RULE_MATCH_CONFIDENCE",
    "TOC_LINE",
    "APPENDIX_PAREN",
    "APPENDIX_PLAIN",
    "PART",
    "ARTICLE",
    "PAGE_POINTER_TAIL",
    "LEGAL_GUIDE_POINTER",
    "BYEOLPYO_PAREN_HEAD_PREFIX",
    "BYEOLPYO_PLAIN_TWO_TOKEN",
    "BYEOLPYO_PLAIN_COMPACT",
    "LEGAL_CITED_LAW_LINE",
    "LEGAL_CITED_LAW_COMPACT",
    "RELATED_LAW_HEADING",
    "RELATED_LAW_HEADING_PREFIX",
    "POLICY_VARIANT_GROUP",
    "GLOBAL_TAIL_HEADING_RE",
    "TAIL_MATTER_SUBSTRINGS",
]
