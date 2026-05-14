from __future__ import annotations

import re
from dataclasses import dataclass

_SUFFIX_RE = re.compile(
    r"^(.+)_policy_terms_(?P<effective_date>\d{8})_(?P<content_hash>[a-f0-9]{8})$"
)

# Longest-first so ``variable_annuity`` wins over the ``annuity`` suffix inside slugs.
_KNOWN_PRODUCT_TYPES: tuple[str, ...] = (
    "variable_annuity",
    "whole_life",
    "annuity",
    "cancer",
)


@dataclass(frozen=True)
class DocumentIdParts:
    """Deterministic segments parsed from normalized ``document_id`` keys."""

    insurer: str
    product_type: str
    product_slug: str
    product_name: str
    effective_date: str
    content_hash: str


def parse_document_id(document_id: str) -> DocumentIdParts | None:
    """Parse normalized ``document_id`` keys used by staging and ingestion.

    Expected pattern: ``{insurer}_{product_type}_{slug}_policy_terms_{YYYYMMDD}_{hash8}``.
    ``product_type`` may contain underscores (for example ``whole_life``).
    ``product_name`` mirrors ``product_slug`` with underscores replaced by spaces.
    """
    stripped = document_id.strip()
    m = _SUFFIX_RE.match(stripped)
    if not m:
        return None
    left = m.group(1)
    if "_" not in left:
        return None
    insurer, rest = left.split("_", 1)

    product_type: str | None = None
    product_slug: str | None = None
    for pt in _KNOWN_PRODUCT_TYPES:
        if rest == pt:
            return None
        if rest.startswith(pt + "_"):
            product_type = pt
            product_slug = rest[len(pt) + 1 :]
            break

    if product_type is None or product_slug is None:
        if "_" not in rest:
            return None
        head, tail = rest.split("_", 1)
        product_type = head
        product_slug = tail

    product_name = product_slug.replace("_", " ")
    return DocumentIdParts(
        insurer=insurer,
        product_type=product_type,
        product_slug=product_slug,
        product_name=product_name,
        effective_date=m.group("effective_date"),
        content_hash=m.group("content_hash"),
    )
