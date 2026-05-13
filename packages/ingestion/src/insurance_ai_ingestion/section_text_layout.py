"""Full-document text indexing for deterministic section and policy-unit logic."""

from __future__ import annotations

from insurance_ai_shared.models.document import Document


def document_full_text_index(
    document: Document,
) -> tuple[str, list[int], list[int], int]:
    """Join page texts with newlines; return (full_text, page_start_offsets, page_numbers, len)."""
    pages = sorted(document.pages, key=lambda p: p.page_number)
    texts = [p.text for p in pages]
    page_numbers = [p.page_number for p in pages]
    full_text = "\n".join(texts)
    starts: list[int] = []
    pos = 0
    for t in texts:
        starts.append(pos)
        pos += len(t) + 1
    return full_text, starts, page_numbers, len(full_text)


def page_for_offset(
    offset: int,
    *,
    page_starts: list[int],
    page_numbers: list[int],
    total_len: int,
) -> int:
    """Map a character offset in ``full_text`` to a 1-based page number."""
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
