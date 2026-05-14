from __future__ import annotations


def format_e5_query(user_query: str) -> str:
    """Prefix a user query for intfloat E5-style multilingual models."""
    q = user_query.strip()
    lower = q.lower()
    if lower.startswith("query:"):
        return q
    return f"query: {q}"


def format_e5_passage(chunk_text: str) -> str:
    """Prefix chunk body text as a passage for intfloat E5-style multilingual models."""
    t = chunk_text.strip()
    lower = t.lower()
    if lower.startswith("passage:"):
        return t
    return f"passage: {t}"
