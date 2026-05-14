from __future__ import annotations

from pathlib import Path

from insurance_ai_shared.models.chunk import DocumentChunk, DocumentChunksArtifact


def load_chunk_artifacts_from_dir(chunks_dir: Path) -> list[DocumentChunksArtifact]:
    """Load every ``*.chunks.json`` under ``chunks_dir`` (sorted by path)."""
    if not chunks_dir.is_dir():
        msg = f"chunks directory not found: {chunks_dir}"
        raise FileNotFoundError(msg)
    paths = sorted(chunks_dir.glob("*.chunks.json"))
    if not paths:
        msg = f"no *.chunks.json files in {chunks_dir}"
        raise ValueError(msg)
    return [
        DocumentChunksArtifact.model_validate_json(p.read_text(encoding="utf-8")) for p in paths
    ]


def flatten_chunks_sorted(artifacts: list[DocumentChunksArtifact]) -> list[DocumentChunk]:
    """Stable global order: ``document_id``, then ``char_start``, ``char_end``, ``chunk_id``."""
    arts = sorted(artifacts, key=lambda a: a.document_id)
    rows: list[DocumentChunk] = []
    for art in arts:
        rows.extend(art.chunks)
    rows.sort(key=lambda c: (c.document_id, c.char_start, c.char_end, c.chunk_id))
    return rows
