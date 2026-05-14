from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from insurance_ai_retrieval.chunks_io import flatten_chunks_sorted, load_chunk_artifacts_from_dir
from insurance_ai_retrieval.embedder import PassageEmbedder
from insurance_ai_retrieval.metadata import ChunkMetadataRecord

EMBEDDINGS_FILENAME = "chunk_embeddings.npy"
METADATA_FILENAME = "chunk_metadata.jsonl"
CONFIG_FILENAME = "index_config.json"


@dataclass(frozen=True)
class IndexConfig:
    model_name: str
    embedding_dim: int
    num_chunks: int
    created_at: str
    batch_size: int
    backend: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_chunks": self.num_chunks,
            "created_at": self.created_at,
            "batch_size": self.batch_size,
            "backend": self.backend,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, Any]) -> IndexConfig:
        return cls(
            model_name=str(raw["model_name"]),
            embedding_dim=int(raw["embedding_dim"]),
            num_chunks=int(raw["num_chunks"]),
            created_at=str(raw["created_at"]),
            batch_size=int(raw["batch_size"]),
            backend=str(raw["backend"]),
        )


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float
    metadata: ChunkMetadataRecord


def _write_jsonl(path: Path, records: list[ChunkMetadataRecord]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")


def _read_jsonl(path: Path) -> list[ChunkMetadataRecord]:
    rows: list[ChunkMetadataRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(ChunkMetadataRecord.model_validate_json(line))
    return rows


def build_local_index(
    *,
    chunks_dir: Path,
    index_dir: Path,
    embedder: PassageEmbedder,
    batch_size: int,
) -> IndexConfig:
    """Embed all chunks and persist numpy matrix + jsonl metadata + config."""
    artifacts = load_chunk_artifacts_from_dir(chunks_dir)
    chunks = flatten_chunks_sorted(artifacts)
    if not chunks:
        msg = "no chunks found across chunk artifacts"
        raise ValueError(msg)

    texts = [c.text for c in chunks]
    vectors = embedder.encode_passages(texts, batch_size=batch_size)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        msg = f"expected 2D embedding matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    n, dim = matrix.shape
    if n != len(chunks):
        msg = f"embedding rows ({n}) != chunks ({len(chunks)})"
        raise ValueError(msg)

    records = [ChunkMetadataRecord.from_document_chunk(c) for c in chunks]
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / EMBEDDINGS_FILENAME, matrix)
    _write_jsonl(index_dir / METADATA_FILENAME, records)

    stamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    cfg = IndexConfig(
        model_name=embedder.model_name,
        embedding_dim=dim,
        num_chunks=n,
        created_at=stamp,
        batch_size=batch_size,
        backend="numpy_normalized_dot",
    )
    (index_dir / CONFIG_FILENAME).write_text(
        json.dumps(cfg.to_json_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return cfg


def load_index_config(index_dir: Path) -> IndexConfig:
    raw_path = index_dir / CONFIG_FILENAME
    if not raw_path.is_file():
        msg = f"missing index config: {raw_path}"
        raise FileNotFoundError(msg)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    return IndexConfig.from_json_dict(raw)


def load_embeddings_matrix(index_dir: Path) -> np.ndarray:
    path = index_dir / EMBEDDINGS_FILENAME
    if not path.is_file():
        msg = f"missing embeddings file: {path}"
        raise FileNotFoundError(msg)
    arr = np.load(path)
    out: np.ndarray = np.asarray(arr, dtype=np.float32)
    return out


def load_metadata_rows(index_dir: Path) -> list[ChunkMetadataRecord]:
    path = index_dir / METADATA_FILENAME
    if not path.is_file():
        msg = f"missing metadata file: {path}"
        raise FileNotFoundError(msg)
    return _read_jsonl(path)


def search_local_index(
    *,
    index_dir: Path,
    query: str,
    embedder: PassageEmbedder,
    top_k: int,
) -> list[SearchHit]:
    """Cosine similarity via dot product on L2-normalized rows (see ``build_local_index``)."""
    cfg = load_index_config(index_dir)
    if embedder.model_name != cfg.model_name:
        msg = (
            f"embedder model {embedder.model_name!r} does not match index "
            f"config model {cfg.model_name!r}"
        )
        raise ValueError(msg)

    matrix = load_embeddings_matrix(index_dir)
    meta = load_metadata_rows(index_dir)
    if matrix.shape[0] != len(meta):
        msg = f"embeddings rows ({matrix.shape[0]}) != metadata rows ({len(meta)})"
        raise ValueError(msg)

    q = np.asarray(embedder.encode_query(query), dtype=np.float32)
    if q.shape != (cfg.embedding_dim,):
        msg = f"query embedding dim {q.shape} != index dim {cfg.embedding_dim}"
        raise ValueError(msg)

    scores = matrix @ q
    k = max(1, min(top_k, len(scores)))
    if len(scores) <= k:
        idx = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, k - 1)[:k]
        idx = part[np.argsort(-scores[part])]

    hits: list[SearchHit] = []
    for rank, i in enumerate(idx.tolist(), start=1):
        hits.append(SearchHit(rank=rank, score=float(scores[i]), metadata=meta[i]))
    return hits
