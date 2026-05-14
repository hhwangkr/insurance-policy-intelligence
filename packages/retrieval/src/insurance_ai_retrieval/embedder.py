from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PassageEmbedder(Protocol):
    """Minimal embedder surface for indexing and search (tests supply fakes)."""

    model_name: str

    def encode_passages(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        """Return ``(len(texts), dim)`` L2-normalized rows when possible."""
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """Return shape ``(dim,)``, L2-normalized when possible."""
        ...


class LocalSentenceTransformerEmbedder:
    """Local ``sentence-transformers`` backend (downloads weights on first use)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode_passages(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        from insurance_ai_retrieval.e5_text import format_e5_passage

        prefixed = [format_e5_passage(t) for t in texts]
        emb = self._model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        out: np.ndarray = np.asarray(emb, dtype=np.float32)
        return out

    def encode_query(self, text: str) -> np.ndarray:
        from insurance_ai_retrieval.e5_text import format_e5_query

        q = format_e5_query(text)
        emb = self._model.encode(
            [q],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vec: np.ndarray = np.asarray(emb, dtype=np.float32)[0]
        return vec
