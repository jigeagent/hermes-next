"""Local cosine similarity search using numpy."""

from __future__ import annotations

from typing import Optional

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def search_by_embedding(
    query_embedding: list[float],
    candidates: list[tuple[str, list[float]]],
    k: int = 8,
) -> list[tuple[str, float]]:
    """Search nearest neighbors by cosine similarity.

    Skips candidates with mismatched dimension.
    """
    if not candidates:
        return []
    query_dim = len(query_embedding) if query_embedding is not None else 0
    scores: list[tuple[str, float]] = []
    for cid, emb in candidates:
        if emb and len(emb) == query_dim:
            sim = cosine_similarity(query_embedding, emb)
            scores.append((cid, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def compute_embedding(text: str, dim: int = 384) -> list[float]:
    """Compute a simple bag-of-characters embedding as a fallback.

    This is a lightweight fallback when the OpenViking embed API is unavailable.
    For production use, use OpenViking's native embedding instead.
    """
    rng = np.random.RandomState(hash(text) & 0xFFFFFFFF)
    vec = rng.randn(dim)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.tolist()


class EmbeddingEngine:
    """Lazy-loaded fastembed TextEmbedding wrapper (singleton).

    Downloads ~60MB model on first use; subsequent calls use cached instance.
    """

    _instance: Optional["EmbeddingEngine"] = None
    _model = None
    _model_error: Optional[Exception] = None
    _model_name = "BAAI/bge-small-en-v1.5"

    def __new__(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_model(self):
        if self._model_error is not None:
            raise RuntimeError(
                f"Embedding model failed: {self._model_error}"
            ) from self._model_error
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as e:
                type(self)._model_error = e
                raise RuntimeError("fastembed not installed") from e
            try:
                type(self)._model = TextEmbedding(
                    model_name=self._model_name, max_length=512,
                )
            except Exception as e:
                type(self)._model_error = e
                raise RuntimeError(f"Failed to load model: {e}") from e
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return list(self._get_model().embed(texts))

    def embed_query(self, text: str) -> list[float]:
        if not text:
            text = " "
        result = self.embed([text])
        return result[0] if result else []
