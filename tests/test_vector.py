"""Tests for hermes_next/cache/vector.py — EmbeddingEngine."""

import numpy as np
from hermes_next.cache.vector import EmbeddingEngine


def test_embedding_engine_embed():
    engine = EmbeddingEngine()
    texts = ["hello world", "test message"]
    result = engine.embed(texts)
    assert len(result) == 2
    for vec in result:
        assert len(vec) == 384
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5


def test_embedding_engine_embed_query():
    engine = EmbeddingEngine()
    vec = engine.embed_query("hello world")
    assert len(vec) == 384


def test_embedding_engine_singleton():
    e1 = EmbeddingEngine()
    e2 = EmbeddingEngine()
    assert e1 is e2


def test_embedding_engine_empty_input():
    engine = EmbeddingEngine()
    assert engine.embed([]) == []
    vec = engine.embed_query("")
    assert len(vec) == 384
