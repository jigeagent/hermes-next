"""Performance benchmarks for hermes-next.

Run with: pytest tests/test_performance.py -v --benchmark
"""

import time
from pathlib import Path

import pytest

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.schema import drop_schema, ensure_schema
from hermes_next.cache.traces import TraceRepository
from hermes_next.cache.vector import cosine_similarity, search_by_embedding
from hermes_next.memos.id import new_id
from hermes_next.memos.types import TraceRow


# Skip unless --benchmark flag is given
pytestmark = pytest.mark.skipif(
    True,
    reason="Run with --benchmark flag: pytest tests/test_performance.py -v",
)


@pytest.fixture
def populated_cache(tmp_path):
    db_path = str(tmp_path / "perf_test.db")
    cache = CacheConnection(db_path)
    ensure_schema(cache)
    repo = TraceRepository(cache)

    # Insert 1000 traces
    traces = []
    for i in range(1000):
        traces.append(TraceRow(
            id=f"perf-{i:04d}",
            session_id=f"session-{i % 10}",
            turn_index=i,
            user_content=f"user query number {i} about topic {i % 20}",
            assistant_content=f"assistant response for query {i} with details",
            embedding=[float((i + j) % 100) / 100.0 for j in range(64)],
            tags=["test", f"topic-{i % 20}"],
            created_at=f"2024-01-01T00:{i // 60:02d}:{i % 60:02d}",
        ))
    repo.insert_batch(traces)
    yield cache, repo
    drop_schema(cache)
    cache.close_all()


class TestBatchInsert:
    """Batch insert performance."""

    def test_batch_insert_1000(self, tmp_path, benchmark):
        db_path = str(tmp_path / "batch_perf.db")
        cache = CacheConnection(db_path)
        ensure_schema(cache)
        repo = TraceRepository(cache)

        traces = [
            TraceRow(
                id=new_id(),
                session_id="perf-test",
                turn_index=i,
                user_content=f"query {i}",
                assistant_content=f"response {i}",
                embedding=[0.1] * 64,
            )
            for i in range(1000)
        ]

        def _batch():
            repo.insert_batch(traces)

        benchmark(_batch)
        drop_schema(cache)
        cache.close_all()


class TestFTSSearch:
    """FTS5 search performance."""

    def test_fts_search(self, populated_cache, benchmark):
        cache, repo = populated_cache

        def _search():
            return repo.search_fts("topic about", limit=10)

        results = benchmark(_search)
        assert len(results) <= 10


class TestVectorSimilarity:
    """Vector similarity performance."""

    def test_cosine_similarity(self, benchmark):
        a = [float(i) / 100.0 for i in range(384)]
        b = [float((i * 2) % 100) / 100.0 for i in range(384)]

        def _sim():
            return cosine_similarity(a, b)

        score = benchmark(_sim)
        assert -1.0 <= score <= 1.0

    def test_search_by_embedding(self, populated_cache, benchmark):
        cache, repo = populated_cache
        query_emb = [0.5] * 64
        candidates = repo.get_all_embeddings(limit=500)

        def _search():
            return search_by_embedding(query_emb, candidates, k=8)

        results = benchmark(_search)
        assert len(results) <= 8


class TestIDGeneration:
    """ID generation throughput."""

    def test_generate_1000_ids(self, benchmark):
        def _gen():
            return [new_id() for _ in range(1000)]

        ids = benchmark(_gen)
        assert len(ids) == 1000
