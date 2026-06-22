"""Tests for OV fallback to local embeddings."""

from unittest.mock import MagicMock
from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.traces import TraceRepository
from hermes_next.memos.types import TraceRow
from hermes_next.retrieval.pipeline import RetrievalPipeline
import os, tempfile, time


def _seed_trace_with_embedding(repo: TraceRepository):
    emb = [0.1] * 384
    repo.insert(TraceRow(
        id="fallback_test", session_id="s1", turn_index=0,
        user_content="test python programming",
        assistant_content="关于Python编程的讨论",
        embedding=emb, reward=0.0, tags=[], metadata={},
        created_at="2026-01-01T00:00:00Z",
    ))


def test_retrieve_fallback_when_ov_offline():
    """When OV is offline, fallback to local embedding search."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    cache = None
    try:
        cache = CacheConnection(db_path)
        from hermes_next.cache.schema import ensure_schema
        ensure_schema(cache)
        repo = TraceRepository(cache)
        _seed_trace_with_embedding(repo)

        mock_client = MagicMock()
        mock_client.health.side_effect = Exception("OV offline")

        pipeline = RetrievalPipeline(ov_client=mock_client, cache=cache)
        results = pipeline.retrieve("Python programming")
        assert len(results) > 0
        # Confirm at least one result came from the local fallback
        sources = {r.get("source") for r in results}
        assert "local_embed" in sources, (
            f"Expected local_embed source, got sources: {sources}"
        )
    finally:
        if cache:
            cache.close_all()
        # Retry deletion with small delay in case SQLite still locks
        for attempt in range(3):
            try:
                os.unlink(db_path)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)


def test_retrieve_fallback_empty_db():
    """When OV is offline and DB is empty, return empty gracefully."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    cache = None
    try:
        cache = CacheConnection(db_path)
        from hermes_next.cache.schema import ensure_schema
        ensure_schema(cache)

        mock_client = MagicMock()
        mock_client.health.side_effect = Exception("OV offline")

        pipeline = RetrievalPipeline(ov_client=mock_client, cache=cache)
        results = pipeline.retrieve("anything")
        assert isinstance(results, list)
        # With no embeddings in DB, fallback returns empty
        assert len(results) == 0
    finally:
        if cache:
            cache.close_all()
        for attempt in range(3):
            try:
                os.unlink(db_path)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)
