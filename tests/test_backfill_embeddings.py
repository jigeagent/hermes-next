"""Tests for embedding backfill."""

import os, sys, tempfile, json
from pathlib import Path
from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.schema import ensure_schema
from hermes_next.cache.traces import TraceRepository
from hermes_next.memos.types import TraceRow


def _seed_unembedded(repo: TraceRepository):
    for i in range(5):
        repo.insert(TraceRow(
            id=f"be_{i}", session_id="s1", turn_index=i,
            user_content=f"test {i}", assistant_content=f"response {i}",
            embedding=None, reward=0.0, tags=[], metadata={},
            created_at="2026-01-01T00:00:00Z",
        ))


def test_backfill_embeddings_idempotent():
    db_dir = Path(tempfile.mkdtemp())
    db_path = str(db_dir / "cache.db")
    try:
        cache = CacheConnection(db_path)
        ensure_schema(cache)
        repo = TraceRepository(cache)
        _seed_unembedded(repo)
        # Close so backfill_embeddings can open its own connection
        cache.close_all()

        from hermes_next.promote import backfill_embeddings
        result = backfill_embeddings(cache_path=db_path)
        assert result["processed"] == 5, f"first run: {result}"

        result2 = backfill_embeddings(cache_path=db_path)
        assert result2["processed"] == 0, f"second run: {result2}"
    finally:
        import shutil
        shutil.rmtree(str(db_dir), ignore_errors=True)
