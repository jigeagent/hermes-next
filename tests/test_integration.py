"""Integration tests for hermes-next.

These tests require a running OpenViking server (default: localhost:1933).
Skip with: pytest -m "not integration"
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from hermes_next.cache.concepts import ConceptRepository, TripleRepository
from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.lifecycle import LifecycleConfig, LifecycleManager
from hermes_next.cache.schema import drop_schema, ensure_schema
from hermes_next.cache.traces import TraceRepository
from hermes_next.config import HermesNextConfig
from hermes_next.memos.types import TraceRow
from hermes_next.memos.world_model import Concept, Triple
from hermes_next.provider import HermesNextProvider
from hermes_next.retrieval.pipeline import RetrievalPipeline

# Mark all tests in this module as integration
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("HERMES_NEXT_INTEGRATION"),
        reason="Set HERMES_NEXT_INTEGRATION=1 to run integration tests",
    ),
]


@pytest.fixture
def temp_cache(tmp_path):
    """Create a temporary SQLite cache."""
    db_path = str(tmp_path / "test_cache.db")
    cache = CacheConnection(db_path)
    ensure_schema(cache)
    yield cache
    drop_schema(cache)
    cache.close_all()


class TestCacheIntegration:
    """SQLite cache integration."""

    def test_trace_crud(self, temp_cache):
        repo = TraceRepository(temp_cache)
        assert repo.count() == 0

        trace = TraceRow(
            id="test-1",
            session_id="s1",
            turn_index=1,
            user_content="user msg",
            assistant_content="assistant msg",
            embedding=[0.1, 0.2],
            tags=["test"],
        )
        repo.insert(trace)
        assert repo.count() == 1

        fetched = repo.get("test-1")
        assert fetched is not None
        assert fetched.user_content == "user msg"
        assert fetched.assistant_content == "assistant msg"
        assert fetched.embedding == [0.1, 0.2]

    def test_list_by_session(self, temp_cache):
        repo = TraceRepository(temp_cache)
        for i in range(3):
            repo.insert(TraceRow(
                id=f"t{i}", session_id="s1", turn_index=i,
                user_content=f"u{i}", assistant_content=f"a{i}",
            ))
        traces = repo.list_by_session("s1")
        assert len(traces) == 3

    def test_mark_synced(self, temp_cache):
        repo = TraceRepository(temp_cache)
        repo.insert(TraceRow(
            id="t1", session_id="s1", turn_index=0,
            user_content="u", assistant_content="a",
        ))
        unsynced = repo.get_unsynced()
        assert len(unsynced) == 1
        repo.mark_synced("t1")
        unsynced = repo.get_unsynced()
        assert len(unsynced) == 0


class TestRetrievalPipelineIntegration:
    """Retrieval pipeline integration."""

    @patch("hermes_next.retrieval.pipeline.OpenVikingClient")
    def test_pipeline_with_mock_ov(self, mock_client_class, temp_cache):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.search_find.return_value = [
            {"id": "ov1", "content": "semantic result", "score": 0.9}
        ]

        repo = TraceRepository(temp_cache)
        repo.insert(TraceRow(
            id="local1", session_id="s1", turn_index=0,
            user_content="local user", assistant_content="local asst",
        ))

        pipeline = RetrievalPipeline(mock_client, temp_cache)

        # We need to inject the trace_repo since it was created inside __init__
        results = pipeline.retrieve("test query", agent="test")
        assert len(results) > 0


class TestProviderIntegration:
    """Full provider integration."""

    @patch("hermes_next.provider.CacheConnection")
    @patch("hermes_next.provider.OpenVikingClient")
    def test_full_lifecycle(self, mock_client_class, mock_cache_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.health.return_value = True
        mock_client.embed.return_value = [0.1, 0.2]
        mock_client.content_write.return_value = True
        mock_client.search_find.return_value = [
            {"id": "1", "content": "memory", "score": 0.9}
        ]

        config = HermesNextConfig()
        provider = HermesNextProvider(config)
        provider.initialize(session_id="int-test")

        assert provider.is_available() is True

        # Prefetch
        ctx = provider.prefetch("test", session_id="int-test")
        assert isinstance(ctx, str)

        # Tool calls
        search_result = provider.handle_tool_call("memos_search", {"query": "test"})
        assert isinstance(search_result, str)

        timeline = provider.handle_tool_call("memos_timeline", {"limit": 5})
        assert isinstance(timeline, str)

        # Compression
        summary = provider.on_pre_compress([
            {"role": "user", "content": "hello"},
        ])
        assert isinstance(summary, str)

        provider.shutdown()
        assert provider._initialized is False


class TestConceptRepository:
    """Concept and Triple repository CRUD."""

    def test_concept_crud(self, temp_cache):
        repo = ConceptRepository(temp_cache)
        assert repo.count() == 0

        concept = Concept(
            id="cpt-1",
            label="test_concept",
            description="A test concept",
            embedding=[0.1, 0.2],
            member_trace_ids=["t1", "t2"],
        )
        repo.insert(concept)
        assert repo.count() == 1

        fetched = repo.get("cpt-1")
        assert fetched is not None
        assert fetched.label == "test_concept"
        assert fetched.member_trace_ids == ["t1", "t2"]

    def test_triple_crud(self, temp_cache):
        repo = TripleRepository(temp_cache)
        assert repo.count() == 0

        triple = Triple(
            id="tri-1",
            subject="Python",
            predicate="is_a",
            object_="programming language",
            confidence=0.95,
            source_trace_id="t1",
        )
        repo.insert(triple)
        assert repo.count() == 1

        # Query by subject
        results = repo.query(subject="Python")
        assert len(results) == 1
        assert results[0].object_ == "programming language"

    def test_triple_batch_insert(self, temp_cache):
        repo = TripleRepository(temp_cache)
        triples = [
            Triple(id=f"tri-{i}", subject="A", predicate="related_to", object_=f"Object{i}")
            for i in range(5)
        ]
        repo.insert_batch(triples)
        assert repo.count() == 5


class TestLifecycleManager:
    """Lifecycle manager basic operations."""

    def test_archive_old_traces(self, temp_cache):
        repo = TraceRepository(temp_cache)
        repo.insert(TraceRow(
            id="old-trace", session_id="s1", turn_index=0,
            user_content="old", assistant_content="old",
            created_at="2020-01-01T00:00:00",
        ))
        # Mark as synced so it's eligible for archival
        repo.mark_synced("old-trace")
        repo.insert(TraceRow(
            id="new-trace", session_id="s1", turn_index=1,
            user_content="new", assistant_content="new",
            created_at="2025-06-08T00:00:00",
        ))

        config = LifecycleConfig(trace_retention_days=30)
        manager = LifecycleManager(temp_cache, config)
        archived = manager._archive_old_traces()
        assert archived >= 1

    def test_get_stats(self, temp_cache):
        manager = LifecycleManager(temp_cache)
        stats = manager.get_stats()
        assert "trace_count" in stats
        assert "policy_count" in stats
        assert "last_cleanup" in stats


class TestProviderWithCognitivePipeline:
    """Provider cognitive pipeline integration tests."""

    @patch("hermes_next.provider.OpenVikingClient")
    def test_pipeline_status_tool(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.health.return_value = True

        config = HermesNextConfig()
        provider = HermesNextProvider(config)
        provider.initialize(session_id="pipeline-test")

        # memos_status should render pipeline info
        status = provider.handle_tool_call("memos_status", {})
        assert isinstance(status, str)
        assert "Pipeline" in status or "Status" in status

        provider.shutdown()

    @patch("hermes_next.provider.OpenVikingClient")
    @patch("hermes_next.provider.CacheConnection")
    def test_cognitive_pipeline_wired(self, mock_cache_class, mock_client_class):
        """Verify sync_turn feeds into cognitive pipeline."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        # capture_trace returns a trace
        mock_client.embed.return_value = [0.1, 0.2]
        mock_client.content_write.return_value = True

        config = HermesNextConfig()
        provider = HermesNextProvider(config)
        provider.initialize(session_id="wire-test")

        # sync_turn should add a trace to session_traces via pipeline
        provider.sync_turn(
            user_content="test user",
            assistant_content="test assistant",
            session_id="wire-test",
        )
        assert len(provider._session_traces) >= 1

        provider.shutdown()
