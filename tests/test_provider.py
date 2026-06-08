"""Tests for HermesNextProvider."""

from unittest.mock import MagicMock, patch

import pytest

from hermes_next.config import HermesNextConfig
from hermes_next.provider import HermesNextProvider


@pytest.fixture
def config():
    return HermesNextConfig()


class TestProviderInit:
    """Provider initialization and lifecycle."""

    def test_name(self, config):
        provider = HermesNextProvider(config)
        assert provider.name == "hermes-next"

    def test_is_available_no_client(self, config):
        provider = HermesNextProvider(config)
        assert provider.is_available() is False

    @patch("hermes_next.provider.OpenVikingClient")
    def test_initialize(self, mock_client_class, config):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = HermesNextProvider(config)
        provider.initialize(session_id="test-session")

        assert provider._initialized is True
        mock_client_class.assert_called_once()

    @patch("hermes_next.provider.OpenVikingClient")
    def test_shutdown(self, mock_client_class, config):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = HermesNextProvider(config)
        provider.initialize(session_id="test-session")
        provider.shutdown()

        assert provider._initialized is False
        mock_client.close.assert_called_once()


class TestProviderTools:
    """Tool schema and dispatch."""

    @patch("hermes_next.provider.OpenVikingClient")
    def test_get_tool_schemas(self, mock_client_class, config):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = HermesNextProvider(config)
        provider.initialize(session_id="test")

        schemas = provider.get_tool_schemas()
        assert len(schemas) == 4
        names = [s["function"]["name"] for s in schemas]
        assert "memos_search" in names
        assert "memos_get" in names
        assert "memos_timeline" in names
        assert "memos_status" in names

    @patch("hermes_next.provider.OpenVikingClient")
    def test_handle_tool_call_unknown(self, mock_client_class, config):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = HermesNextProvider(config)
        provider.initialize(session_id="test")

        result = provider.handle_tool_call("nonexistent", {})
        assert "Unknown tool" in result

    @patch("hermes_next.provider.OpenVikingClient")
    def test_handle_tool_call_not_initialized(self, mock_client_class, config):
        provider = HermesNextProvider(config)
        result = provider.handle_tool_call("memos_search", {"query": "test"})
        assert "not initialized" in result


class TestProviderPrefetch:
    """Prefetch pipeline."""

    @patch("hermes_next.provider.OpenVikingClient")
    def test_prefetch_uninitialized(self, mock_client_class, config):
        provider = HermesNextProvider(config)
        result = provider.prefetch("test query", session_id="s1")
        assert result == ""

    @patch("hermes_next.provider.CacheConnection")
    @patch("hermes_next.provider.RetrievalPipeline")
    @patch("hermes_next.provider.OpenVikingClient")
    def test_prefetch_with_results(
        self,
        mock_client_class,
        mock_pipeline_class,
        mock_cache_class,
        config,
    ):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = [
            {"id": "1", "content": "test content", "score": 0.95}
        ]
        mock_pipeline_class.return_value = mock_pipeline

        provider = HermesNextProvider(config)
        provider.initialize(session_id="test")
        result = provider.prefetch("test query", session_id="test")

        assert "相关记忆" in result
        assert "test content" in result
        mock_pipeline.retrieve.assert_called_once()


class TestProviderSyncTurn:
    """Turn synchronization."""

    @patch("hermes_next.provider.capture_trace")
    @patch("hermes_next.provider.OpenVikingClient")
    def test_sync_turn(self, mock_client_class, mock_capture, config):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_capture.return_value = None

        provider = HermesNextProvider(config)
        provider.initialize(session_id="test")

        provider.sync_turn(
            user_content="hello",
            assistant_content="world",
            session_id="test",
        )

        assert provider._turn_index == 1
        mock_capture.assert_called_once()


class TestProviderCompression:
    """Context compression hooks."""

    def test_on_pre_compress_empty(self, config):
        provider = HermesNextProvider(config)
        assert provider.on_pre_compress([]) == ""

    def test_on_pre_compress_with_messages(self, config):
        provider = HermesNextProvider(config)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = provider.on_pre_compress(messages)
        assert "[user]" in result
        assert "[assistant]" in result
