"""Tests for MemOS L1 Trace capture."""

from unittest.mock import MagicMock, patch

from hermes_next.memos.capture import capture_trace
from hermes_next.memos.id import new_id, timestamp_from_id
from hermes_next.memos.types import TraceRow


class TestIDGeneration:
    """UUID v7 + Crockford base32 ID generation."""

    def test_new_id_length(self):
        """Default ID should be 26 chars."""
        id_str = new_id()
        assert len(id_str) == 26

    def test_new_id_custom_length(self):
        """Custom length ID."""
        id_str = new_id(length=16)
        assert len(id_str) == 16

    def test_new_id_alphanumeric(self):
        """ID should only contain Crockford base32 chars."""
        id_str = new_id()
        valid_chars = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
        assert all(c in valid_chars for c in id_str)

    def test_new_id_monotonic(self):
        """IDs should be roughly time-ordered."""
        id1 = new_id()
        id2 = new_id()
        ts1 = timestamp_from_id(id1)
        ts2 = timestamp_from_id(id2)
        assert ts1 is not None
        assert ts2 is not None
        assert ts2 >= ts1

    def test_timestamp_from_id(self):
        """Should extract a plausible timestamp."""
        id_str = new_id()
        ts = timestamp_from_id(id_str)
        assert ts is not None
        assert ts > 1_700_000_000_000  # After 2023


class TestTraceRow:
    """TraceRow data model."""

    def test_to_dict(self):
        trace = TraceRow(
            id="test123",
            session_id="session-1",
            turn_index=1,
            user_content="hello",
            assistant_content="world",
            embedding=[0.1, 0.2],
            tags=["test"],
        )
        d = trace.to_dict()
        assert d["id"] == "test123"
        assert d["embedding"] == [0.1, 0.2]

    def test_from_dict(self):
        data = {
            "id": "abc",
            "session_id": "s1",
            "turn_index": 2,
            "user_content": "hi",
            "assistant_content": "there",
            "tags": ["a", "b"],
        }
        trace = TraceRow.from_dict(data)
        assert trace.id == "abc"
        assert trace.tags == ["a", "b"]
        assert trace.embedding is None


class TestCaptureTrace:
    """L1 Trace capture logic."""

    @patch("hermes_next.memos.capture.OpenVikingClient")
    def test_capture_success(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.embed.return_value = [0.1, 0.2, 0.3]
        mock_client.content_write.return_value = True

        trace = capture_trace(
            client=mock_client,
            session_id="s1",
            turn_index=1,
            user_content="user msg",
            assistant_content="assistant msg",
            agent_name="test-agent",
        )

        assert trace is not None
        assert trace.user_content == "user msg"
        assert trace.assistant_content == "assistant msg"
        assert trace.embedding == [0.1, 0.2, 0.3]
        assert trace.session_id == "s1"
        assert trace.turn_index == 1

        # Verify API calls
        mock_client.embed.assert_called_once()
        mock_client.content_write.assert_called_once()

    @patch("hermes_next.memos.capture.OpenVikingClient")
    def test_capture_failed_persist(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.embed.return_value = [0.1, 0.2, 0.3]
        mock_client.content_write.return_value = False

        trace = capture_trace(
            client=mock_client,
            session_id="s1",
            turn_index=1,
            user_content="user msg",
            assistant_content="assistant msg",
        )

        assert trace is None

    @patch("hermes_next.memos.capture.OpenVikingClient")
    def test_capture_with_tags_and_metadata(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.embed.return_value = [0.1, 0.2]
        mock_client.content_write.return_value = True

        trace = capture_trace(
            client=mock_client,
            session_id="s1",
            turn_index=2,
            user_content="user",
            assistant_content="assistant",
            agent_name="my-agent",
            tags=["chat"],
            metadata={"context": "debugging"},
        )

        assert trace is not None
        assert "chat" in trace.tags
        assert trace.metadata["context"] == "debugging"
