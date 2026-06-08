"""Tests for FeedbackSignal, FeedbackRepository, and Decision Repair."""

from hermes_next.memos.feedback import FeedbackSignal


class TestFeedbackSignal:
    """FeedbackSignal data model."""

    def test_default_created_at(self):
        signal = FeedbackSignal(episode_id="ep1", polarity="positive")
        assert signal.created_at != ""
        assert "T" in signal.created_at

    def test_agent_default(self):
        signal = FeedbackSignal(episode_id="ep1", polarity="negative")
        assert signal.agent_name == "default"

    def test_custom_agent(self):
        signal = FeedbackSignal(episode_id="ep1", polarity="negative", agent_name="haoermei")
        assert signal.agent_name == "haoermei"

    def test_magnitude_default(self):
        signal = FeedbackSignal(episode_id="ep1", polarity="positive")
        assert signal.magnitude == 1.0


class TestFeedbackRepository:
    """FeedbackRepository CRUD."""

    def test_insert_and_count(self, tmp_path):
        from hermes_next.cache.connection import CacheConnection
        from hermes_next.cache.feedback import FeedbackRepository

        cache = CacheConnection(str(tmp_path / "test.db"))
        repo = FeedbackRepository(cache)
        assert repo.count_negative_since("ep1") == 0

        signal = FeedbackSignal(
            episode_id="ep1", polarity="negative", text="wrong answer",
            created_at="2026-06-08T10:00:00",
        )
        repo.insert(signal)
        assert repo.count_negative_since("ep1") == 1

        signal2 = FeedbackSignal(
            episode_id="ep1", polarity="positive",
            created_at="2026-06-08T10:01:00",
        )
        repo.insert(signal2)

        signals = repo.list_by_episode("ep1")
        assert len(signals) == 2

    def test_count_recent(self, tmp_path):
        import time
        from hermes_next.cache.connection import CacheConnection
        from hermes_next.cache.feedback import FeedbackRepository

        cache = CacheConnection(str(tmp_path / "test2.db"))
        repo = FeedbackRepository(cache)

        # Insert 2 signals
        for _ in range(2):
            s = FeedbackSignal(episode_id="ep1", polarity="negative")
            repo.insert(s)

        count = repo.count_recent("ep1", polarity="negative", within_seconds=30)
        assert count >= 1

    def test_count_recent_debounce(self, tmp_path):
        import time
        from hermes_next.cache.connection import CacheConnection
        from hermes_next.cache.feedback import FeedbackRepository

        cache = CacheConnection(str(tmp_path / "test3.db"))
        repo = FeedbackRepository(cache)

        # Insert 3 signals
        for i in range(3):
            s = FeedbackSignal(episode_id="ep1", polarity="positive",
                               created_at=f"2025-01-01T00:00:{i:02d}Z")
            repo.insert(s)

        # With within_seconds=0, should return 0 (all are "older")
        count = repo.count_recent("ep1", polarity="positive", within_seconds=0)
        assert count == 0


class TestDecisionRepair:
    """Decision Repair module."""

    def test_no_text_returns_none(self):
        from hermes_next.memos.repair import apply_decision_repair
        from hermes_next.memos.feedback import FeedbackSignal

        signal = FeedbackSignal(episode_id="ep1", polarity="negative", text=None)
        result = apply_decision_repair(signal, [], None)
        assert result is None

    def test_no_policies_returns_none(self, tmp_path):
        from hermes_next.memos.repair import apply_decision_repair
        from hermes_next.memos.feedback import FeedbackSignal

        signal = FeedbackSignal(episode_id="ep1", polarity="negative", text="this is wrong")
        result = apply_decision_repair(signal, [], None)
        assert result is None

    def test_keyword_matching(self, tmp_path):
        from hermes_next.memos.repair import apply_decision_repair
        from hermes_next.memos.feedback import FeedbackSignal
        from hermes_next.memos.types import PolicyRow
        from unittest.mock import MagicMock

        policy = PolicyRow(
            id="p1", name="data-analysis",
            description="Analyze data and generate reports",
            trigger_pattern="analysis data",
            action_template="Run data analysis pipeline",
            confidence=0.8,
        )

        signal = FeedbackSignal(
            episode_id="ep1", polarity="negative",
            text="this analysis approach is wrong",
        )

        mock_repo = MagicMock()
        result = apply_decision_repair(signal, [policy], mock_repo)
        assert result is not None
        assert result.name == "data-analysis"
        mock_repo.update_metadata.assert_called_once()

    def test_scope_inference_global(self):
        from hermes_next.memos.repair import _infer_scope
        from hermes_next.memos.types import PolicyRow

        policy = PolicyRow(
            id="p1", name="general-greeting",
            description="Say hello",
            trigger_pattern="hello",
            action_template="Say hi",
        )
        scope = _infer_scope("this is not right", policy)
        assert scope == "global"

    def test_scope_inference_scene_specific(self):
        from hermes_next.memos.repair import _infer_scope
        from hermes_next.memos.types import PolicyRow

        policy = PolicyRow(
            id="p1", name="project-specific-analysis",
            description="Analysis for a specific project",
            trigger_pattern="project analysis",
            action_template="Run project analysis",
        )
        scope = _infer_scope("this method is wrong", policy)
        assert scope == "scene-specific"


class TestSessionStateRepository:
    """SessionState persistence."""

    def test_upsert_and_get(self, tmp_path):
        from hermes_next.cache.connection import CacheConnection
        from hermes_next.cache.session_state import SessionStateRepository, SessionState

        cache = CacheConnection(str(tmp_path / "test.db"))
        repo = SessionStateRepository(cache)

        state = SessionState(session_id="s1", agent_name="test-agent")
        repo.upsert(state)

        fetched = repo.get("s1")
        assert fetched is not None
        assert fetched.session_id == "s1"
        assert fetched.agent_name == "test-agent"

    def test_touch_updates_timestamp(self, tmp_path):
        from hermes_next.cache.connection import CacheConnection
        from hermes_next.cache.session_state import SessionStateRepository, SessionState

        cache = CacheConnection(str(tmp_path / "test2.db"))
        repo = SessionStateRepository(cache)

        repo.upsert(SessionState(session_id="s1"))
        repo.touch("s1", turn_index=5)

        fetched = repo.get("s1")
        assert fetched is not None
        assert fetched.turn_index == 5

    def test_close(self, tmp_path):
        from hermes_next.cache.connection import CacheConnection
        from hermes_next.cache.session_state import SessionStateRepository, SessionState

        cache = CacheConnection(str(tmp_path / "test3.db"))
        repo = SessionStateRepository(cache)

        repo.upsert(SessionState(session_id="s1"))
        repo.close("s1")

        fetched = repo.get("s1")
        assert fetched is not None
        assert fetched.status == "closed"

    def test_list_stale(self, tmp_path):
        from hermes_next.cache.connection import CacheConnection
        from hermes_next.cache.session_state import SessionStateRepository, SessionState

        cache = CacheConnection(str(tmp_path / "test4.db"))
        repo = SessionStateRepository(cache)

        repo.upsert(SessionState(
            session_id="old-session",
            last_active="2020-01-01T00:00:00",
        ))
        repo.upsert(SessionState(
            session_id="recent-session",
        ))

        stale = repo.list_stale(stale_hours=1)
        stale_ids = [s.session_id for s in stale]
        assert "old-session" in stale_ids
        assert "recent-session" not in stale_ids
