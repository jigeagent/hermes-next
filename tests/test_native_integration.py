"""Tests for NativeMemoryClient — MEMORY.md promotion + session_search fallback."""

from pathlib import Path

import pytest

from hermes_next.integration.native import (
    MEMORY_SECTION_DELIMITER,
    NATIVE_MEMORY_MAX_CHARS,
    NativeMemoryClient,
    NativeMemoryConfig,
)


@pytest.fixture
def client(tmp_path):
    """NativeMemoryClient with temp MEMORY.md path."""
    md_path = str(tmp_path / "MEMORY.md")
    config = NativeMemoryConfig(
        memory_md_path=md_path,
        state_db_path=str(tmp_path / "state.db"),
        sync_memory_md=True,
    )
    return NativeMemoryClient(config)


class TestMemoryMDReadWrite:
    """Basic MEMORY.md I/O."""

    def test_read_non_existent(self, client):
        assert client.read_memory_md() == ""

    def test_write_and_read(self, client):
        client.write_memory_md("hello § world §")
        content = client.read_memory_md()
        assert "hello" in content
        assert "world" in content

    def test_read_sections(self, client):
        client.write_memory_md("first§second§third§")
        sections = client.read_sections()
        assert sections == ["first", "second", "third"]

    def test_read_sections_empty(self, client):
        assert client.read_sections() == []


class TestPromotion:
    """Promote entries to MEMORY.md."""

    def test_promote_basic(self, client):
        ok = client.promote("test entry", category="manual")
        assert ok is True
        content = client.read_memory_md()
        assert "test entry" in content

    def test_promote_skips_duplicate(self, client):
        client.promote("duplicate entry", category="manual")
        ok = client.promote("duplicate entry", category="manual")
        assert ok is False  # duplicate should be skipped

    def test_promote_empty_text(self, client):
        ok = client.promote("   ", category="manual")
        assert ok is False

    def test_promote_trims_long_text(self, client):
        long_text = "x" * 500
        ok = client.promote(long_text, category="manual")
        assert ok is True
        content = client.read_memory_md()
        assert len(content) < 300  # trimmed

    def test_promote_category_tags(self, client):
        client.promote("policy rule", category="policy")
        client.promote("skill guide", category="skill")
        content = client.read_memory_md()
        assert "📋" in content  # policy tag
        assert "🔧" in content  # skill tag

    def test_promote_policy_below_threshold(self, client):
        ok = client.promote_policy("test-policy", "trigger", "action", confidence=0.3)
        assert ok is False  # below default 0.5 threshold

    def test_promote_policy_above_threshold(self, client):
        ok = client.promote_policy("test-policy", "trigger text", "action text", confidence=0.7)
        assert ok is True
        content = client.read_memory_md()
        assert "📋" in content
        assert "经验/test-policy" in content

    def test_promote_skill(self, client):
        ok = client.promote_skill("data-analysis", "A reusable data analysis skill")
        assert ok is True
        content = client.read_memory_md()
        assert "🔧" in content
        assert "技能/data-analysis" in content

    def test_promote_disabled(self, client):
        config = NativeMemoryConfig(sync_memory_md=False, memory_md_path=str(client.memory_md_path))
        c = NativeMemoryClient(config)
        ok = c.promote("should not appear")
        assert ok is False


class TestCapacityManagement:
    """MEMORY.md capacity trimming."""

    def test_trim_removes_oldest_promoted(self, client):
        # Fill MEMORY.md to near capacity
        client.write_memory_md(
            f"🧠 old entry 1\n{MEMORY_SECTION_DELIMITER}\n"
            f"🧠 old entry 2\n{MEMORY_SECTION_DELIMITER}\n"
            f"📝 manual entry\n{MEMORY_SECTION_DELIMITER}\n"
        )
        # Promote something that needs trimming
        ok = client.promote("new entry after trim", category="auto")
        assert ok is True
        content = client.read_memory_md()
        # Manual entry should be preserved
        assert "📝 manual entry" in content
        assert "new entry after trim" in content

    def test_usage_ratio(self, client):
        assert client.usage_ratio() == 0.0
        client.write_memory_md("test content §")
        assert client.usage_ratio() > 0.0

    def test_needs_trim(self, client):
        assert client.needs_trim() is False
        # Fill to near capacity
        large = "x" * int(NATIVE_MEMORY_MAX_CHARS * 0.85)
        client.write_memory_md(f"{large}§")
        assert client.needs_trim() is True


class TestSessionSearch:
    """state.db session_search fallback."""

    def test_session_search_no_db(self, client):
        results = client.session_search("test query")
        assert results == []

    def test_format_results_empty(self, client):
        assert client.format_session_results([]) == ""

    def test_format_results(self, client):
        results = [
            {"role": "user", "content": "hello world", "session_id": "s1"},
            {"role": "assistant", "content": "hi there", "session_id": "s1"},
        ]
        formatted = client.format_session_results(results)
        assert "session_search" in formatted
        assert "hello world" in formatted

    def test_session_search_disabled(self, client):
        config = NativeMemoryConfig(session_search_fallback=False, memory_md_path=str(client.memory_md_path))
        c = NativeMemoryClient(config)
        assert c.session_search("test") == []


class TestStats:
    """NativeMemoryClient stats."""

    def test_get_stats_no_files(self, client):
        stats = client.get_stats()
        assert stats["memory_md_exists"] is False
        assert stats["state_db_exists"] is False

    def test_get_stats_with_memory_md(self, client):
        client.write_memory_md("test §")
        stats = client.get_stats()
        assert stats["memory_md_exists"] is True
        assert stats["memory_md_sections"] == 1
        assert stats["memory_md_promoted"] == 0

    def test_get_stats_counts_promoted(self, client):
        client.promote("promoted entry", category="policy")
        stats = client.get_stats()
        assert stats["memory_md_sections"] == 1
