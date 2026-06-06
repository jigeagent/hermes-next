"""Tests for Hermes Next Viewer."""

import json
import threading
from http.server import HTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request

import pytest

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.schema import drop_schema, ensure_schema
from hermes_next.memos.types import TraceRow, PolicyRow, SkillRow
from hermes_next.viewer.server import _APIHandler, serve


@pytest.fixture
def temp_cache(tmp_path):
    db_path = str(tmp_path / "viewer_test.db")
    cache = CacheConnection(db_path)
    ensure_schema(cache)

    # Insert test data
    cache.execute(
        "INSERT INTO traces (id, session_id, turn_index, user_content, assistant_content, reward, created_at, synced) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("trace-1", "session-1", 0, "hello", "world", 1.0, "2024-01-01T00:00:00", 1),
    )
    cache.execute(
        "INSERT INTO traces (id, session_id, turn_index, user_content, assistant_content, reward, created_at, synced) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("trace-2", "session-1", 1, "how are you", "fine", 0.5, "2024-01-01T00:01:00", 0),
    )
    cache.execute(
        "INSERT INTO policies (id, name, description, confidence, activation_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("pol-1", "test_policy", "A test policy", 0.8, 5, "2024-01-01T00:00:00"),
    )
    cache.execute(
        "INSERT INTO skills (name, description, usage_guide, version, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("test_skill", "A test skill", "## Usage\nDo X", 1, "2024-01-01T00:00:00"),
    )

    yield cache
    drop_schema(cache)
    cache.close_all()


class TestViewerAPI:
    """Viewer API endpoint tests."""

    def _start_server(self, cache: CacheConnection, port: int = 0):
        """Start a test server on a random port."""
        _APIHandler._cache = cache
        _APIHandler._ov_url = "http://localhost:1933"
        server = HTTPServer(("127.0.0.1", port or 0), _APIHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server, port

    def _fetch(self, port: int, path: str) -> Any:
        url = f"http://127.0.0.1:{port}{path}"
        with urlopen(url) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_health(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            data = self._fetch(port, "/api/health")
            assert data["status"] == "ok"
        finally:
            server.shutdown()

    def test_stats(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            data = self._fetch(port, "/api/stats")
            assert data["traces"] == 2
            assert data["policies"] == 1
            assert data["skills"] == 1
            assert data["synced"] == 1
        finally:
            server.shutdown()

    def test_traces_list(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            data = self._fetch(port, "/api/traces")
            assert data["count"] == 2
            assert len(data["traces"]) == 2
        finally:
            server.shutdown()

    def test_trace_detail(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            data = self._fetch(port, "/api/traces/trace-1")
            assert data["user_content"] == "hello"
            assert data["assistant_content"] == "world"
        finally:
            server.shutdown()

    def test_trace_detail_not_found(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            with pytest.raises(Exception):
                self._fetch(port, "/api/traces/nonexistent")
        finally:
            server.shutdown()

    def test_policies(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            data = self._fetch(port, "/api/policies")
            assert data["count"] == 1
            assert data["policies"][0]["name"] == "test_policy"
        finally:
            server.shutdown()

    def test_skills(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            data = self._fetch(port, "/api/skills")
            assert data["count"] == 1
            assert data["skills"][0]["name"] == "test_skill"
        finally:
            server.shutdown()

    def test_timeline(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            data = self._fetch(port, "/api/timeline")
            assert data["count"] == 2
        finally:
            server.shutdown()

    def test_search(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            data = self._fetch(port, "/api/search?q=hello")
            assert data["count"] >= 1
        finally:
            server.shutdown()

    def test_search_empty(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            data = self._fetch(port, "/api/search?q=")
            assert data["count"] == 0
        finally:
            server.shutdown()

    def test_not_found(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            with pytest.raises(Exception):
                self._fetch(port, "/api/nonexistent")
        finally:
            server.shutdown()

    def test_spa_root(self, temp_cache):
        server, port = self._start_server(temp_cache)
        try:
            with urlopen(f"http://127.0.0.1:{port}/") as resp:
                html = resp.read().decode("utf-8")
                assert "Hermes Next Viewer" in html
        finally:
            server.shutdown()


class TestViewerServe:
    """serve() function integration."""

    def test_serve_function(self, tmp_path):
        db_path = str(tmp_path / "serve_test.db")
        # Just test it starts without error
        from hermes_next.viewer import serve
        import threading
        t = threading.Thread(
            target=serve,
            args=(db_path, "http://localhost:1933", 0),
            daemon=True,
        )
        t.start()
        assert t.is_alive()
