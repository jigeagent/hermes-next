"""Session state persistence — track open sessions for crash recovery.

v0.4.0: 启动恢复
- session_state 表记录每个 session 的状态
- initialize() 时扫描 stale session → 执行 session_end 流程
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.schema import ensure_schema


@dataclass
class SessionState:
    """Track a single session's lifecycle state."""

    session_id: str
    agent_name: str = "default"
    turn_index: int = 0
    status: str = "open"  # open | closed
    opened_at: str = ""
    last_active: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionStateRepository:
    """Persist and query session state for crash recovery."""

    def __init__(self, cache: CacheConnection):
        self._cache = cache
        ensure_schema(cache)

    def upsert(self, state: SessionState) -> None:
        self._cache.execute(
            """
            INSERT OR REPLACE INTO session_state
                (session_id, agent_name, turn_index, status,
                 opened_at, last_active, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.session_id,
                state.agent_name,
                state.turn_index,
                state.status,
                state.opened_at or datetime.now(timezone.utc).isoformat(),
                state.last_active or datetime.now(timezone.utc).isoformat(),
                json.dumps(state.metadata, ensure_ascii=False, default=str),
            ),
        )

    def touch(self, session_id: str, turn_index: int = 0) -> None:
        """Update last_active timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        self._cache.execute(
            "UPDATE session_state SET last_active = ?, turn_index = ? WHERE session_id = ?",
            (now, turn_index, session_id),
        )

    def close(self, session_id: str) -> None:
        self._cache.execute(
            "UPDATE session_state SET status = 'closed' WHERE session_id = ?",
            (session_id,),
        )

    def get(self, session_id: str) -> Optional[SessionState]:
        row = self._cache.execute(
            "SELECT * FROM session_state WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_state(row)

    def list_open(self, agent_name: Optional[str] = None) -> list[SessionState]:
        if agent_name:
            rows = self._cache.execute(
                "SELECT * FROM session_state WHERE status = 'open' AND agent_name = ? ORDER BY last_active DESC",
                (agent_name,),
            ).fetchall()
        else:
            rows = self._cache.execute(
                "SELECT * FROM session_state WHERE status = 'open' ORDER BY last_active DESC"
            ).fetchall()
        return [self._row_to_state(r) for r in rows]

    def list_stale(
        self,
        stale_hours: int = 4,
        agent_name: Optional[str] = None,
    ) -> list[SessionState]:
        """Return open sessions that haven't been active in stale_hours."""
        import time

        cutoff_ts = time.time() - (stale_hours * 3600)
        all_open = self.list_open(agent_name=agent_name)
        stale = []
        for s in all_open:
            try:
                ts = datetime.fromisoformat(s.last_active).timestamp()
                if ts < cutoff_ts:
                    stale.append(s)
            except (ValueError, TypeError):
                stale.append(s)
        return stale

    def delete_old(self, days: int = 30) -> int:
        """Delete closed session state older than days."""
        import time

        cutoff_ts = time.time() - (days * 86400)
        rows = self._cache.execute(
            "SELECT session_id, last_active FROM session_state WHERE status = 'closed'"
        ).fetchall()
        deleted = 0
        for row in rows:
            try:
                ts = datetime.fromisoformat(row["last_active"]).timestamp()
                if ts < cutoff_ts:
                    self._cache.execute(
                        "DELETE FROM session_state WHERE session_id = ?", (row["session_id"],)
                    )
                    deleted += 1
            except (ValueError, TypeError):
                continue
        return deleted

    @staticmethod
    def _row_to_state(row) -> SessionState:
        meta_raw = row["metadata"] if "metadata" in row and row["metadata"] else "{}"
        return SessionState(
            session_id=row["session_id"],
            agent_name=row["agent_name"],
            turn_index=row["turn_index"],
            status=row["status"],
            opened_at=row["opened_at"],
            last_active=row["last_active"],
            metadata=json.loads(meta_raw) if isinstance(meta_raw, str) else {},
        )
