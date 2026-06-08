"""Feedback repository — persist and query user feedback signals."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.schema import ensure_schema
from hermes_next.memos.feedback import FeedbackSignal


class FeedbackRepository:
    """Persist and query feedback signals."""

    def __init__(self, cache: CacheConnection):
        self._cache = cache
        ensure_schema(cache)

    def insert(self, signal: FeedbackSignal) -> None:
        self._cache.execute(
            """
            INSERT OR REPLACE INTO feedback
                (id, episode_id, trace_id, polarity, magnitude,
                 text, source, agent_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"fb_{signal.episode_id}_{signal.created_at}",
                signal.episode_id,
                signal.trace_id,
                signal.polarity,
                signal.magnitude,
                signal.text,
                signal.source,
                signal.agent_name,
                signal.created_at,
            ),
        )

    def count_recent(
        self,
        episode_id: str,
        polarity: Optional[str] = None,
        within_seconds: int = 30,
        agent_name: Optional[str] = None,
    ) -> int:
        """Count recent feedback signals for debouncing.

        Args:
            episode_id: Target episode.
            polarity: Optional filter by polarity ('positive'|'negative').
            within_seconds: Time window from now.
            agent_name: Optional filter by agent.
        """
        cutoff = datetime.now(timezone.utc).isoformat()
        # Simple approach: count rows after a time offset
        # SQLite doesn't do datetime math natively with ISO strings,
        # so we fetch recent rows and filter in Python
        conditions = ["episode_id = ?"]
        params: list = [episode_id]

        if polarity:
            conditions.append("polarity = ?")
            params.append(polarity)
        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)

        sql = f"SELECT created_at FROM feedback WHERE {' AND '.join(conditions)} ORDER BY created_at DESC"
        rows = self._cache.execute(sql, tuple(params)).fetchall()

        if not rows:
            return 0

        import time
        now = time.time()
        count = 0
        for row in rows:
            try:
                ts = datetime.fromisoformat(row["created_at"]).timestamp()
                if now - ts <= within_seconds:
                    count += 1
                else:
                    break
            except (ValueError, TypeError):
                continue
        return count

    def list_by_episode(
        self,
        episode_id: str,
        limit: int = 50,
    ) -> list[FeedbackSignal]:
        rows = self._cache.execute(
            "SELECT * FROM feedback WHERE episode_id = ? ORDER BY created_at DESC LIMIT ?",
            (episode_id, limit),
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def count_negative_since(
        self,
        episode_id: str,
        hours: int = 24,
    ) -> int:
        """Count negative feedback signals within a time window."""
        import time
        cutoff_ts = time.time() - (hours * 3600)
        rows = self._cache.execute(
            "SELECT created_at FROM feedback WHERE episode_id = ? AND polarity = 'negative' ORDER BY created_at DESC",
            (episode_id,),
        ).fetchall()
        count = 0
        for row in rows:
            try:
                ts = datetime.fromisoformat(row["created_at"]).timestamp()
                if ts >= cutoff_ts:
                    count += 1
            except (ValueError, TypeError):
                continue
        return count

    @staticmethod
    def _row_to_signal(row) -> FeedbackSignal:
        return FeedbackSignal(
            episode_id=row["episode_id"],
            trace_id=row["trace_id"] if "trace_id" in row and row["trace_id"] else None,
            polarity=row["polarity"],
            magnitude=row["magnitude"],
            text=row["text"] if "text" in row else None,
            source=row["source"] if "source" in row else "user",
            agent_name=row["agent_name"] if "agent_name" in row else "default",
            created_at=row["created_at"],
        )
