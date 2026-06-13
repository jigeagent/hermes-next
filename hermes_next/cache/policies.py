"""Policy repository — local SQLite CRUD for policies."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.schema import ensure_schema
from hermes_next.memos.promote_gate import (
    evaluate_gate,
    load_gate_state,
    save_gate_state,
    update_gate_state,
    log_rejection,
)
from hermes_next.memos.types import PolicyRow


# Global step counter for gate tracking
_GLOBAL_STEP: int = 0


def _gate_enabled() -> bool:
    return os.environ.get("HERMES_NEXT_GATE_ENABLED", "True") in ("1", "true", "True")


class PolicyRepository:
    """Persist and query policies locally."""

    def __init__(self, cache: CacheConnection):
        self._cache = cache
        ensure_schema(cache)

    def insert(self, policy: PolicyRow) -> None:
        """Insert a policy into local cache."""
        conn = self._cache.conn
        conn.execute(
            """
            INSERT OR REPLACE INTO policies
                (id, name, description, trigger_pattern, action_template,
                 embedding, confidence, activation_count, source_trace_ids,
                 metadata, created_at, synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                policy.id,
                policy.name,
                policy.description,
                policy.trigger_pattern,
                policy.action_template,
                json.dumps(policy.embedding) if policy.embedding else None,
                policy.confidence,
                policy.activation_count,
                json.dumps(policy.source_trace_ids, ensure_ascii=False),
                json.dumps(policy.metadata, ensure_ascii=False, default=str),
                policy.created_at,
                0,
            ),
        )
        conn.commit()

    def get(self, policy_id: str) -> Optional[PolicyRow]:
        """Get a policy by ID."""
        row = self._cache.conn.execute(
            "SELECT * FROM policies WHERE id = ?", (policy_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_policy(row)

    def list_active(
        self,
        min_confidence: float = 0.3,
        limit: int = 20,
        use_gate: Optional[bool] = None,
    ) -> list[PolicyRow]:
        """List policies with confidence above threshold, optionally gated.

        When use_gate is True (default), candidates are filtered through
        evaluate_gate(): only policies that exceed the current gate score
        are returned. When use_gate is False, falls back to v0.4.0 behavior
        (simple threshold filter).

        Args:
            min_confidence: Minimum confidence threshold (pre-filter).
            limit: Max number of policies to return.
            use_gate: Override for gate. Defaults to HERMES_NEXT_GATE_ENABLED.

        Returns:
            List of active PolicyRow objects.
        """
        use_gate = _gate_enabled() if use_gate is None else use_gate

        rows = self._cache.conn.execute(
            "SELECT * FROM policies WHERE confidence >= ? ORDER BY confidence DESC LIMIT ?",
            (min_confidence, limit),
        ).fetchall()
        policies = [self._row_to_policy(r) for r in rows]

        if not use_gate or not policies:
            return policies

        # Gate filtering: only return policies that beat current score
        global _GLOBAL_STEP
        gate_state = load_gate_state()
        accepted = []

        for policy in policies:
            _GLOBAL_STEP += 1
            # Soft score based on activation frequency
            soft = min(policy.activation_count / 10.0, 1.0)

            result = evaluate_gate(
                candidate_id=policy.id,
                candidate_hard=policy.confidence,
                current_state=gate_state,
                global_step=_GLOBAL_STEP,
                candidate_soft=soft,
            )

            if result.action != "reject":
                accepted.append(policy)
                gate_state = update_gate_state(gate_state, result)
            else:
                log_rejection(result, reason=f"confidence {policy.confidence:.2f} <= gate current")

        save_gate_state(gate_state)
        return accepted[:limit]

    def increment_activation(self, policy_id: str) -> None:
        """Increment activation count for a policy."""
        self._cache.conn.execute(
            "UPDATE policies SET activation_count = activation_count + 1 WHERE id = ?",
            (policy_id,),
        )
        self._cache.conn.commit()

    def count(self) -> int:
        """Total policy count."""
        row = self._cache.conn.execute("SELECT COUNT(*) as cnt FROM policies").fetchone()
        return row["cnt"] if row else 0

    def update_metadata(self, policy_id: str, metadata: dict[str, Any]) -> None:
        """Update policy metadata (e.g., repair blocks)."""
        self._cache.execute(
            "UPDATE policies SET metadata = ? WHERE id = ?",
            (json.dumps(metadata, ensure_ascii=False, default=str), policy_id),
        )
        self._cache.conn.commit()

    @staticmethod
    def _row_to_policy(row: Any) -> PolicyRow:
        return PolicyRow(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            trigger_pattern=row["trigger_pattern"],
            action_template=row["action_template"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            confidence=row["confidence"],
            activation_count=row["activation_count"],
            source_trace_ids=json.loads(row["source_trace_ids"])
            if isinstance(row["source_trace_ids"], str)
            else [],
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else {},
            created_at=row["created_at"],
        )
