"""Memory lifecycle management — archive, decay, and cleanup.

Handles trace archival, policy confidence decay, and skill version pruning
to keep the cognitive store healthy and bounded in size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from hermes_next.cache.connection import CacheConnection

logger = logging.getLogger(__name__)


@dataclass
class LifecycleConfig:
    """Configuration for memory lifecycle management."""

    trace_retention_days: int = 90
    """Traces older than this many days are archived (deleted from local cache)."""

    policy_decay_rate: float = 0.03
    """Confidence decay per day of inactivity (0.0 = no decay)."""

    policy_min_confidence: float = 0.05
    """Policies below this confidence are pruned."""

    skill_min_version: int = 1
    """Minimum skill version to keep (older versions pruned)."""

    cleanup_interval_traces: int = 500
    """Run trace cleanup every N new traces."""


@dataclass
class LifecycleStats:
    """Snapshot of lifecycle management state."""

    traces_before_cleanup: int = 0
    traces_after_cleanup: int = 0
    traces_archived: int = 0
    policies_before_decay: int = 0
    policies_after_decay: int = 0
    policies_pruned: int = 0
    skills_pruned: int = 0


class LifecycleManager:
    """Manages memory lifecycle: archival, decay, and pruning."""

    def __init__(
        self,
        cache: CacheConnection,
        config: Optional[LifecycleConfig] = None,
    ):
        self._cache = cache
        self._config = config or LifecycleConfig()
        self._trace_count_since_cleanup: int = 0
        self._last_cleanup: Optional[str] = None

    # ── Public API ────────────────────────────────────────

    def on_trace_inserted(self) -> None:
        """Increment counter and trigger cleanup if threshold reached."""
        self._trace_count_since_cleanup += 1
        if self._trace_count_since_cleanup >= self._config.cleanup_interval_traces:
            self.run_cleanup()

    def run_cleanup(self) -> LifecycleStats:
        """Run full lifecycle sweep: archive → decay → prune.

        Returns:
            Stats snapshot of what was cleaned up.
        """
        stats = LifecycleStats()

        # 1. Archive old traces
        archive_stats = self._archive_old_traces()
        stats.traces_archived = archive_stats
        stats.traces_before_cleanup = self._count_traces()

        # 2. Decay policy confidence
        decay_stats = self._decay_policies()
        stats.policies_before_decay = decay_stats["before"]
        stats.policies_after_decay = decay_stats["after"]
        stats.policies_pruned = decay_stats["pruned"]

        # 3. Prune low-confidence policies
        pruned_policies = self._prune_policies()
        stats.policies_pruned += pruned_policies

        # 4. Prune old skill versions
        pruned_skills = self._prune_skills()
        stats.skills_pruned = pruned_skills

        stats.traces_after_cleanup = self._count_traces()

        self._trace_count_since_cleanup = 0
        self._last_cleanup = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Lifecycle cleanup done: %d traces archived, %d policies decayed, "
            "%d policies pruned, %d skills pruned",
            stats.traces_archived,
            decay_stats.get("decayed", 0),
            pruned_policies,
            pruned_skills,
        )
        return stats

    # ── Internal Operations ───────────────────────────────

    def _archive_old_traces(self) -> int:
        """Delete traces older than retention_days from local cache."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._config.trace_retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")
        cursor = self._cache.execute(
            "DELETE FROM traces WHERE created_at < ? AND synced = 1",
            (cutoff_str,),
        )
        return cursor.rowcount if cursor else 0

    def _decay_policies(self) -> dict:
        """Apply time-based confidence decay to all policies.

        Confidences are reduced by decay_rate per day since creation.
        Returns stats dict.
        """
        rate = self._config.policy_decay_rate
        if rate <= 0.0:
            return {"before": 0, "after": 0, "pruned": 0, "decayed": 0}

        rows = self._cache.execute(
            "SELECT id, confidence, created_at FROM policies"
        ).fetchall()

        now = datetime.now(timezone.utc)
        decayed_count = 0
        for row in rows:
            if not row["created_at"]:
                continue
            try:
                created = datetime.fromisoformat(row["created_at"])
                days_since = (now - created).total_seconds() / 86400.0
                if days_since > 1:
                    decay = 1.0 - (rate * days_since)
                    new_confidence = max(0.0, row["confidence"] * decay)
                    self._cache.execute(
                        "UPDATE policies SET confidence = ? WHERE id = ?",
                        (new_confidence, row["id"]),
                    )
                    decayed_count += 1
            except (ValueError, TypeError):
                continue

        return {
            "before": len(rows),
            "after": self._count_policies(),
            "pruned": 0,
            "decayed": decayed_count,
        }

    def _prune_policies(self) -> int:
        """Delete policies below minimum confidence threshold."""
        cursor = self._cache.execute(
            "DELETE FROM policies WHERE confidence < ?",
            (self._config.policy_min_confidence,),
        )
        return cursor.rowcount if cursor else 0

    def _prune_skills(self) -> int:
        """Prune skills below minimum version (placeholder for versioned skills)."""
        # Current schema has one row per skill (no versioning),
        # so this is a no-op for now. Future enhancement.
        return 0

    # ── Helpers ───────────────────────────────────────────

    def _count_traces(self) -> int:
        row = self._cache.execute("SELECT COUNT(*) as cnt FROM traces").fetchone()
        return row["cnt"] if row else 0

    def _count_policies(self) -> int:
        row = self._cache.execute("SELECT COUNT(*) as cnt FROM policies").fetchone()
        return row["cnt"] if row else 0

    def get_stats(self) -> dict:
        """Return lifecycle health stats."""
        return {
            "trace_count": self._count_traces(),
            "policy_count": self._count_policies(),
            "trace_retention_days": self._config.trace_retention_days,
            "policy_decay_rate": self._config.policy_decay_rate,
            "policy_min_confidence": self._config.policy_min_confidence,
            "last_cleanup": self._last_cleanup,
            "traces_since_cleanup": self._trace_count_since_cleanup,
            "cleanup_interval": self._config.cleanup_interval_traces,
        }
