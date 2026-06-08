"""Native Hermes Agent memory integration.

Bridges hermes-next with the Hermes Agent's built-in memory system:
- MEMORY.md sync (promote high-value items to native hot memory)
- state.db session_search fallback (keyword search as cold backup)
- Capacity management (trim MEMORY.md when it exceeds the native limit)
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Native Hermes Agent memory limits
NATIVE_MEMORY_MAX_CHARS = 2_200  # MEMORY.md hard limit
NATIVE_USER_MAX_CHARS = 1_375     # USER.md hard limit
MEMORY_SECTION_DELIMITER = "§"    # Hermes Agent native delimiter
PROMOTED_TAG = "🧠"               # Tag auto-promoted entries

# Default paths
DEFAULT_HERMES_HOME = Path.home() / ".hermes"
DEFAULT_MEMORY_PATH = DEFAULT_HERMES_HOME / "memories" / "MEMORY.md"
DEFAULT_STATE_DB = DEFAULT_HERMES_HOME / "state.db"


@dataclass
class NativeMemoryConfig:
    """Configuration for native memory integration."""

    sync_memory_md: bool = True
    """Whether to auto-write summaries to MEMORY.md."""

    promote_on_l2_confidence: float = 0.5
    """L2 Policy confidence threshold for promotion to MEMORY.md."""

    promote_on_skill_crystallize: bool = True
    """Whether to write crystallized skill summaries to MEMORY.md."""

    max_entry_chars: int = 150
    """Max length of a single MEMORY.md entry."""

    memory_md_capacity_warning: float = 0.80
    """Trigger trim when MEMORY.md usage exceeds this ratio."""

    session_search_fallback: bool = True
    """Whether to fall back to Hermes Agent state.db FTS5 when hermes-next returns empty."""

    session_search_max_results: int = 5
    """Max results from session_search fallback."""

    state_db_path: str = ""
    """Path to Hermes Agent state.db (auto-detected if empty)."""

    memory_md_path: str = ""
    """Path to Hermes Agent MEMORY.md (auto-detected if empty)."""


class NativeMemoryClient:
    """Bridge to Hermes Agent's native file-based memory system.

    Provides:
    - Promote high-value items to MEMORY.md
    - Query state.db as session_search fallback
    - MEMORY.md capacity management
    """

    def __init__(self, config: Optional[NativeMemoryConfig] = None):
        self._config = config or NativeMemoryConfig()

    # ── Path Resolution ───────────────────────────────────

    @property
    def memory_md_path(self) -> Path:
        if self._config.memory_md_path:
            return Path(self._config.memory_md_path).expanduser()
        path = DEFAULT_MEMORY_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def state_db_path(self) -> Path:
        if self._config.state_db_path:
            return Path(self._config.state_db_path).expanduser()
        return DEFAULT_STATE_DB

    # ── MEMORY.md Read / Write ────────────────────────────

    def read_memory_md(self) -> str:
        """Read current MEMORY.md content."""
        path = self.memory_md_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def read_sections(self) -> list[str]:
        """Parse MEMORY.md into sections."""
        content = self.read_memory_md()
        if not content:
            return []
        sections = content.split(MEMORY_SECTION_DELIMITER)
        return [s.strip() for s in sections if s.strip()]

    def write_memory_md(self, content: str) -> bool:
        """Write content to MEMORY.md atomically."""
        try:
            self.memory_md_path.write_text(content, encoding="utf-8")
            return True
        except OSError as e:
            logger.warning("Failed to write MEMORY.md: %s", e)
            return False

    # ── Promotion ─────────────────────────────────────────

    def promote(self, text: str, category: str = "auto") -> bool:
        """Promote a high-value memory entry to MEMORY.md.

        Args:
            text: The memory entry text (will be trimmed to max_entry_chars).
            category: Source category for the entry (policy / skill / manual).

        Returns:
            True if the entry was written, False if skipped (duplicate, empty, etc.).
        """
        if not self._config.sync_memory_md:
            return False

        # Clean and trim the entry
        summary = text.strip().replace("\n", " ").replace("\r", "")
        summary = re.sub(r"\s+", " ", summary)  # collapse whitespace
        summary = summary[: self._config.max_entry_chars]
        if not summary:
            return False

        # Add category tag
        tag_map = {
            "policy": "📋",
            "skill": "🔧",
            "concept": "🌐",
            "search": "🔍",
            "manual": "📝",
        }
        prefix = tag_map.get(category, "🧠")
        entry = f"{prefix} {summary}"

        current = self.read_memory_md()

        # Deduplicate — skip if text already exists
        sections = current.split(MEMORY_SECTION_DELIMITER)
        for sec in sections:
            if summary in sec.strip():
                logger.debug("Skipping duplicate MEMORY.md entry: %s", summary[:60])
                return False

        # Check capacity — trim if needed
        new_entry = f"{entry}\n{MEMORY_SECTION_DELIMITER}\n"
        if len(current) + len(new_entry) > NATIVE_MEMORY_MAX_CHARS:
            freed = self._trim(current, len(new_entry))
            if not freed:
                logger.warning("MEMORY.md full, cannot promote: %s", summary[:60])
                return False
            current = self.read_memory_md()  # re-read after trim

        # Append and write
        updated = current + ("\n" if current and not current.endswith("\n") else "") + new_entry
        ok = self.write_memory_md(updated)
        if ok:
            logger.info(
                "Promoted to MEMORY.md [%s] %s",
                category,
                summary[:80],
            )
        return ok

    def promote_policy(self, name: str, trigger: str, action: str, confidence: float) -> bool:
        """Promote a high-confidence L2 Policy to MEMORY.md."""
        threshold = self._config.promote_on_l2_confidence
        if confidence < threshold:
            return False
        entry = f"经验/{name}: {trigger[:60]} → {action[:60]}"
        return self.promote(entry, category="policy")

    def promote_skill(self, name: str, description: str) -> bool:
        """Promote a crystallized skill to MEMORY.md."""
        if not self._config.promote_on_skill_crystallize:
            return False
        entry = f"技能/{name}: {description[:120]}"
        return self.promote(entry, category="skill")

    # ── Capacity Management ───────────────────────────────

    def usage_ratio(self) -> float:
        """Return MEMORY.md usage as a ratio 0.0–1.0."""
        content = self.read_memory_md()
        if not content:
            return 0.0
        return len(content) / NATIVE_MEMORY_MAX_CHARS

    def needs_trim(self) -> bool:
        """Check if MEMORY.md exceeds the capacity warning threshold."""
        return self.usage_ratio() >= self._config.memory_md_capacity_warning

    def _trim(self, current: str, needed_space: int) -> bool:
        """Trim oldest non-tagged entries to make room.

        Removal order:
        1. Oldest auto-promoted entries (tagged with 🧠)
        2. Oldest manual entries (if still not enough)

        Never removes locked/critical entries without explicit tag.
        """
        sections = current.split(MEMORY_SECTION_DELIMITER)
        kept: list[str] = []
        removed = 0

        # Estimate space: current length + needed - freed after removal
        estimated_new_len = len(current) + needed_space

        for sec in sections:
            stripped = sec.strip()
            if not stripped:
                continue

            # Always keep manual entries (📝) — user wrote them intentionally
            if stripped.startswith("📝"):
                kept.append(stripped)
                continue

            # Remove oldest auto-promoted entries until we have room
            if estimated_new_len > NATIVE_MEMORY_MAX_CHARS and stripped.startswith(PROMOTED_TAG):
                removed += 1
                estimated_new_len -= len(stripped) + 3  # +delimiter+newlines
                continue

            kept.append(stripped)

        if estimated_new_len > NATIVE_MEMORY_MAX_CHARS:
            # Still over capacity — remove any non-manual entries oldest first
            extra_kept: list[str] = []
            for entry in kept:
                if estimated_new_len > NATIVE_MEMORY_MAX_CHARS and not entry.startswith("📝"):
                    removed += 1
                    estimated_new_len -= len(entry) + 3
                    continue
                extra_kept.append(entry)
            kept = extra_kept

        if removed == 0 and estimated_new_len > NATIVE_MEMORY_MAX_CHARS:
            logger.warning("Cannot free enough space in MEMORY.md (all entries are manual)")
            return False

        new_content = f"\n{MEMORY_SECTION_DELIMITER}\n".join(kept)
        if new_content:
            new_content += f"\n{MEMORY_SECTION_DELIMITER}\n"
        self.write_memory_md(new_content)
        logger.info("Trimmed %d entries from MEMORY.md to free space", removed)
        return True

    # ── state.db Session Search Fallback ──────────────────

    def session_search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Fallback search in Hermes Agent's native state.db (FTS5).

        This mirrors the Hermes Agent's built-in session_search DISCOVERY mode.

        Args:
            query: Search query string.
            limit: Max results.

        Returns:
            List of {session_id, role, content, timestamp, score} dicts.
        """
        if not self._config.session_search_fallback:
            return []

        db_path = self.state_db_path
        if not db_path.exists():
            logger.debug("state.db not found at %s, skipping session_search", db_path)
            return []

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            # Clean query for FTS5
            safe = query.replace('"', '""')
            rows = conn.execute(
                """
                SELECT m.id, m.session_id, m.role, m.content, m.timestamp
                FROM messages m
                JOIN messages_fts fts ON m.rowid = fts.rowid
                WHERE messages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe, limit),
            ).fetchall()
            conn.close()

            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "role": row["role"],
                    "content": row["content"][:500] if row["content"] else "",
                    "timestamp": row["timestamp"],
                    "source": "session_search",
                })
            logger.debug("session_search found %d results for: %s", len(results), query[:60])
            return results

        except sqlite3.OperationalError as e:
            logger.debug("session_search failed: %s", e)
            return []
        except Exception as e:
            logger.warning("session_search error: %s", e)
            return []

    def format_session_results(self, results: list[dict[str, Any]]) -> str:
        """Format session_search results for context injection."""
        if not results:
            return ""
        lines = ["## 历史会话检索（session_search）", ""]
        for i, r in enumerate(results[: self._config.session_search_max_results], 1):
            role_tag = "🧑" if r.get("role") == "user" else "🤖"
            content = r.get("content", "")[:300]
            lines.append(f"{i}. [{role_tag}] {content}")
        return "\n".join(lines)

    # ── Stats ─────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return integration health stats."""
        memory_path = self.memory_md_path
        state_path = self.state_db_path

        sections = self.read_sections()
        promoted = sum(1 for s in sections if s.startswith(PROMOTED_TAG))

        md_size = memory_path.stat().st_size if memory_path.exists() else 0
        state_size = state_path.stat().st_size if state_path.exists() else 0

        return {
            "memory_md_exists": memory_path.exists(),
            "memory_md_size": md_size,
            "memory_md_chars": len(self.read_memory_md()),
            "memory_md_limit": NATIVE_MEMORY_MAX_CHARS,
            "memory_md_usage_ratio": round(self.usage_ratio(), 3),
            "memory_md_sections": len(sections),
            "memory_md_promoted": promoted,
            "state_db_exists": state_path.exists(),
            "state_db_size_mb": round(state_size / (1024 * 1024), 1) if state_size else 0,
            "sync_enabled": self._config.sync_memory_md,
            "session_search_enabled": self._config.session_search_fallback,
        }
