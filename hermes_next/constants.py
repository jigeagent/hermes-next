"""Shared constants for hermes-next.

Centralises values that must remain consistent across the codebase
to prevent accidental drift during upgrades.
"""

# ── OpenViking resource paths ──────────────────────────────
# These MUST stay in sync across all agents for cross-agent sharing.
# Do NOT change without updating 好妹/好二妹/灵儿 simultaneously.

OV_TRACE_PATH = "viking://resources/memory/traces/{id}.json"
"""Unified trace path for cross-agent sharing.
All agents (好妹/好二妹/灵儿) write and read traces from this path.
Do NOT add agent-specific prefixes — that would break shared access."""

OV_POLICY_PATH = "viking://resources/memory/policies/{id}.json"
"""Unified policy path."""

OV_SKILL_PATH = "viking://resources/memory/skills/{name}.md"
"""Unified skill path."""

# ── Tag conventions ────────────────────────────────────────
# sync_turn(tags=...) should use one of these three categories.
# Custom tags are allowed but won't be indexed for cross-agent search.

TAG_CHAT = "chat"
"""日常对话 — casual conversation, Q&A, general discussion."""

TAG_DECISION = "decision"
"""决策/架构/协议 — architectural decisions, protocols, design choices."""

TAG_BUGFIX = "bugfix"
"""故障/修复/教训 — bugs, fixes, lessons learned."""

TAG_CATEGORIES = [TAG_CHAT, TAG_DECISION, TAG_BUGFIX]
"""The three canonical tag categories."""

# ── Access tracking ───────────────────────────────────────

TRACK_ACCESS = True
"""When True, every prefetch/search/get call increments access_count
and updates last_accessed on the target trace. Required for
forgetting-curve computation and OV sharing relevance ranking."""
