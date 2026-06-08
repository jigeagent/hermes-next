"""Decision Repair — write user corrections as Policy @repair blocks.

v0.4.0-alpha: keyword + time-window matching
v0.4.0-beta: embedding semantic matching + scope inference

When a user says "that's wrong" or "do it this way instead",
Decision Repair finds the most relevant Policy and appends a
@repair block with the correction. On subsequent retrievals,
the Policy with repair guidance ranks higher in context injection.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hermes_next.memos.feedback import FeedbackSignal
from hermes_next.memos.types import PolicyRow

logger = logging.getLogger(__name__)


def apply_decision_repair(
    signal: FeedbackSignal,
    policies: list[PolicyRow],
    policy_repo: Any,
) -> Optional[PolicyRow]:
    """Find the most relevant Policy and append a @repair block.

    v0.4.0-alpha: keyword overlap matching between signal.text and
    each policy's trigger_pattern / name / description.

    v0.4.0-beta: will add embedding-based semantic matching.

    Args:
        signal: The negative feedback signal with correction text.
        policies: Active policies to search.
        policy_repo: PolicyRepository for persisting the update.

    Returns:
        The matched PolicyRow if a repair was applied, None otherwise.
    """
    if not signal.text or signal.polarity != "negative":
        return None
    if not policies:
        return None

    text_lower = signal.text.lower()
    text_words = set(text_lower.split())

    # Score each policy by keyword overlap
    scored: list[tuple[PolicyRow, float]] = []
    for p in policies:
        candidates = [
            p.name.lower(),
            p.trigger_pattern.lower(),
            p.description.lower(),
        ]
        overlap = 0
        for candidate in candidates:
            candidate_words = set(candidate.split())
            if candidate_words:
                overlap += len(text_words & candidate_words)
        # Normalize by target length to avoid favouring long entries
        if len(text_words) > 0:
            score = overlap / len(text_words)
        else:
            score = 0
        if score > 0:
            scored.append((p, score))

    if not scored:
        return None

    # Take the best match
    scored.sort(key=lambda x: x[1], reverse=True)
    target = scored[0][0]

    # Determine scope
    scope = _infer_scope(signal.text, target)

    # Build repair entry
    repair_entry = {
        "anti_pattern": [signal.text],
        "preference": [],
        "scope": scope,
        "agent": signal.agent_name,
        "source": f"feedback_{signal.episode_id}",
    }

    # Append to existing repair block
    metadata = dict(target.metadata) if target.metadata else {}
    existing = metadata.get("repair", [])
    if isinstance(existing, list):
        existing.append(repair_entry)
    else:
        existing = [repair_entry]
    metadata["repair"] = existing

    # Persist
    policy_repo.update_metadata(target.id, metadata)

    logger.info(
        "Decision Repair [%s] → policy %s: %s",
        scope,
        target.name,
        signal.text[:80],
    )
    return target


def _infer_scope(text: str, policy: PolicyRow) -> str:
    """Infer whether this repair is global or scene-specific.

    v0.4.0-alpha: heuristic based on policy name and trigger.
    If the policy name or trigger contains project-specific keywords,
    default to scene-specific. Otherwise global.
    """
    text_lower = text.lower()
    name_lower = policy.name.lower()

    # Scene-specific markers in policy name
    scene_markers = ["project", "scene", "task", "case", "mission"]
    for marker in scene_markers:
        if marker in name_lower:
            return "scene-specific"

    # If user text mentions a specific project/context
    project_markers = ["大国中医", "this project", "this scene", "当前项目"]
    for marker in project_markers:
        if marker in text_lower:
            return "scene-specific"

    return "global"
