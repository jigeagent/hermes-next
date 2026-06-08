"""Feedback signal model — user feedback on memory traces.

v0.4.0: Feedback 闭环基础版
- FeedbackSignal dataclass with agent isolation
- Decision Repair writes @repair blocks to matched policies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional


FeedbackPolarity = Literal["positive", "negative", "neutral"]
FeedbackSource = Literal["user", "implicit", "delegation"]


@dataclass
class FeedbackSignal:
    """A single feedback event from a user or implicit signal.

    v0.4 isolation rule: each agent's feedback only affects its own
    policies — no cross-agent merging until v0.5 Hub support.
    """

    episode_id: str
    trace_id: Optional[str] = None
    polarity: FeedbackPolarity = "neutral"
    magnitude: float = 1.0
    """Strength of the signal. 0.0–1.0. magnitude < 0.3 = weak feedback
    that only writes @repair without triggering reward recomputation."""

    text: Optional[str] = None
    """Free-text correction from the user. Required for Decision Repair."""

    source: FeedbackSource = "user"
    agent_name: str = "default"
    """Source agent identifier. Multi-agent isolation: each agent's
    feedback only updates its own policies."""

    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
