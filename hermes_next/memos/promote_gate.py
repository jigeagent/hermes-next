"""Validation gate for hermes-next memory promotion.

移植自 cc-star v0.4.0 promote_gate.py（源自 SkillOpt evaluate_gate()）。
在 policy induction 晋升时加上验证门控：
- candidate 必须优于 current 才接受
- 追踪 current_score + best_score 双线
- reject 不丢信息，记入拒绝缓冲
- promote 前置新鲜度检查（好二妹建议）

核心差异：
  cc-star gate 基于 _score_trace() 打分 + FTS5 相似度
  hermes-next gate 基于 policy confidence + 激活频率
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional


GateAction = Literal["accept_new_best", "accept", "reject"]
GateMetric = Literal["hard", "soft", "mixed"]

_GATE_STATE_FILE = "gate_state.json"


@dataclass
class GateResult:
    """Immutable outcome of the promotion gate."""

    action: GateAction
    candidate_id: str
    candidate_score: float
    current_score: float
    best_score: float
    best_id: str
    best_step: int


@dataclass
class GateState:
    """Persistent gate state — tracks current/best across promote runs."""

    current_score: float = 0.0
    current_id: str = ""
    best_score: float = 0.0
    best_id: str = ""
    best_step: int = 0
    accept_count: int = 0
    reject_count: int = 0

    def to_dict(self) -> dict:
        return {
            "current_score": self.current_score,
            "current_id": self.current_id,
            "best_score": self.best_score,
            "best_id": self.best_id,
            "best_step": self.best_step,
            "accept_count": self.accept_count,
            "reject_count": self.reject_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GateState":
        return cls(
            current_score=float(d.get("current_score", 0.0)),
            current_id=str(d.get("current_id", "")),
            best_score=float(d.get("best_score", 0.0)),
            best_id=str(d.get("best_id", "")),
            best_step=int(d.get("best_step", 0)),
            accept_count=int(d.get("accept_count", 0)),
            reject_count=int(d.get("reject_count", 0)),
        )


# ── Pure gate function (from SkillOpt evaluate_gate) ──


def select_gate_score(
    hard: float,
    soft: float,
    metric: GateMetric = "mixed",
    mixed_weight: float = 0.3,
) -> float:
    """Project (hard, soft) onto a single comparison score.

    - hard: policy confidence (0-1)
    - soft: activation frequency or trace coverage (0-1 normalized)
    - mixed: weighted fusion
    """
    if metric == "hard":
        return float(hard)
    if metric == "soft":
        return float(soft)
    if metric == "mixed":
        w = max(0.0, min(1.0, float(mixed_weight)))
        return (1.0 - w) * float(hard) + w * float(soft)
    raise ValueError(f"unknown gate metric {metric!r}")


def evaluate_gate(
    candidate_id: str,
    candidate_hard: float,
    current_state: GateState,
    global_step: int,
    *,
    candidate_soft: float = 0.0,
    metric: GateMetric = "mixed",
    mixed_weight: float = 0.3,
) -> GateResult:
    """Gate decision: compare candidate score to current/best.

    Args:
        candidate_id: Policy ID being evaluated.
        candidate_hard: Policy confidence score (0-1).
        current_state: Current gate state.
        global_step: Promote run counter.
        candidate_soft: Soft score (activation frequency, optional).

    Returns:
        GateResult with action and updated state.
    """
    cand_score = select_gate_score(
        candidate_hard, candidate_soft, metric, mixed_weight,
    )

    if cand_score > current_state.current_score:
        if cand_score > current_state.best_score:
            return GateResult(
                action="accept_new_best",
                candidate_id=candidate_id,
                candidate_score=cand_score,
                current_score=cand_score,
                best_score=cand_score,
                best_id=candidate_id,
                best_step=global_step,
            )
        return GateResult(
            action="accept",
            candidate_id=candidate_id,
            candidate_score=cand_score,
            current_score=cand_score,
            best_score=current_state.best_score,
            best_id=current_state.best_id,
            best_step=current_state.best_step,
        )
    return GateResult(
        action="reject",
        candidate_id=candidate_id,
        candidate_score=cand_score,
        current_score=current_state.current_score,
        best_score=current_state.best_score,
        best_id=current_state.best_id,
        best_step=current_state.best_step,
    )


# ── State persistence ──


def _data_dir() -> Path:
    """Resolve data directory."""
    raw = os.environ.get("HERMES_NEXT_CACHE_PATH", "")
    if raw:
        return Path(raw).parent
    return Path.home() / ".hermes-next"


def gate_state_path() -> Path:
    """Path to gate state JSON file."""
    return _data_dir() / _GATE_STATE_FILE


def load_gate_state() -> GateState:
    """Load persistent gate state."""
    path = gate_state_path()
    if path.is_file():
        try:
            return GateState.from_dict(json.loads(path.read_text("utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return GateState()


def save_gate_state(state: GateState) -> None:
    """Save gate state to disk."""
    path = gate_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), "utf-8")


def update_gate_state(state: GateState, result: GateResult) -> GateState:
    """Update gate state from a gate result."""
    state.current_score = result.current_score
    state.current_id = result.candidate_id if result.action != "reject" else state.current_id
    if result.action == "accept_new_best":
        state.best_score = result.best_score
        state.best_id = result.best_id
        state.best_step = result.best_step
    if result.action in ("accept", "accept_new_best"):
        state.accept_count += 1
    else:
        state.reject_count += 1
    return state


# ── Trace freshness check (好二妹建议) ──


FreshnessResult = dict
"""{'fresh': bool, 'reason': str, 'last_ts': str, 'age_hours': float}"""


def check_trace_freshness(cache_conn: sqlite3.Connection, max_age_hours: int = 24) -> dict:
    """Check if traces table has recent writes.

    Before running promote+gate, verify traces are being fed in.
    If the latest trace is older than max_age_hours, skip promote
    to avoid wasting compute on stale data.

    Args:
        cache_conn: SQLite connection to hermes-next cache.db
        max_age_hours: Max allowed age of latest trace. Default 24h.

    Returns:
        {'fresh': True/False, 'reason': str, 'last_ts': str, 'age_hours': float}
    """
    try:
        row = cache_conn.execute(
            "SELECT MAX(created_at) as last_ts FROM traces"
        ).fetchone()
    except Exception as e:
        return {"fresh": False, "reason": f"db_error: {e}", "last_ts": "", "age_hours": -1}

    if not row or not row["last_ts"]:
        return {"fresh": False, "reason": "no_traces", "last_ts": "", "age_hours": -1}

    try:
        last = datetime.fromisoformat(str(row["last_ts"]))
    except (ValueError, TypeError):
        return {"fresh": False, "reason": "invalid_timestamp", "last_ts": str(row["last_ts"]), "age_hours": -1}

    age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
    fresh = age_hours <= max_age_hours

    return {
        "fresh": fresh,
        "reason": f"last trace {age_hours:.0f}h ago",
        "last_ts": str(row["last_ts"]),
        "age_hours": round(age_hours, 1),
    }


# ── Reject buffer ──


_REJECT_LOG = "reject_log.jsonl"


def reject_log_path() -> Path:
    return _data_dir() / _REJECT_LOG


def log_rejection(result: GateResult, reason: str = "") -> None:
    """Log a rejected candidate for later analysis."""
    path = reject_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "candidate_id": result.candidate_id,
        "candidate_score": result.candidate_score,
        "current_score": result.current_score,
        "best_score": result.best_score,
        "reason": reason,
        "rejected_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(str(path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def load_recent_rejections(n: int = 5) -> list[dict]:
    """Load most recent rejection records."""
    path = reject_log_path()
    if not path.is_file():
        return []
    try:
        lines = path.read_text("utf-8").strip().split("\n")
        records = []
        for line in lines:
            if line.strip():
                records.append(json.loads(line))
        return records[-n:]
    except (OSError, json.JSONDecodeError):
        return []
