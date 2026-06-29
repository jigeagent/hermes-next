"""
Cognitive pipeline background runner for hermes-next.

Reads accumulated traces from the local cache database, runs the full
cognitive pipeline (Reward -> L2 Induction -> Skill Crystallization),
and persists results (policies, skills) back to the cache and optionally
to OpenViking.

Usage:
    python -m hermes_next.promote                         # full run
    python -m hermes_next.promote --dry-run                # preview only
    python -m hermes_next.promote --backfill               # estimate missing rewards
    python -m hermes_next.promote --cache-path <path>      # custom cache

Designed to be triggered by the OpenViking watchdog or a scheduled task,
analogous to ``cc-star promote``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.policies import PolicyRepository
from hermes_next.cache.schema import ensure_schema
from hermes_next.cache.traces import TraceRepository
from hermes_next.config import (
    CacheConfig,
    CognitiveConfig,
    HermesNextConfig,
    OpenVikingConfig,
)
from hermes_next.memos.pipeline import (
    CognitivePipeline,
    CognitivePipelineConfig,
    PipelineStage,
)
from hermes_next.memos.reward import OutcomeSignal, RewardEngine
from hermes_next.memos.types import TraceRow

logger = logging.getLogger(__name__)


# ---- Helpers ------------------------------------------------------------


_ERR_PATTERNS = [
    "error", "Error", "traceback", "Traceback", "failed", "Failed",
    "exception", "Exception", "timeout", "Timeout",
    "cannot access local variable",
    "UnboundLocalError",
    "NameError",
    "AttributeError",
    "TypeError",
    "KeyError",
    "IndexError",
]


def _resolve_path(raw: str) -> str:
    """Expand ~ and env vars in a path string."""
    return os.path.expanduser(os.path.expandvars(raw))


def _open_cache(cache_path: str) -> CacheConnection:
    """Open local cache DB (create schema if missing)."""
    path = _resolve_path(cache_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cache = CacheConnection(path)
    ensure_schema(cache)
    return cache


def _load_traces(repo: TraceRepository) -> list:
    """Load all recent traces from cache.

    Returns TraceRow namedtuples as-is -- the pipeline expects them.
    """
    return repo.list_recent(limit=5000)


def _has_reward(traces: list[TraceRow]) -> bool:
    """Check if any trace in a group already has a non-zero reward."""
    for t in traces:
        if t.reward is not None and t.reward != 0:
            return True
    return False


def _estimate_session_success(traces: list[TraceRow]) -> bool:
    """Heuristic: was the session likely a success or failure?

    Returns True (success) or False (failure).
    """
    assistant_texts = [t.assistant_content or "" for t in traces]
    combined = " ".join(assistant_texts)

    # Check for error indicators
    error_hits = sum(1 for p in _ERR_PATTERNS if re.search(p, combined))
    if error_hits >= 3:
        return False

    # Check for meaningful content (both sides of conversation)
    user_texts = [t.user_content or "" for t in traces]
    meaningful_user = sum(1 for u in user_texts if len(u) > 50)
    meaningful_assistant = sum(1 for a in assistant_texts if len(a) > 50)

    # Sessions with substance on both sides are likely successes
    return meaningful_user >= 1 and meaningful_assistant >= 1


# ---- Backfill -----------------------------------------------------------


def _backfill_rewards(
    traces: list[TraceRow],
    cache_path: str = "",
    dry_run: bool = False,
) -> tuple[list[TraceRow], dict]:
    """Group traces by session, estimate reward for un-rewarded sessions.

    Persists rewards to DB when not dry-run.
    Returns (updated_traces, stats).
    """
    # Group by session_id
    sessions: dict[str, list[TraceRow]] = defaultdict(list)
    for t in traces:
        sessions[t.session_id].append(t)

    stats = {
        "total_sessions": len(sessions),
        "already_rewarded": 0,
        "backfilled_success": 0,
        "backfilled_failure": 0,
        "skipped_no_content": 0,
    }

    engine = RewardEngine()
    repo: TraceRepository | None = None

    # Open DB for writing rewards back
    if not dry_run and cache_path:
        try:
            cache = _open_cache(cache_path)
            repo = TraceRepository(cache)
        except Exception:
            pass

    for session_id, session_traces in sessions.items():
        if _has_reward(session_traces):
            stats["already_rewarded"] += 1
            continue

        # Heuristic: success or failure?
        is_success = _estimate_session_success(session_traces)
        signal = OutcomeSignal.TASK_SUCCESS if is_success else OutcomeSignal.TASK_FAILURE

        if is_success:
            stats["backfilled_success"] += 1
        else:
            # Double-check: if there's really no content, skip
            has_any_content = any(
                (t.user_content or "").strip() or (t.assistant_content or "").strip()
                for t in session_traces
            )
            if not has_any_content:
                stats["skipped_no_content"] += 1
                continue
            stats["backfilled_failure"] += 1

        # Apply reward signal and persist
        try:
            updated_traces = engine.apply_outcome(session_traces, signal)
            if repo and not dry_run:
                for t in updated_traces:
                    if t.id and (t.reward is not None):
                        repo.update_reward(t.id, t.reward)
        except Exception as e:
            logger.warning("Backfill reward failed for session %s: %s", session_id, e)

    return traces, stats


# ---- Pipeline Runner ----------------------------------------------------


def run_pipeline(
    cache_path: str,
    ov_base_url: str = "http://localhost:1933",
    dry_run: bool = False,
    enable_l3: bool = False,
    enable_skill: bool = False,
    backfill: bool = False,
) -> dict[str, object]:
    """Open cache, load traces, run cognitive pipeline, persist results.

    Returns a summary dict (JSON-serialisable) for logging or display.
    """
    result: dict[str, object] = {
        "action": "promote",
        "dry_run": dry_run,
        "backfill": backfill,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_path": _resolve_path(cache_path),
    }

    # ---- Open cache ------------------------------------------------------
    try:
        cache = _open_cache(cache_path)
        repo = TraceRepository(cache)
        total = repo.count()
        result["total_traces"] = total
    except Exception as e:
        result["status"] = f"error: {e}"
        return result

    if total == 0:
        result["status"] = "no_traces"
        return result

    # ---- Load traces -----------------------------------------------------
    traces = _load_traces(repo)
    result["loaded_traces"] = len(traces)

    if not traces:
        result["status"] = "no_traces"
        return result

    # ---- Backfill rewards if requested ----------------------------------
    if backfill:
        traces, backfill_stats = _backfill_rewards(
            traces, cache_path=cache_path, dry_run=dry_run,
        )
        result["backfill_stats"] = backfill_stats

    # ---- Build pipeline --------------------------------------------------
    enabled_stages = {PipelineStage.L1_CAPTURE, PipelineStage.REWARD, PipelineStage.L2_INDUCTION}
    if enable_l3:
        enabled_stages.add(PipelineStage.L3_WORLD_MODEL)
    if enable_skill:
        enabled_stages.add(PipelineStage.SKILL_CRYSTALLIZATION)

    pipeline_config = CognitivePipelineConfig(enabled_stages=enabled_stages)

    # ---- Process traces per-session -------------------------------------
    try:
        sessions: dict[str, list[TraceRow]] = defaultdict(list)
        for t in traces:
            sessions[t.session_id].append(t)

        total_policies: list = []
        per_session: dict[str, dict] = {}

        for session_id, session_traces in sessions.items():
            session_pipeline = CognitivePipeline(config=pipeline_config)
            session_results = session_pipeline.process_session_end(
                traces=session_traces,
                session_success=True,
            )
            new_p = session_results.get("new_policies", [])
            if new_p:
                total_policies.extend(new_p)
            per_session[session_id] = {
                "traces": len(session_traces),
                "policies": len(new_p),
            }

        result["pipeline"] = {
            "sessions_processed": len(sessions),
            "per_session": per_session,
            "reward": {"applied": bool(backfill)},
            "induction": {"new_count": len(total_policies), "total": len(total_policies)},
        }
        result["new_policies_count"] = len(total_policies)
        new_policies = total_policies
    except Exception as e:
        result["status"] = f"pipeline_error: {e}"
        return result

    # ---- Persist ---------------------------------------------------------
    if new_policies and not dry_run:
        try:
            policy_repo = PolicyRepository(cache)
            for p in new_policies:
                policy_repo.insert(p)
            result["persisted_policies"] = len(new_policies)

            # Also attempt OV resource push
            try:
                from hermes_next.ov.client import OpenVikingClient
                ov = OpenVikingClient(base_url=ov_base_url)
                for p in new_policies:
                    ov.upsert_resource(
                        uri=f"viking://policies/{p.name}",
                        content=p.action_template,
                        metadata={
                            "type": "policy",
                            "trigger": p.trigger_pattern,
                            "confidence": p.confidence,
                        },
                    )
                result["ov_synced"] = len(new_policies)
            except Exception as ov_err:
                result["ov_sync_warning"] = str(ov_err)
        except Exception as e:
            result["persist_error"] = str(e)
            result["status"] = "persist_failed"
            return result

    result["status"] = "ok"
    cache.close_all()
    return result


# ---- Embedding Backfill --------------------------------------------------


def backfill_embeddings(
    cache_path: str = "",
    dry_run: bool = False,
    batch_size: int = 32,
) -> dict:
    """Backfill missing embeddings for traces without them (or wrong dimension)."""
    import time, json
    from hermes_next.cache.vector import EmbeddingEngine

    if not cache_path:
        cache_path = _resolve_path("~/.hermes-next/cache.db")

    try:
        from hermes_next.cache.connection import CacheConnection
        from hermes_next.cache.traces import TraceRepository
        cache = CacheConnection(_resolve_path(cache_path))
        ensure_schema(cache)
        repo = TraceRepository(cache)
    except Exception as e:
        return {"error": f"cannot open cache: {e}"}

    all_traces = repo.list_recent(limit=50000)
    target_dim = 384
    pending = [
        t for t in all_traces
        if t.embedding is None or len(t.embedding) != target_dim
    ]
    total = len(all_traces)
    pending_count = len(pending)

    if dry_run:
        cache.close_all()
        return {"processed": 0, "total": total, "pending": pending_count, "dry_run": True}

    engine = EmbeddingEngine()
    processed = 0
    t0 = time.time()

    for i in range(0, pending_count, batch_size):
        batch = pending[i:i + batch_size]
        texts = [
            (t.user_content or "") + " " + (t.assistant_content or "")
            for t in batch
        ]
        embeddings = engine.embed(texts)
        for t, emb in zip(batch, embeddings):
            emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
            repo.update_embedding(t.id, json.dumps(emb_list))
            processed += 1

    cache.close_all()
    elapsed = time.time() - t0
    return {"processed": processed, "total": total, "pending": pending_count, "elapsed_seconds": round(elapsed, 1)}


# ---- CLI -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hermes-next cognitive pipeline runner (background promote)",
    )
    parser.add_argument(
        "--cache-path",
        default="~/.hermes-next/cache.db",
        help="Path to cache.db (default: ~/.hermes-next/cache.db)",
    )
    parser.add_argument(
        "--ov-url",
        default="http://localhost:1933",
        help="OpenViking base URL (default: http://localhost:1933)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing anything",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Estimate missing rewards via heuristics before induction",
    )
    parser.add_argument(
        "--backfill-embeddings",
        action="store_true",
        help="Backfill missing embeddings for traces without them (or wrong dimension)",
    )
    parser.add_argument(
        "--enable-l3",
        action="store_true",
        help="Enable L3 world model clustering (opt-in, default: off)",
    )
    parser.add_argument(
        "--enable-skill",
        action="store_true",
        help="Enable skill crystallization (opt-in, default: off)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Handle backfill-embeddings separately (standalone CLI)
    if args.backfill_embeddings:
        logger.info("Running standalone backfill (pipeline skipped)")
        result = backfill_embeddings(
            cache_path=args.cache_path,
            dry_run=args.dry_run,
        )
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False, default=str)
        print()
        sys.exit(0 if result.get("error") is None else 1)

    result = run_pipeline(
        cache_path=args.cache_path,
        ov_base_url=args.ov_url,
        dry_run=args.dry_run,
        backfill=args.backfill,
        enable_l3=args.enable_l3,
        enable_skill=args.enable_skill,
    )

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False, default=str)
    print()

    if result.get("status") == "ok":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
