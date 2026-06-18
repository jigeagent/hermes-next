"""HermesNextProvider — MemoryProvider implementation for Hermes Agent."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.feedback import FeedbackRepository
from hermes_next.cache.lifecycle import LifecycleManager, LifecycleStats
from hermes_next.cache.policies import PolicyRepository
from hermes_next.cache.schema import ensure_schema
from hermes_next.cache.session_state import SessionState, SessionStateRepository
from hermes_next.config import HermesNextConfig
from hermes_next.integration.native import NativeMemoryClient, NativeMemoryConfig
from hermes_next.memos.capture import capture_trace
from hermes_next.memos.feedback import FeedbackSignal
from hermes_next.memos.pipeline import (
    CognitivePipeline,
    CognitivePipelineConfig,
    PipelineStage,
)
from hermes_next.memos.repair import apply_decision_repair
from hermes_next.memos.retrieval import format_results, retrieve_semantic, retrieve_timeline
from hermes_next.memos.reward import OutcomeSignal
from hermes_next.memos.types import TraceRow
from hermes_next.ov.client import OpenVikingClient
from hermes_next.ov.session import OVSession
from hermes_next.retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


class HermesNextProvider:
    """Hermes Agent memory provider backed by OpenViking + MemOS cognitive engine."""

    def __init__(self, config: Optional[HermesNextConfig] = None):
        self._config = config or HermesNextConfig()
        self._client: Optional[OpenVikingClient] = None
        self._cache: Optional[CacheConnection] = None
        self._retrieval: Optional[RetrievalPipeline] = None
        self._pipeline: Optional[CognitivePipeline] = None
        self._lifecycle: Optional[LifecycleManager] = None
        self._native: Optional[NativeMemoryClient] = None
        self._session: Optional[OVSession] = None
        self._agent_name: str = self._config.agent.name
        self._turn_index: int = 0
        self._initialized: bool = False
        # Session-scoped trace accumulator for cognitive pipeline
        self._session_traces: list[TraceRow] = []
        # Embedding cache: {text: embedding}
        self._embed_cache: dict[str, list[float]] = {}
        self._embed_cache_max: int = 256
        # v0.4: Feedback & session state repos (lazy init)
        self._feedback_repo: Optional[FeedbackRepository] = None
        self._session_repo: Optional[SessionStateRepository] = None
        self._policy_repo: Optional[PolicyRepository] = None

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Cached embedding lookup with DashScope fallback."""
        if not self._client:
            return None
        if text in self._embed_cache:
            return self._embed_cache[text]
        if len(self._embed_cache) >= self._embed_cache_max:
            evict = len(self._embed_cache) // 2
            for k in list(self._embed_cache)[:evict]:
                del self._embed_cache[k]
        emb = self._client.embed(text, dashscope_config=self._config.dashscope)
        if emb:
            self._embed_cache[text] = emb
        return emb

    # ── Required MemoryProvider interface ───────────────────

    @property
    def name(self) -> str:
        return "hermes-next"

    def is_available(self) -> bool:
        """Check if the OpenViking server is reachable."""
        if not self._client:
            return False
        return self._client.health()

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Establish OpenViking connection, open local cache, and initialise retrieval pipeline."""
        cfg = self._config
        self._client = OpenVikingClient(
            base_url=cfg.openviking.base_url,
            api_key=cfg.openviking.api_key,
            timeout=cfg.openviking.timeout,
            max_retries=cfg.openviking.max_retries,
        )
        # Set DashScope fallback for embedding
        if cfg.dashscope.enabled:
            self._client.set_dashscope_config(cfg.dashscope)
        self._agent_name = kwargs.get("agent_name", cfg.agent.name)

        # Local SQLite cache for FTS5 + offline search
        self._cache = CacheConnection(cfg.cache.path, wal_mode=cfg.cache.wal_mode)
        ensure_schema(self._cache)

        # Fusion retrieval pipeline (semantic + FTS5 + timeline + RRF + MMR)
        self._retrieval = RetrievalPipeline(
            ov_client=self._client,
            cache=self._cache,
            config=cfg.retrieval,
        )

        # Memory lifecycle manager (archival, decay, cleanup)
        self._lifecycle = LifecycleManager(
            cache=self._cache,
            config=cfg.lifecycle,  # type: ignore[arg-type]
        )
        # Wire MEMORY.md auto-trim into lifecycle cleanup
        orig_cleanup = self._lifecycle.run_cleanup

        def _wrapped_cleanup() -> LifecycleStats:
            return orig_cleanup(trim_callback=self._auto_trim_memory_md)

        self._lifecycle.run_cleanup = _wrapped_cleanup  # type: ignore[method-assign]

        # Native Hermes Agent memory bridge (MEMORY.md sync + session_search)
        self._native = NativeMemoryClient(
            config=NativeMemoryConfig(
                sync_memory_md=cfg.integration.sync_memory_md,
                promote_on_l2_confidence=cfg.integration.promote_on_l2_confidence,
                promote_on_skill_crystallize=cfg.integration.promote_on_skill_crystallize,
                session_search_fallback=cfg.integration.session_search_fallback,
                session_search_max_results=cfg.integration.session_search_max_results,
            ),
        )

        # Run initial MEMORY.md auto-trim check
        self._auto_trim_memory_md()

        # Cognitive pipeline (L1 → Reward → L2 → L3 → Skill)
        pipeline_enabled_stages = {PipelineStage.L1_CAPTURE}
        if cfg.cognitive.auto_reward_on_session_end:
            pipeline_enabled_stages.add(PipelineStage.REWARD)
        if cfg.cognitive.enable_l2_induction:
            pipeline_enabled_stages.add(PipelineStage.L2_INDUCTION)
        if cfg.cognitive.enable_l3_world_model:
            pipeline_enabled_stages.add(PipelineStage.L3_WORLD_MODEL)
        if cfg.cognitive.enable_skill_crystallization:
            pipeline_enabled_stages.add(PipelineStage.SKILL_CRYSTALLIZATION)
        self._pipeline = CognitivePipeline(
            config=CognitivePipelineConfig(enabled_stages=pipeline_enabled_stages),
        )

        # v0.4: Feedback + session state repos
        from hermes_next.cache.feedback import FeedbackRepository
        from hermes_next.cache.policies import PolicyRepository
        from hermes_next.cache.session_state import SessionStateRepository

        self._feedback_repo = FeedbackRepository(self._cache)
        self._session_repo = SessionStateRepository(self._cache)
        self._policy_repo = PolicyRepository(self._cache)

        # v0.4: Session state tracking for crash recovery
        self._session_repo.upsert(SessionState(
            session_id=session_id,
            agent_name=self._agent_name,
            turn_index=0,
            status="open",
        ))

        # v0.4: Recover orphan sessions from previous crashes
        self._recover_orphan_sessions()

        self._session = OVSession(
            client=self._client,
            session_id=session_id,
            agent_name=self._agent_name,
        )
        self._turn_index = 0
        self._session_traces = []
        self._initialized = True
        logger.info(
            "HermesNextProvider initialized (session=%s, agent=%s, pipeline=%s)",
            session_id,
            self._agent_name,
            [s.value for s in pipeline_enabled_stages],
        )

    def prefetch(self, query: str, *, session_id: str) -> str:
        """Fusion retrieval: OpenViking semantic + local FTS5 + timeline,
        fused via RRF, boosted by recency, diversified by MMR.

        Falls through to Hermes Agent native session_search (state.db FTS5)
        when the pipeline returns no results (v0.3.1+).
        """
        if not self._client or not self._retrieval:
            return ""

        agent = self._agent_name
        query_embedding = self._get_embedding(query)

        results = self._retrieval.retrieve(
            query=query,
            agent=agent,
            query_embedding=query_embedding,
        )

        # Access tracking: increment access_count for each hit
        if results and self._cache:
            from hermes_next.cache.traces import TraceRepository
            repo = TraceRepository(self._cache)
            for r in results:
                trace_id = r.get("id") or r.get("trace_id")
                if trace_id:
                    repo.mark_accessed(trace_id)

        formatted = format_results(results)

        # Fall through to Hermes Agent native session_search when pipeline returns empty
        if not formatted and self._native:
            session_results = self._native.session_search(
                query=query,
                limit=self._config.integration.session_search_max_results,
            )
            if session_results:
                formatted = self._native.format_session_results(session_results)
                logger.debug(
                    "session_search fallback: %d results for '%s'",
                    len(session_results), query[:60],
                )

        return formatted

    def sync_turn(  # noqa: PLR0913
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        """Record a single interaction turn as an L1 Trace and feed into cognitive pipeline."""
        if not self._client:
            logger.warning("Provider not initialized, skipping turn capture")
            return

        self._turn_index += 1
        trace = capture_trace(
            client=self._client,
            session_id=session_id,
            turn_index=self._turn_index,
            user_content=user_content,
            assistant_content=assistant_content,
            agent_name=self._agent_name,
            tags=tags,
            metadata={
                **(metadata or {}),
                "source": "hermes-next",
            },
        )

        # Persist to local SQLite cache for FTS5 + offline retrieval
        if trace and self._cache:
            from hermes_next.cache.traces import TraceRepository

            repo = TraceRepository(self._cache)
            repo.insert(trace)

            # Trigger lifecycle management (archival/decay if threshold reached)
            if self._lifecycle:
                self._lifecycle.on_trace_inserted()

        # Feed into cognitive pipeline (L1 capture stage)
        if trace and self._pipeline:
            self._pipeline.process_trace(trace)
            self._session_traces.append(trace)

        # v0.4: Update session state for crash recovery
        if self._session_repo:
            self._session_repo.touch(session_id, turn_index=self._turn_index)

    # ── v0.4: Feedback Loop ─────────────────────────────

    def submit_feedback(
        self,
        polarity: str,
        text: Optional[str] = None,
        trace_id: Optional[str] = None,
        magnitude: float = 1.0,
        *,
        episode_id: str,
    ) -> str:
        """Submit user feedback → reward recalculation → L2 re-induction → Decision Repair.

        Agent-facing interface for the feedback tool.

        Args:
            polarity: 'positive' | 'negative' | 'neutral'
            text: Optional correction text (required for Decision Repair)
            trace_id: Optional target trace for the feedback
            magnitude: Signal strength 0.0–1.0. < 0.3 = weak feedback.
            episode_id: Episode to apply feedback to.

        Returns:
            Human-readable result string.
        """
        if not self._config.feedback.enabled:
            return "Feedback is disabled."

        cfg = self._config.feedback
        signal = FeedbackSignal(
            episode_id=episode_id,
            trace_id=trace_id,
            polarity=polarity,  # type: ignore[arg-type]
            magnitude=min(1.0, max(0.0, magnitude)),
            text=text,
            source="user",
            agent_name=self._agent_name,
        )

        # 1. Persist feedback
        self._feedback_repo.insert(signal)

        # 2. Weak feedback (< threshold) → only repair, no reward
        if signal.magnitude < cfg.weak_magnitude_threshold:
            if signal.polarity == "negative" and signal.text:
                self._apply_decision_repair(signal)
            return f"Weak feedback recorded (magnitude={signal.magnitude}). Repair written."

        # 3. Debounce: same polarity within window
        recent = self._feedback_repo.count_recent(
            episode_id=signal.episode_id,
            polarity=signal.polarity,
            within_seconds=cfg.debounce_seconds,
        )
        if recent > 0:
            return f"Feedback debounced ({recent} similar within {cfg.debounce_seconds}s)."

        # 4. Reward recalculation
        outcome = (
            OutcomeSignal.USER_THUMBS_UP
            if signal.polarity == "positive"
            else OutcomeSignal.USER_THUMBS_DOWN
        )
        if self._pipeline:
            updated_traces = self._pipeline.reward_engine.apply_outcome(
                traces=self._session_traces or self._get_session_traces(signal.episode_id),
                signal=outcome,
                manual_value=signal.magnitude,
            )

            # 5. L2 re-induction (only after ≥threshold negative feedbacks)
            if (
                signal.polarity == "negative"
                and self._feedback_repo.count_negative_since(
                    signal.episode_id, hours=24
                )
                >= cfg.l2_reinduction_min_negatives
            ):
                new_policies = self._pipeline.policy_inducer.batch_induce(
                    traces=updated_traces,
                    existing_policies=self._pipeline.policies,
                )
                for p in new_policies:
                    self._policy_repo.insert(p)
                if new_policies:
                    logger.info("L2 re-induced %d policies from feedback", len(new_policies))

        # 6. Decision Repair
        if signal.polarity == "negative" and signal.text:
            self._apply_decision_repair(signal)

        return (
            f"Feedback recorded ({polarity}, mag={signal.magnitude:.1f}). "
            f"{'Repair applied.' if signal.polarity == 'negative' and signal.text else 'Reward updated.'}"
        )

    def _apply_decision_repair(self, signal: FeedbackSignal) -> None:
        """Find matching policy and append @repair block."""
        policies = self._policy_repo.list_active()
        apply_decision_repair(signal, policies, self._policy_repo)

    def _get_session_traces(self, episode_id: str) -> list:
        """Fallback: load traces from cache when _session_traces is empty."""
        if self._cache:
            from hermes_next.cache.traces import TraceRepository

            return TraceRepository(self._cache).list_by_session(episode_id)
        return []

    # ── v0.4: Crash Recovery ────────────────────────────

    def _recover_orphan_sessions(self) -> None:
        """Scan for stale open sessions and run session_end on them."""
        if not self._session_repo:
            return

        stale_hours = self._config.lifecycle.session_stale_hours
        stale = self._session_repo.list_stale(
            stale_hours=stale_hours,
            agent_name=self._agent_name,
        )
        if not stale:
            return

        logger.info("Recovering %d orphan sessions (stale > %dh)", len(stale), stale_hours)
        from hermes_next.cache.traces import TraceRepository

        trace_repo = TraceRepository(self._cache)
        for session in stale:
            try:
                traces = trace_repo.list_by_session(session.session_id)
                if not traces:
                    self._session_repo.close(session.session_id)
                    continue

                # Run cognitive pipeline on recovered traces
                if self._pipeline:
                    results = self._pipeline.process_session_end(
                        traces=traces,
                        session_success=True,
                    )
                    self._persist_pipeline_results(results)

                self._session_repo.close(session.session_id)
                logger.info(
                    "Recovered session %s: %d traces, %d policies",
                    session.session_id,
                    len(traces),
                    len(results.get("new_policies", [])) if self._pipeline else 0,
                )
            except Exception as e:
                logger.warning("Failed to recover session %s: %s", session.session_id, e)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas exposed to the Hermes agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "memos_search",
                    "description": "Search across all memory traces for relevant context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for relevant memories",
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (max 32)",
                                "default": 8,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memos_get",
                    "description": "Read a specific trace by its ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trace_id": {
                                "type": "string",
                                "description": "ID of the trace to read",
                            },
                        },
                        "required": ["trace_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memos_timeline",
                    "description": "View recent memory activity timeline",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of recent entries",
                                "default": 8,
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memos_status",
                    "description": "View cognitive pipeline status: trace/policy/skill counts and promotion stats",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memos_feedback",
                    "description": "Submit user feedback on a memory trace. "
                                   "Use this when the user corrects or confirms a past interaction. "
                                   "Positive feedback reinforces the behavior; negative feedback "
                                   "triggers Decision Repair and may re-induce policies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "polarity": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"],
                                "description": "Whether this is positive, negative, or neutral feedback",
                            },
                            "text": {
                                "type": "string",
                                "description": "Optional correction text (required for Decision Repair)",
                            },
                            "magnitude": {
                                "type": "number",
                                "description": "Signal strength 0.0-1.0 (default 1.0, <0.3 = weak feedback)",
                                "default": 1.0,
                            },
                        },
                        "required": ["polarity"],
                    },
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any]) -> str:
        """Dispatch a tool call from the agent."""
        if not self._client:
            return "Error: provider not initialized"

        if tool_name == "memos_search":
            query = args.get("query", "")
            k = min(args.get("k", 8), 32)
            results = retrieve_semantic(self._client, query, k=k, agent=self._agent_name)
            # Access tracking
            if results and self._cache:
                from hermes_next.cache.traces import TraceRepository
                repo = TraceRepository(self._cache)
                for r in results:
                    trace_id = r.get("id") or r.get("trace_id")
                    if trace_id:
                        repo.mark_accessed(trace_id)
            return format_results(results) or "No relevant memories found."

        if tool_name == "memos_get":
            trace_id = args.get("trace_id", "")
            uri = f"viking://resources/memory/traces/{trace_id}.json"
            content = self._client.content_read(uri)
            if content:
                try:
                    data = json.loads(content)
                    trace = TraceRow.from_dict(data)
                    # Access tracking
                    if self._cache:
                        from hermes_next.cache.traces import TraceRepository
                        TraceRepository(self._cache).mark_accessed(trace_id)
                    return (
                        f"## Trace: {trace.id}\n"
                        f"**User:** {trace.user_content}\n"
                        f"**Assistant:** {trace.assistant_content}\n"
                        f"**Tags:** {', '.join(trace.tags) if trace.tags else 'none'}\n"
                        f"**Created:** {trace.created_at}"
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    return f"Error parsing trace: {e}"
            return f"Trace {trace_id} not found."

        if tool_name == "memos_timeline":
            limit = min(args.get("limit", 8), 50)
            results = retrieve_timeline(self._client, agent=self._agent_name, limit=limit)
            if not results:
                return "No recent activity."
            lines = ["## Recent Memory Timeline", ""]
            for r in results:
                created = r.get("created_at", r.get("timestamp", "?"))
                summary = r.get("content", r.get("text", ""))[:100]
                lines.append(f"- [{created}] {summary}")
            return "\n".join(lines)

        if tool_name == "memos_status":
            return self._render_pipeline_status()

        if tool_name == "memos_feedback":
            if not self._feedback_repo:
                return "Error: feedback system not initialized"
            return self.submit_feedback(
                polarity=args.get("polarity", "neutral"),
                text=args.get("text"),
                magnitude=float(args.get("magnitude", 1.0)),
                episode_id=args.get("episode_id", self._agent_name),
            )

        return f"Unknown tool: {tool_name}"

    def _render_pipeline_status(self) -> str:
        """Render cognitive pipeline health and promotion status."""
        lines = ["## Cognitive Pipeline Status", ""]

        if not self._pipeline:
            lines.append("Pipeline not initialized.")
            return "\n".join(lines)

        stats = self._pipeline.get_stats()
        lines.append(f"**Session traces accumulated:** {len(self._session_traces)}")
        lines.append(f"**Traces processed:** {stats.get('traces_processed', 0)}")
        lines.append(f"**Policies induced:** {stats.get('policies', 0)}")
        lines.append(f"**Skills crystallized:** {stats.get('skills', 0)}")
        lines.append(f"**Concepts discovered:** {stats.get('concepts', 0)}")
        lines.append(f"**Triples extracted:** {stats.get('triples', 0)}")
        lines.append(f"**Enabled stages:** {', '.join(stats.get('enabled_stages', []))}")
        lines.append("")

        # Add cache and lifecycle stats
        if self._cache:
            try:
                cnt = self._cache.execute(
                    "SELECT COUNT(*) FROM traces"
                ).fetchone()[0]
                lines.append(f"**Total traces in cache:** {cnt}")
            except Exception:
                pass

        if self._lifecycle:
            lc = self._lifecycle.get_stats()
            lines.append("")
            lines.append("---")
            lines.append("### Lifecycle")
            lines.append(f"**Trace retention:** {lc.get('trace_retention_days', '-')}d")
            lines.append(f"**Policy decay rate:** {lc.get('policy_decay_rate', '-')}")
            lines.append(f"**Last cleanup:** {lc.get('last_cleanup', 'never')}")
            lines.append(f"**Traces since cleanup:** {lc.get('traces_since_cleanup', 0)}")
            lines.append(f"**Session stale hours:** {self._config.lifecycle.session_stale_hours}h")

        if self._config.feedback.enabled:
            lines.append("")
            lines.append("---")
            lines.append("### Feedback")
            lines.append("**Feedback enabled:** ✅")
            lines.append(f"**Debounce window:** {self._config.feedback.debounce_seconds}s")
            lines.append(f"**L2 re-induction threshold:** {self._config.feedback.l2_reinduction_min_negatives} negatives")
            if self._cache:
                try:
                    fb_count = self._cache.execute(
                        "SELECT COUNT(*) FROM feedback"
                    ).fetchone()[0]
                    lines.append(f"**Total feedback signals:** {fb_count}")
                except Exception:
                    pass
                try:
                    # Count policies with @repair blocks
                    rows = self._cache.execute(
                        "SELECT metadata FROM policies WHERE metadata LIKE '%repair%'"
                    ).fetchall()
                    repair_count = len(rows)
                    if repair_count > 0:
                        lines.append(f"**已修复反模式:** {repair_count} 个 Policy 含 @repair 块")
                except Exception:
                    pass

        if self._native:
            ni = self._native.get_stats()
            lines.append("")
            lines.append("---")
            lines.append("### 原生集成")
            lines.append(f"**MEMORY.md 同步:** {'✅' if ni.get('sync_enabled') else '❌'}")
            lines.append(f"**session_search 回退:** {'✅' if ni.get('session_search_enabled') else '❌'}")
            if ni.get("memory_md_exists"):
                pct = ni.get("memory_md_usage_ratio", 0) * 100
                lines.append(f"**MEMORY.md 使用率:** {pct:.0f}% ({ni.get('memory_md_sections', 0)} 条)")
                lines.append(f"**已晋升条目:** {ni.get('memory_md_promoted', 0)}")
            if ni.get("state_db_exists"):
                lines.append(f"**state.db 大小:** {ni.get('state_db_size_mb', 0)}MB")

        return "\n".join(lines)

    def _auto_trim_memory_md(self) -> None:
        """Auto-trim MEMORY.md if native integration is active and over capacity."""
        if not self._native:
            return
        try:
            ratio = self._native.usage_ratio()
            if ratio >= self._config.integration.memory_md_capacity_warning:
                removed = self._native.trim_to_fit()
                if removed > 0:
                    logger.info(
                        "MEMORY.md auto-trim: removed %d entries, usage %.0f%%",
                        removed, ratio * 100,
                    )
        except Exception as e:
            logger.debug("MEMORY.md auto-trim check skipped: %s", e)

    def shutdown(self) -> None:
        """Close the OpenViking connection, local cache, and reset pipeline state."""
        if self._client:
            self._client.close()
        if self._cache:
            self._cache.close_all()
        if self._pipeline:
            self._pipeline.reset()
        self._session_traces = []
        self._initialized = False
        logger.info("HermesNextProvider shut down")

    def on_session_end(self, messages: list[dict[str, Any]], session_id: str = "") -> None:
        """Called when the Hermes session ends.

        Runs the cognitive pipeline (Reward → L2 → L3 → Skill) on
        accumulated session traces, persists new policies/skills, and
        commits the OpenViking session.
        """
        # Run cognitive pipeline on accumulated traces
        if self._pipeline and self._session_traces:
            try:
                results = self._pipeline.process_session_end(
                    traces=self._session_traces,
                    session_success=bool(messages),
                )
                self._persist_pipeline_results(results)
            except Exception as e:
                logger.warning("Cognitive pipeline failed: %s", e)

        # v0.4: Close session state for crash recovery
        if self._session_repo:

            self._session_repo.close(session_id)  # type: ignore[arg-type]

        # Reset session traces
        self._session_traces = []

        # Commit OpenViking session
        if self._session:
            import asyncio

            try:
                asyncio.run(self._session.commit())
            except Exception as e:
                logger.warning("Session commit failed: %s", e)

    def _persist_pipeline_results(self, results: dict[str, Any]) -> None:
        """Persist cognitive pipeline results (new policies/skills/concepts) to local cache."""
        if not self._cache:
            return

        # Persist new policies
        new_policies = results.get("new_policies", [])
        if new_policies:
            from hermes_next.cache.policies import PolicyRepository

            repo = PolicyRepository(self._cache)
            for policy in new_policies:
                try:
                    repo.insert(policy)
                    # Promote high-confidence policies to native MEMORY.md
                    if self._native:
                        self._native.promote_policy(
                            name=policy.name,
                            trigger=policy.trigger_pattern,
                            action=policy.action_template,
                            confidence=policy.confidence,
                        )
                except Exception as e:
                    logger.warning("Failed to persist policy %s: %s", policy.name, e)

        # Persist new skills
        new_skills = results.get("new_skills", [])
        if new_skills:
            from hermes_next.cache.skills import SkillRepository

            repo = SkillRepository(self._cache)
            for skill in new_skills:
                try:
                    repo.insert(skill)
                    # Promote crystallized skills to native MEMORY.md
                    if self._native:
                        self._native.promote_skill(
                            name=skill.name,
                            description=skill.description,
                        )
                except Exception as e:
                    logger.warning("Failed to persist skill %s: %s", skill.name, e)

        # Persist new concepts
        new_concepts = results.get("new_concepts", [])
        if new_concepts:
            from hermes_next.cache.concepts import ConceptRepository

            repo = ConceptRepository(self._cache)
            for concept in new_concepts:
                try:
                    repo.insert(concept)
                except Exception as e:
                    logger.warning("Failed to persist concept %s: %s", concept.id, e)

        # Persist triples extracted by the pipeline
        pipeline_triples = results.get("stage", {}).get("world_model", {}).get("triples_extracted", 0)
        if pipeline_triples > 0 and self._pipeline:
            from hermes_next.cache.concepts import TripleRepository

            repo = TripleRepository(self._cache)
            for triple in self._pipeline.triples:
                try:
                    repo.insert(triple)
                except Exception as e:
                    logger.warning("Failed to persist triple %s: %s", triple.id, e)

        # Persist updated traces (with reward values)
        updated_traces = results.get("updated_traces", [])
        if updated_traces and len(updated_traces) != len(self._session_traces):
            from hermes_next.cache.traces import TraceRepository

            repo = TraceRepository(self._cache)
            for trace in updated_traces:
                try:
                    repo.insert(trace)
                except Exception as e:
                    logger.warning("Failed to persist updated trace %s: %s", trace.id, e)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Generate a memory summary for context compression."""
        if not messages:
            return ""
        recent = messages[-6:]
        summary_parts: list[str] = []
        for msg in recent:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 100:
                content = content[:100] + "..."
            summary_parts.append(f"[{role}] {content}")
        return "\n".join(summary_parts[-3:])

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        session_id: str,
        target_agent: Optional[str] = None,
    ) -> None:
        """Record sub-agent delegation as a tagged trace and feed into pipeline."""
        if not self._client:
            return
        self._turn_index += 1
        tags = ["delegation"]
        if target_agent:
            tags.append(f"agent:{target_agent}")
        trace = capture_trace(
            client=self._client,
            session_id=session_id,
            turn_index=self._turn_index,
            user_content=f"[Delegation to {target_agent or 'sub-agent'}] {task}",
            assistant_content=result,
            agent_name=self._agent_name,
            tags=tags,
            metadata={"type": "delegation", "target_agent": target_agent},
        )

        # Also feed delegation into cognitive pipeline
        if trace and self._pipeline:
            self._pipeline.process_trace(trace)
            self._session_traces.append(trace)


