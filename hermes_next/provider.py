"""HermesNextProvider — MemoryProvider implementation for Hermes Agent."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.lifecycle import LifecycleManager
from hermes_next.cache.schema import ensure_schema
from hermes_next.config import HermesNextConfig
from hermes_next.integration.native import NativeMemoryClient, NativeMemoryConfig
from hermes_next.memos.capture import capture_trace
from hermes_next.memos.pipeline import (
    CognitivePipeline,
    CognitivePipelineConfig,
    PipelineStage,
)
from hermes_next.memos.retrieval import format_results, retrieve_semantic, retrieve_timeline
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

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Cached embedding lookup."""
        if not self._client:
            return None
        if text in self._embed_cache:
            return self._embed_cache[text]
        if len(self._embed_cache) >= self._embed_cache_max:
            # Simple evict: clear half
            evict = len(self._embed_cache) // 2
            for k in list(self._embed_cache)[:evict]:
                del self._embed_cache[k]
        emb = self._client.embed(text)
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
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any]) -> str:
        """Dispatch a tool call from the agent."""
        if not self._client:
            return "Error: provider not initialized"

        if tool_name == "memos_search":
            query = args.get("query", "")
            k = min(args.get("k", 8), 32)
            results = retrieve_semantic(self._client, query, k=k, agent=self._agent_name)
            return format_results(results) or "No relevant memories found."

        if tool_name == "memos_get":
            trace_id = args.get("trace_id", "")
            uri = f"viking://resources/{self._agent_name}/memos/traces/{trace_id}.json"
            content = self._client.content_read(uri)
            if content:
                try:
                    data = json.loads(content)
                    trace = TraceRow.from_dict(data)
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

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
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

        # Reset session state
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
