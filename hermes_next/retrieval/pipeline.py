"""6-step fusion retrieval pipeline.

Steps:
1. Semantic search (OpenViking vector)
2. Full-text search (SQLite FTS5)
3. Policy matching
4. Timeline context
5. Recency boost
6. MMR diversification
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.traces import TraceRepository
from hermes_next.config import RetrievalConfig
from hermes_next.memos.retrieval import retrieve_semantic
from hermes_next.ov.client import OpenVikingClient
from hermes_next.retrieval.ranker import mmr_diversify, rrf_merge

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Unified retrieval pipeline combining OpenViking search with local cache."""

    def __init__(
        self,
        ov_client: OpenVikingClient,
        cache: CacheConnection,
        config: Optional[RetrievalConfig] = None,
    ):
        self._client = ov_client
        self._cache = cache
        self._config = config or RetrievalConfig()
        self._trace_repo = TraceRepository(cache)

    def retrieve(
        self,
        query: str,
        agent: str = "default",
        query_embedding: Optional[list[float]] = None,
    ) -> list[dict[str, Any]]:
        """Run the full 6-step retrieval pipeline.

        Args:
            query: Search query string.
            agent: Agent name for namespace scoping.
            query_embedding: Optional pre-computed query embedding.

        Returns:
            Ranked and diversified list of memory items.
        """
        cfg = self._config

        # Step 1: Semantic search via OpenViking
        semantic_results = retrieve_semantic(
            self._client, query, k=cfg.semantic_k, agent=agent
        )

        # Step 2: Full-text search via local SQLite FTS5
        fts_results = self._fts_search(query, k=cfg.fts_k)

        # Step 3: Timeline (recent context)
        timeline_results = self._recent_timeline(k=cfg.timeline_k)

        # Step 4: RRF merge of all result sets
        rankings = [semantic_results, fts_results, timeline_results]
        rankings = [r for r in rankings if r]

        merged = rrf_merge(rankings, k=cfg.rrf_k)

        # Step 5: Recency boost
        merged = self._apply_recency_boost(merged, decay=cfg.recency_decay)

        # Step 6: MMR diversification
        diversified = mmr_diversify(
            merged,
            query_embedding=query_embedding,
            lambda_=cfg.mmr_lambda,
            k=cfg.mmr_k,
        )

        return diversified

    def _fts_search(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        """Full-text search via local SQLite FTS5."""
        try:
            traces = self._trace_repo.search_fts(query, limit=k)
            return [
                {
                    "id": t.id,
                    "content": f"User: {t.user_content}\nAssistant: {t.assistant_content}",
                    "user_content": t.user_content,
                    "assistant_content": t.assistant_content,
                    "score": 0.5,
                    "source": "fts",
                    "created_at": t.created_at,
                }
                for t in traces
            ]
        except Exception as e:
            logger.warning("FTS search failed: %s", e)
            return []

    def _recent_timeline(self, k: int = 8) -> list[dict[str, Any]]:
        """Fetch recent trace timeline."""
        try:
            traces = self._trace_repo.list_recent(limit=k)
            return [
                {
                    "id": t.id,
                    "content": f"User: {t.user_content}\nAssistant: {t.assistant_content}",
                    "score": 0.3,
                    "source": "timeline",
                    "created_at": t.created_at,
                }
                for t in traces
            ]
        except Exception as e:
            logger.warning("Timeline query failed: %s", e)
            return []

    @staticmethod
    def _apply_recency_boost(
        results: list[dict[str, Any]],
        decay: float = 0.9,
        time_key: str = "created_at",
        score_key: str = "score",
    ) -> list[dict[str, Any]]:
        """Apply recency-based score boost using exponential decay."""
        if not results:
            return results

        # Find the latest timestamp
        timestamps = [
            r.get(time_key, "") for r in results if r.get(time_key)
        ]
        if not timestamps:
            return results

        # Simple heuristic: more recent = higher boost
        for i, r in enumerate(results):
            base_score = r.get(score_key, 0)
            # Position-based recency boost (earlier in list = more recent)
            boost = decay ** i
            r[score_key] = base_score * (1 + boost * 0.2)

        results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
        return results
