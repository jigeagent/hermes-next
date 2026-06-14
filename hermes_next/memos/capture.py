"""L1 Trace capture — records interaction turns as MemOS traces."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from hermes_next.memos.id import new_id
from hermes_next.memos.types import TraceRow
from hermes_next.ov.client import OpenVikingClient

logger = logging.getLogger(__name__)


def capture_trace(
    client: OpenVikingClient,
    session_id: str,
    turn_index: int,
    user_content: str,
    assistant_content: str,
    agent_name: str = "default",
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> Optional[TraceRow]:
    """Capture a single interaction turn as an L1 Trace.

    Steps:
    1. Generate a UUID v7 ID for the trace
    2. Compute embedding via OpenViking API (失败不阻塞写入)
    3. Build TraceRow and persist to local cache
    4. Try OV sync (失败不阻塞, log 警告)

    Note: embedding 和 OV 写入都是非关键路径。
    即使 OV 离线，trace 仍会写入本地 cache.db。
    """
    # Tags validation: only allow standardized categories
    VALID_TAGS = [["chat"], ["decision"], ["bugfix"], []]
    if tags is not None:
        assert [tags] in VALID_TAGS or tags in [["chat"], ["decision"], ["bugfix"], []], \
            f"Invalid tags: {tags}. Must be one of: {VALID_TAGS}"

    trace_id = new_id()
    created_at = datetime.now(timezone.utc).isoformat()

    # Compute combined text for embedding
    embed_text = f"User: {user_content}\nAssistant: {assistant_content}"

    # Get embedding from OpenViking — non-blocking: 失败则 embedding=None
    embedding = None
    try:
        embedding = client.embed(embed_text)
    except Exception as e:
        logger.warning("Embedding failed for trace %s (OV may be offline): %s", trace_id, e)

    trace = TraceRow(
        id=trace_id,
        session_id=session_id,
        turn_index=turn_index,
        user_content=user_content,
        assistant_content=assistant_content,
        embedding=embedding,
        tags=tags or [],
        metadata=metadata or {},
        created_at=created_at,
    )

    # Persist to OpenViking — non-blocking: 失败不阻塞写入
    try:
        uri = f"viking://resources/memory/traces/{trace_id}.json"
        content = json.dumps(trace.to_dict(), ensure_ascii=False, default=str)
        success = client.content_write(
            uri=uri,
            content=content,
            content_type="application/json",
            metadata={
                "type": "trace",
                "session_id": session_id,
                "turn_index": turn_index,
                "agent": agent_name,
            },
        )
        if not success:
            logger.warning("OV write failed for trace %s (OV may be offline)", trace_id)
    except Exception as e:
        logger.warning("OV write exception for trace %s: %s", trace_id, e)

    logger.info("Trace %s captured (turn %d, embedding=%s)", trace_id, turn_index,
                "yes" if embedding is not None else "no")
    return trace
