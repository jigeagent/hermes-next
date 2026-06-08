"""Concept and Triple repositories — persist L3 World Model to SQLite."""

from __future__ import annotations

import json
from typing import Any, Optional

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.schema import ensure_schema
from hermes_next.memos.world_model import Concept, Triple


class ConceptRepository:
    """Persist and query L3 concepts locally."""

    def __init__(self, cache: CacheConnection):
        self._cache = cache
        ensure_schema(cache)

    def insert(self, concept: Concept) -> None:
        self._cache.execute(
            """
            INSERT OR REPLACE INTO concepts
                (id, label, description, embedding, member_trace_ids,
                 member_policy_ids, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                concept.id,
                concept.label,
                concept.description,
                json.dumps(concept.embedding) if concept.embedding else None,
                json.dumps(concept.member_trace_ids, ensure_ascii=False),
                json.dumps(concept.member_policy_ids, ensure_ascii=False),
                json.dumps(concept.metadata, ensure_ascii=False, default=str),
                concept.created_at,
            ),
        )

    def get(self, concept_id: str) -> Optional[Concept]:
        row = self._cache.execute(
            "SELECT * FROM concepts WHERE id = ?", (concept_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_concept(row)

    def list_all(self) -> list[Concept]:
        rows = self._cache.execute(
            "SELECT * FROM concepts ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_concept(r) for r in rows]

    def search_by_label(self, query: str) -> list[Concept]:
        like = f"%{query}%"
        rows = self._cache.execute(
            "SELECT * FROM concepts WHERE label LIKE ? OR description LIKE ?",
            (like, like),
        ).fetchall()
        return [self._row_to_concept(r) for r in rows]

    def count(self) -> int:
        row = self._cache.execute("SELECT COUNT(*) as cnt FROM concepts").fetchone()
        return row["cnt"] if row else 0

    def delete(self, concept_id: str) -> None:
        self._cache.execute("DELETE FROM concepts WHERE id = ?", (concept_id,))

    @staticmethod
    def _row_to_concept(row: Any) -> Concept:
        return Concept(
            id=row["id"],
            label=row["label"],
            description=row["description"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            member_trace_ids=json.loads(row["member_trace_ids"])
            if isinstance(row["member_trace_ids"], str)
            else [],
            member_policy_ids=json.loads(row["member_policy_ids"])
            if isinstance(row["member_policy_ids"], str)
            else [],
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else {},
            created_at=row["created_at"],
        )


class TripleRepository:
    """Persist and query L3 triples locally."""

    def __init__(self, cache: CacheConnection):
        self._cache = cache
        ensure_schema(cache)

    def insert(self, triple: Triple) -> None:
        self._cache.execute(
            """
            INSERT OR REPLACE INTO triples
                (id, subject, predicate, object, confidence,
                 source_trace_id, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                triple.id,
                triple.subject,
                triple.predicate,
                triple.object_,
                triple.confidence,
                triple.source_trace_id,
                json.dumps(triple.metadata, ensure_ascii=False, default=str),
                triple.created_at,
            ),
        )

    def insert_batch(self, triples: list[Triple]) -> None:
        rows = []
        for t in triples:
            rows.append((
                t.id, t.subject, t.predicate, t.object_, t.confidence,
                t.source_trace_id,
                json.dumps(t.metadata, ensure_ascii=False, default=str),
                t.created_at,
            ))
        self._cache.executemany(
            """
            INSERT OR REPLACE INTO triples
                (id, subject, predicate, object, confidence,
                 source_trace_id, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
        limit: int = 50,
    ) -> list[Triple]:
        conditions = []
        params = []
        if subject:
            conditions.append("subject LIKE ?")
            params.append(f"%{subject}%")
        if predicate:
            conditions.append("predicate LIKE ?")
            params.append(f"%{predicate}%")
        if object_:
            conditions.append("object LIKE ?")
            params.append(f"%{object_}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM triples WHERE {where} ORDER BY confidence DESC LIMIT ?"
        params.append(limit)
        rows = self._cache.execute(sql, tuple(params)).fetchall()
        return [self._row_to_triple(r) for r in rows]

    def list_all(self) -> list[Triple]:
        rows = self._cache.execute(
            "SELECT * FROM triples ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_triple(r) for r in rows]

    def count(self) -> int:
        row = self._cache.execute("SELECT COUNT(*) as cnt FROM triples").fetchone()
        return row["cnt"] if row else 0

    @staticmethod
    def _row_to_triple(row: Any) -> Triple:
        return Triple(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_=row["object"],
            confidence=row["confidence"],
            source_trace_id=row["source_trace_id"] if "source_trace_id" in row else "",
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else {},
            created_at=row["created_at"],
        )
