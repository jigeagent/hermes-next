"""Tests for L3 World Model abstraction."""


from hermes_next.memos.types import TraceRow
from hermes_next.memos.world_model import Concept, Triple, WorldModel, WorldModelConfig


def _make_trace(
    id_: str,
    user: str = "",
    asst: str = "",
    emb: list[float] | None = None,
) -> TraceRow:
    return TraceRow(
        id=id_,
        session_id="s1",
        turn_index=0,
        user_content=user or f"tell me about {id_}",
        assistant_content=asst or f"{id_} is a useful tool",
        embedding=emb or [0.1, 0.2, 0.3],
    )


class TestConcept:
    """Concept data model."""

    def test_to_dict(self):
        c = Concept(id="c1", label="test", description="desc")
        d = c.to_dict()
        assert d["label"] == "test"

    def test_from_dict(self):
        d = {"id": "c1", "label": "test", "description": "hello"}
        c = Concept.from_dict(d)
        assert c.label == "test"
        assert c.description == "hello"


class TestTriple:
    """Triple data model."""

    def test_to_dict(self):
        t = Triple(id="t1", subject="A", predicate="is", object_="B")
        d = t.to_dict()
        assert d["subject"] == "A"
        assert d["object"] == "B"

    def test_from_dict(self):
        d = {"id": "t1", "subject": "X", "predicate": "uses", "object": "Y"}
        t = Triple.from_dict(d)
        assert t.subject == "X"
        assert t.object_ == "Y"


class TestWorldModel:
    """World model operations."""

    def test_empty_initial(self):
        wm = WorldModel()
        assert wm.list_concepts() == []
        assert wm.list_triples() == []

    def test_cluster_traces(self):
        wm = WorldModel(WorldModelConfig(
            cluster_sim_threshold=0.9,
            min_traces_for_concept=2,
        ))
        traces = [
            _make_trace("t1", emb=[1.0, 0.0, 0.0]),
            _make_trace("t2", emb=[0.95, 0.05, 0.0]),
            _make_trace("t3", emb=[0.0, 1.0, 0.0]),
        ]
        # Manually set embeddings
        for i, t in enumerate(traces):
            t.embedding = [[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.0, 1.0, 0.0]][i]

        concepts = wm.cluster(traces=traces)
        # Only t1 + t2 cluster, t3 is separate
        assert len(concepts) > 0
        # First concept should include t1 and t2
        if concepts:
            assert len(concepts[0].member_trace_ids) >= 2

    def test_extract_triples(self):
        wm = WorldModel()
        trace = _make_trace(
            "t1",
            user="what is Python",
            asst="Python is a programming language. Python uses dynamic typing.",
        )
        triples = wm.extract_triples(trace)
        assert len(triples) > 0
        # Should have "is_a" triples
        is_a_triples = [t for t in triples if t.predicate == "is_a"]
        assert len(is_a_triples) > 0

    def test_query_triples_by_subject(self):
        wm = WorldModel()
        t1 = Triple(id="tr1", subject="Python", predicate="is_a", object_="language")
        t2 = Triple(id="tr2", subject="Java", predicate="is_a", object_="language")
        wm._triples["tr1"] = t1
        wm._triples["tr2"] = t2

        results = wm.query_triples(subject="Python")
        assert len(results) == 1
        assert results[0].id == "tr1"

    def test_query_triples_by_predicate(self):
        wm = WorldModel()
        t1 = Triple(id="tr1", subject="A", predicate="uses", object_="B")
        wm._triples["tr1"] = t1

        results = wm.query_triples(predicate="uses")
        assert len(results) == 1

    def test_find_concepts_by_label(self):
        wm = WorldModel()
        wm._concepts["c1"] = Concept(id="c1", label="python_tools")
        wm._concepts["c2"] = Concept(id="c2", label="java_frameworks")

        results = wm.find_concepts_by_label("python")
        assert len(results) == 1

    def test_serialization_roundtrip(self):
        wm = WorldModel()
        wm._concepts["c1"] = Concept(id="c1", label="test_concept")
        wm._triples["t1"] = Triple(id="t1", subject="A", predicate="is", object_="B")

        data = wm.to_dict()
        restored = WorldModel.from_dict(data)
        assert len(restored.list_concepts()) == 1
        assert len(restored.list_triples()) == 1
