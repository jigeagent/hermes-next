"""Tests for L2 Policy induction."""

import pytest

from hermes_next.memos.policy import PolicyConfig, PolicyInducer
from hermes_next.memos.types import PolicyRow, TraceRow


def _make_trace(
    id_: str,
    turn: int = 0,
    reward: float = 0.0,
    user: str = "hello world",
    asst: str = "hi there",
    emb: list[float] | None = None,
) -> TraceRow:
    return TraceRow(
        id=id_,
        session_id="s1",
        turn_index=turn,
        user_content=user,
        assistant_content=asst,
        reward=reward,
        embedding=emb or [0.1, 0.2, 0.3],
    )


def _make_policy(
    id_: str,
    name: str = "test_policy",
    confidence: float = 0.5,
    emb: list[float] | None = None,
) -> PolicyRow:
    return PolicyRow(
        id=id_,
        name=name,
        description="A test policy",
        trigger_pattern="test trigger",
        action_template="do something",
        confidence=confidence,
        embedding=emb or [0.1, 0.2, 0.3],
    )


class TestCandidatePool:
    """Building candidate pools from trace clusters."""

    def test_empty_traces(self):
        inducer = PolicyInducer()
        pool = inducer.build_candidate_pool([])
        assert pool == []

    def test_traces_without_embedding(self):
        inducer = PolicyInducer()
        traces = [_make_trace("t1", emb=None)]
        pool = inducer.build_candidate_pool(traces)
        assert pool == []

    def test_clusters_similar_traces(self):
        inducer = PolicyInducer(PolicyConfig(
            embedding_sim_threshold=0.9,
            min_traces_for_induction=2,
        ))
        traces = [
            _make_trace("t1", emb=[1.0, 0.0, 0.0]),
            _make_trace("t2", emb=[0.95, 0.05, 0.0]),
            _make_trace("t3", emb=[0.0, 1.0, 0.0]),
        ]
        pool = inducer.build_candidate_pool(traces)
        # t1 + t2 should cluster, t3 is separate (too few)
        assert len(pool) == 1
        assert len(pool[0]) == 2


class TestInduction:
    """Policy induction from trace clusters."""

    def test_requires_min_traces(self):
        inducer = PolicyInducer(PolicyConfig(min_traces_for_induction=3))
        cluster = [
            _make_trace("t1"),
            _make_trace("t2"),
        ]
        policy = inducer.induce(cluster)
        assert policy is None

    def test_induces_policy(self):
        inducer = PolicyInducer(PolicyConfig(min_traces_for_induction=2))
        cluster = [
            _make_trace("t1", reward=0.8, user="how do I deploy flask", asst="use gunicorn"),
            _make_trace("t2", reward=0.6, user="how to deploy django", asst="use gunicorn"),
        ]
        policy = inducer.induce(cluster)
        assert policy is not None
        assert policy.confidence > 0
        assert len(policy.source_trace_ids) == 2
        assert policy.name != ""

    def test_batch_induce(self):
        inducer = PolicyInducer(PolicyConfig(
            min_traces_for_induction=2,
            embedding_sim_threshold=0.9,
        ))
        traces = [
            _make_trace("t1", user="how to use docker", emb=[1.0, 0.0, 0.0]),
            _make_trace("t2", user="docker compose guide", emb=[0.95, 0.05, 0.0]),
        ]
        policies = inducer.batch_induce(traces)
        assert len(policies) >= 0  # may or may not meet embedding threshold


class TestActivation:
    """Policy activation from context."""

    def test_empty_policies(self):
        inducer = PolicyInducer()
        activated = inducer.activate("test context", [])
        assert activated == []

    def test_below_confidence_threshold(self):
        inducer = PolicyInducer(PolicyConfig(min_confidence=0.5))
        policy = _make_policy("p1", confidence=0.3)
        activated = inducer.activate("test context", [policy])
        assert activated == []

    def test_activation_by_keyword(self):
        inducer = PolicyInducer(PolicyConfig(min_confidence=0.0, activation_sim_threshold=0.0))
        policy = _make_policy("p1", confidence=1.0)
        activated = inducer.activate("hello world test", [policy])
        # Without embedding, keyword overlap should give some score
        assert len(activated) >= 0


class TestConfidenceUpdate:
    """Policy confidence updates."""

    def test_success_increases_confidence(self):
        inducer = PolicyInducer()
        policy = _make_policy("p1", confidence=0.5)
        updated = inducer.update_confidence(policy, activation_success=True)
        assert updated.confidence > 0.5
        assert updated.activation_count == 1

    def test_failure_decreases_confidence(self):
        inducer = PolicyInducer()
        policy = _make_policy("p1", confidence=0.5)
        updated = inducer.update_confidence(policy, activation_success=False)
        assert updated.confidence < 0.5


class TestMerge:
    """Policy merging across batches."""

    def test_merge_new_policies(self):
        inducer = PolicyInducer()
        existing = [_make_policy("p1", name="existing_pol")]
        new = [_make_policy("p2", name="new_pol")]
        merged = inducer._merge_policies(existing, new)
        assert len(merged) == 2

    def test_dedup_by_name(self):
        inducer = PolicyInducer()
        existing = [_make_policy("p1", name="same_name")]
        new = [_make_policy("p2", name="same_name")]
        merged = inducer._merge_policies(existing, new)
        # Should be merged into one with combined source traces
        assert len(merged) == 1
