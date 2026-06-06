"""Tests for reward backpropagation engine."""

import pytest

from hermes_next.memos.reward import OutcomeSignal, RewardConfig, RewardEngine
from hermes_next.memos.types import TraceRow


def _make_trace(id_: str, turn: int, reward: float = 0.0) -> TraceRow:
    return TraceRow(
        id=id_,
        session_id="s1",
        turn_index=turn,
        user_content=f"user {turn}",
        assistant_content=f"asst {turn}",
        reward=reward,
        embedding=[0.1 * turn, 0.2 * turn],
    )


class TestRewardConfig:
    """Default config sanity."""

    def test_defaults(self):
        cfg = RewardConfig()
        assert cfg.temporal_discount == 0.85
        assert cfg.min_reward == -2.0
        assert cfg.max_reward == 2.0


class TestOutcomeSignals:
    """Reward values for different signals."""

    def test_success_signal(self):
        engine = RewardEngine()
        traces = [_make_trace("t1", 1), _make_trace("t2", 2)]
        updated = engine.apply_outcome(traces, OutcomeSignal.TASK_SUCCESS)
        # Last trace gets full reward
        assert updated[-1].reward > 0.5

    def test_failure_signal(self):
        engine = RewardEngine()
        traces = [_make_trace("t1", 1), _make_trace("t2", 2)]
        updated = engine.apply_outcome(traces, OutcomeSignal.TASK_FAILURE)
        assert updated[-1].reward < 0

    def test_manual_reward(self):
        engine = RewardEngine()
        traces = [_make_trace("t1", 1)]
        updated = engine.apply_outcome(
            traces,
            OutcomeSignal.MANUAL_REWARD,
            manual_value=0.75,
        )
        assert updated[-1].reward == pytest.approx(0.75)


class TestBackpropagation:
    """Reward backpropagation through trace chains."""

    def test_backpropagate_discounts(self):
        engine = RewardEngine(RewardConfig(temporal_discount=0.5))
        traces = [
            _make_trace("t0", 0),
            _make_trace("t1", 1),
            _make_trace("t2", 2),
        ]
        updated = engine.apply_outcome(traces, OutcomeSignal.TASK_SUCCESS)
        # Target turn gets highest reward
        assert updated[2].reward > updated[1].reward > updated[0].reward

    def test_backpropagate_steps_limited(self):
        engine = RewardEngine(RewardConfig(backpropagation_steps=2))
        traces = [_make_trace(f"t{i}", i) for i in range(10)]
        updated = engine.apply_outcome(traces, OutcomeSignal.TASK_SUCCESS)
        # Only last 2 steps should have non-default reward
        assert updated[7].reward == 0.0  # unchanged
        assert updated[8].reward != 0.0  # changed

    def test_reward_clipping(self):
        engine = RewardEngine(RewardConfig(min_reward=-1.0, max_reward=1.0))
        traces = [_make_trace("t0", 0)]
        # Apply a signal with high base value
        updated = engine.apply_outcome(traces, OutcomeSignal.TASK_SUCCESS)
        # TASK_SUCCESS is 1.0, which is exactly at max
        assert updated[0].reward <= 1.0


class TestSessionReward:
    """Session-level reward computation."""

    def test_session_success(self):
        engine = RewardEngine()
        traces = [
            _make_trace("t0", 0),
            _make_trace("t1", 1),
        ]
        updated = engine.compute_session_reward(traces, session_success=True)
        assert updated[-1].reward > 0

    def test_session_failure(self):
        engine = RewardEngine()
        traces = [
            _make_trace("t0", 0),
            _make_trace("t1", 1),
        ]
        updated = engine.compute_session_reward(traces, session_success=False)
        assert updated[-1].reward < 0


class TestAggregation:
    """Reward statistics aggregation."""

    def test_empty(self):
        stats = RewardEngine.aggregate_rewards([])
        assert stats["mean"] == 0.0

    def test_statistics(self):
        traces = [
            _make_trace("t0", 0, reward=1.0),
            _make_trace("t1", 1, reward=0.5),
        ]
        stats = RewardEngine.aggregate_rewards(traces)
        assert stats["mean"] == pytest.approx(0.75)
        assert stats["max"] == 1.0
        assert stats["min"] == 0.5
        assert stats["total"] == 1.5
