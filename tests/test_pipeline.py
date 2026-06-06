"""Tests for cognitive pipeline orchestration."""

import pytest

from hermes_next.memos.pipeline import (
    CognitivePipeline,
    CognitivePipelineConfig,
    PipelineStage,
)
from hermes_next.memos.reward import OutcomeSignal
from hermes_next.memos.types import TraceRow


def _make_trace(id_: str, turn: int = 0, reward: float = 0.0) -> TraceRow:
    return TraceRow(
        id=id_,
        session_id="s1",
        turn_index=turn,
        user_content=f"user {turn}",
        assistant_content=f"asst {turn}",
        reward=reward,
        embedding=[0.1, 0.2, 0.3],
    )


class TestPipelineInit:
    """Pipeline initialization."""

    def test_default_config(self):
        pipeline = CognitivePipeline()
        assert PipelineStage.L1_CAPTURE in pipeline._config.enabled_stages
        assert PipelineStage.REWARD in pipeline._config.enabled_stages
        assert PipelineStage.L2_INDUCTION in pipeline._config.enabled_stages
        # L3 and Skill are opt-in
        assert PipelineStage.L3_WORLD_MODEL not in pipeline._config.enabled_stages
        assert PipelineStage.SKILL_CRYSTALLIZATION not in pipeline._config.enabled_stages


class TestProcessTrace:
    """Single trace processing."""

    def test_process_adds_to_current(self):
        pipeline = CognitivePipeline()
        trace = _make_trace("t1")
        result = pipeline.process_trace(trace)
        assert result.id == "t1"
        assert len(pipeline._current_traces) == 1


class TestSessionEnd:
    """Session-end pipeline execution."""

    def test_session_end_returns_results(self):
        pipeline = CognitivePipeline()
        traces = [_make_trace("t1", turn=0), _make_trace("t2", turn=1)]
        results = pipeline.process_session_end(traces, session_success=True)

        assert "stage" in results
        assert "updated_traces" in results
        assert "new_policies" in results

    def test_session_end_applies_reward(self):
        pipeline = CognitivePipeline()
        traces = [_make_trace("t1", turn=0)]
        results = pipeline.process_session_end(traces, session_success=True)
        assert results["stage"]["reward"]["applied"] is True
        # Trace reward should be updated
        assert results["updated_traces"][0].reward != 0.0

    def test_reward_stats(self):
        pipeline = CognitivePipeline()
        traces = [_make_trace("t1", turn=0)]
        results = pipeline.process_session_end(traces, session_success=True)
        stats = results["stage"]["reward"]["stats"]
        assert "mean" in stats
        assert "total" in stats


class TestOutcomeProcessing:
    """Outcome signal processing."""

    def test_outcome_updates_rewards(self):
        pipeline = CognitivePipeline()
        traces = [_make_trace("t1", turn=0), _make_trace("t2", turn=1)]
        updated = pipeline.process_outcome(traces, OutcomeSignal.TASK_SUCCESS)
        assert updated[-1].reward > 0

    def test_outcome_triggers_reinduction(self):
        pipeline = CognitivePipeline()
        traces = [_make_trace("t1", turn=0, reward=0.5), _make_trace("t2", turn=1, reward=0.5)]
        updated = pipeline.process_outcome(traces, OutcomeSignal.TASK_SUCCESS)
        assert len(updated) == 2


class TestHooks:
    """Pipeline event hooks."""

    def test_hook_is_called(self):
        pipeline = CognitivePipeline()
        events: list[PipelineStage] = []

        def hook(stage, ctx):
            events.append(stage)

        pipeline.add_hook(hook)
        pipeline._emit(PipelineStage.L1_CAPTURE, {"trace": _make_trace("t1")})
        assert PipelineStage.L1_CAPTURE in events


class TestStats:
    """Pipeline statistics."""

    def test_initial_stats(self):
        pipeline = CognitivePipeline()
        stats = pipeline.get_stats()
        assert stats["traces_processed"] == 0
        assert stats["policies"] == 0
        assert stats["skills"] == 0

    def test_stats_after_processing(self):
        pipeline = CognitivePipeline()
        pipeline.process_trace(_make_trace("t1"))
        stats = pipeline.get_stats()
        assert stats["traces_processed"] == 1


class TestReset:
    """Pipeline reset."""

    def test_reset_clears_state(self):
        pipeline = CognitivePipeline()
        pipeline.process_trace(_make_trace("t1"))
        pipeline.reset()
        assert len(pipeline._current_traces) == 0
        assert pipeline._policies == []
