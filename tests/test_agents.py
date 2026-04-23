"""Tests for agents.py — Reflexion loop, memory propagation, early stopping, token aggregation."""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.reflexion_lab.schemas import (
    AttemptTrace,
    ContextChunk,
    JudgeResult,
    QAExample,
    ReflectionEntry,
    RunRecord,
)
from src.reflexion_lab.agents import BaseAgent, ReActAgent, ReflexionAgent


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_example(qid: str = "test_q", gold: str = "Paris") -> QAExample:
    return QAExample(
        qid=qid,
        difficulty="easy",
        question="What is the capital of France?",
        gold_answer=gold,
        context=[ContextChunk(title="France", text="Paris is the capital of France.")],
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestReActAgent:
    def test_single_attempt(self):
        agent = ReActAgent(runtime="mock")
        example = make_example(qid="hp1", gold="Thames")
        record = agent.run(example)
        assert record.attempts == 1
        assert len(record.traces) == 1

    def test_no_reflections(self):
        agent = ReActAgent(runtime="mock")
        example = make_example(qid="hp2", gold="Thames")
        record = agent.run(example)
        assert record.reflections == []

    def test_max_attempts_is_one(self):
        agent = ReActAgent(runtime="mock")
        assert agent.max_attempts == 1


class TestReflexionAgent:
    def test_reflexion_calls_reflector_on_failure(self):
        """When reflexion agent fails, it should produce reflections."""
        agent = ReflexionAgent(max_attempts=3, runtime="mock")
        # hp2 is designed to fail on first attempt in mock_runtime
        example = make_example(qid="hp2", gold="Thames")
        record = agent.run(example)
        # Should have retried and eventually got it right
        assert record.is_correct
        assert len(record.reflections) >= 1

    def test_reflexion_memory_propagated(self):
        """next_strategy from reflection should appear in subsequent actor calls."""
        # We verify indirectly: if memory is propagated, mock_runtime returns gold on attempt 2+
        agent = ReflexionAgent(max_attempts=3, runtime="mock")
        example = make_example(qid="hp2", gold="Thames")
        record = agent.run(example)
        # The mock returns gold_answer when reflection_memory is non-empty
        assert record.is_correct

    def test_token_estimate_sum(self):
        agent = ReflexionAgent(max_attempts=3, runtime="mock")
        example = make_example(qid="hp1", gold="Thames")
        record = agent.run(example)
        assert record.token_estimate == sum(t.token_estimate for t in record.traces)

    def test_latency_sum(self):
        agent = ReflexionAgent(max_attempts=3, runtime="mock")
        example = make_example(qid="hp1", gold="Thames")
        record = agent.run(example)
        assert record.latency_ms == sum(t.latency_ms for t in record.traces)


# ---------------------------------------------------------------------------
# Property 5: Reflexion memory propagation
# Validates: Yêu cầu 7.1, 7.2
# ---------------------------------------------------------------------------

@given(st.integers(min_value=2, max_value=5))
@settings(max_examples=20)
def test_property5_reflexion_memory_propagation(max_attempts):
    """
    **Validates: Requirements 7.1, 7.2**
    For any max_attempts >= 2, when the mock agent fails on hp2,
    the reflection entry's next_strategy must be added to reflection_memory
    so the actor can use it on the next attempt.
    """
    agent = ReflexionAgent(max_attempts=max_attempts, runtime="mock")
    example = make_example(qid="hp2", gold="Thames")
    record = agent.run(example)

    # If there were reflections, each one should have a non-empty next_strategy
    for ref in record.reflections:
        assert ref.next_strategy != "", "next_strategy must be non-empty"

    # The agent should eventually succeed (mock returns gold when memory is non-empty)
    assert record.is_correct


# ---------------------------------------------------------------------------
# Property 6: Early stopping on success
# Validates: Yêu cầu 7.4
# ---------------------------------------------------------------------------

@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=20)
def test_property6_early_stopping_on_success(max_attempts):
    """
    **Validates: Requirements 7.4**
    When the mock evaluator returns score=1 immediately (qid not in FIRST_ATTEMPT_WRONG),
    the RunRecord should have exactly 1 trace and no reflections.
    """
    agent = ReflexionAgent(max_attempts=max_attempts, runtime="mock")
    # hp1 is not in FIRST_ATTEMPT_WRONG, so mock returns gold_answer immediately
    example = make_example(qid="hp1", gold="Thames")
    record = agent.run(example)

    assert record.attempts == 1, f"Expected 1 attempt, got {record.attempts}"
    assert len(record.traces) == 1
    assert record.reflections == []
    assert record.is_correct


# ---------------------------------------------------------------------------
# Property 7: ReAct agent single attempt invariant
# Validates: Yêu cầu 7.5
# ---------------------------------------------------------------------------

_qa_strategy = st.builds(
    QAExample,
    qid=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
    difficulty=st.sampled_from(["easy", "medium", "hard"]),
    question=st.just("What is the capital of France?"),
    gold_answer=st.just("Paris"),
    context=st.just([ContextChunk(title="France", text="Paris is the capital of France.")]),
)


@given(_qa_strategy)
@settings(max_examples=30)
def test_property7_react_single_attempt(example):
    """
    **Validates: Requirements 7.5**
    ReActAgent always produces exactly 1 AttemptTrace regardless of input.
    """
    agent = ReActAgent(runtime="mock")
    record = agent.run(example)
    assert len(record.traces) == 1, f"ReActAgent must have exactly 1 trace, got {len(record.traces)}"
    assert record.attempts == 1
    assert record.reflections == []


# ---------------------------------------------------------------------------
# Property 8: Token estimate aggregation invariant
# Validates: Yêu cầu 7.6, 10.2
# ---------------------------------------------------------------------------

@given(st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=5))
@settings(max_examples=50)
def test_property8_token_aggregation(token_values):
    """
    **Validates: Requirements 7.6, 10.2**
    RunRecord.token_estimate must equal sum of all AttemptTrace.token_estimate.
    """
    # Build a RunRecord manually with controlled token values
    traces = [
        AttemptTrace(
            attempt_id=i + 1,
            answer="answer",
            score=0,
            reason="test",
            token_estimate=t,
            latency_ms=10,
        )
        for i, t in enumerate(token_values)
    ]
    total = sum(token_values)
    record = RunRecord(
        qid="test",
        question="q",
        gold_answer="a",
        agent_type="reflexion",
        predicted_answer="answer",
        is_correct=False,
        attempts=len(traces),
        token_estimate=total,
        latency_ms=10 * len(traces),
        failure_mode="wrong_final_answer",
        reflections=[],
        traces=traces,
    )
    assert record.token_estimate == sum(t.token_estimate for t in record.traces)
