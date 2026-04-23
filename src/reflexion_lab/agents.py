from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from . import mock_runtime
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

# FAILURE_MODE_BY_QID is only used with mock_runtime
from .mock_runtime import FAILURE_MODE_BY_QID


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: str = "mock"

    def _get_runtime(self):
        if self.runtime == "real":
            from . import llm_runtime
            return llm_runtime
        return mock_runtime

    def _call_actor(self, rt, example: QAExample, attempt_id: int, reflection_memory: list[str]):
        result = rt.actor_answer(example, attempt_id, self.agent_type, reflection_memory)
        if isinstance(result, tuple):
            answer, tokens, latency = result
        else:
            answer, tokens, latency = result, 0, 0.0
        return answer, tokens, latency

    def _call_evaluator(self, rt, example: QAExample, answer: str):
        result = rt.evaluator(example, answer)
        if isinstance(result, tuple):
            judge, tokens, latency = result
        else:
            judge, tokens, latency = result, 0, 0.0
        return judge, tokens, latency

    def _call_reflector(self, rt, example: QAExample, attempt_id: int, judge):
        result = rt.reflector(example, attempt_id, judge)
        if isinstance(result, tuple):
            entry, tokens, latency = result
        else:
            entry, tokens, latency = result, 0, 0.0
        return entry, tokens, latency

    def _classify_failure_mode(self, example: QAExample, answer: str, judge, attempts: int) -> str:
        """Phân loại failure mode dựa trên đặc điểm câu trả lời và lý do đánh giá."""
        if judge.score == 1:
            return "none"
        
        # Empty answer
        if not answer or answer.strip() == "":
            return "no_answer_generated"
        
        # Looping (Reflexion thử nhiều lần nhưng vẫn sai)
        if self.agent_type == "reflexion" and attempts >= self.max_attempts:
            return "looping"
        
        # Incomplete multi-hop (dựa vào reason từ evaluator)
        reason_lower = judge.reason.lower()
        if any(keyword in reason_lower for keyword in ["first hop", "incomplete", "stopped", "partial"]):
            return "incomplete_multi_hop"
        
        # Entity drift (nhầm entity)
        if any(keyword in reason_lower for keyword in ["wrong entity", "entity", "drift", "confused"]):
            return "entity_drift"
        
        # Default
        return "wrong_final_answer"

    def run(self, example: QAExample) -> RunRecord:
        rt = self._get_runtime()
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_judge = None

        for attempt_id in range(1, self.max_attempts + 1):
            answer, actor_tokens, actor_latency = self._call_actor(rt, example, attempt_id, reflection_memory)
            judge, eval_tokens, eval_latency = self._call_evaluator(rt, example, answer)

            token_estimate = actor_tokens + eval_tokens
            latency_ms = actor_latency + eval_latency

            final_answer = answer
            final_score = judge.score
            final_judge = judge

            if self.agent_type == "reflexion" and judge.score == 0 and attempt_id < self.max_attempts:
                entry, ref_tokens, ref_latency = self._call_reflector(rt, example, attempt_id, judge)
                reflection_memory.append(entry.next_strategy)
                reflections.append(entry)
                token_estimate += ref_tokens
                latency_ms += ref_latency
                trace = AttemptTrace(
                    attempt_id=attempt_id,
                    answer=answer,
                    score=judge.score,
                    reason=judge.reason,
                    reflection=entry,
                    token_estimate=token_estimate,
                    latency_ms=int(latency_ms),
                )
            else:
                trace = AttemptTrace(
                    attempt_id=attempt_id,
                    answer=answer,
                    score=judge.score,
                    reason=judge.reason,
                    token_estimate=token_estimate,
                    latency_ms=int(latency_ms),
                )

            traces.append(trace)

            if judge.score == 1:
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        
        # Classify failure mode based on answer characteristics
        if self.runtime == "real":
            failure_mode = self._classify_failure_mode(example, final_answer, final_judge, len(traces))
        else:
            failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self, runtime: str = "mock") -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime=runtime)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, runtime: str = "mock") -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, runtime=runtime)
