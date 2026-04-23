from __future__ import annotations

import json
import logging
import os
import re
import time

from dotenv import load_dotenv

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

import openai  # noqa: E402 — import after env load so key is available

client = openai.OpenAI(api_key=OPENAI_API_KEY)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Parse JSON from text that may be wrapped in markdown code blocks or have extra content."""
    # 1. Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find first {...} in text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from text: {text[:200]}")


# ---------------------------------------------------------------------------
# actor_answer
# ---------------------------------------------------------------------------

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, float]:
    """Call the Actor LLM and return (answer, token_count, latency_ms)."""
    # Build context string
    context_parts = []
    for chunk in example.context:
        context_parts.append(f"Title: {chunk.title}\n{chunk.text}")
    context_str = "\n\n".join(context_parts)

    user_message = f"Question: {example.question}\n\nContext:\n{context_str}"

    if reflection_memory:
        memory_str = "\n".join(f"- {m}" for m in reflection_memory)
        user_message += f"\n\nReflection memory (lessons from previous attempts):\n{memory_str}"

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": ACTOR_SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
        latency_ms = (time.time() - t0) * 1000

        answer = response.choices[0].message.content or ""
        token_count = response.usage.total_tokens if response.usage else 0
        return (answer.strip(), token_count, latency_ms)

    except Exception as exc:
        logger.error("actor_answer error: %s", exc)
        return ("", 0, 0.0)


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------

def evaluator(
    example: QAExample,
    answer: str,
) -> tuple[JudgeResult, int, float]:
    """Call the Evaluator LLM and return (JudgeResult, token_count, latency_ms)."""
    user_message = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}"
    )

    fallback = JudgeResult(score=0, reason="evaluation error")

    for attempt in range(2):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": EVALUATOR_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
            latency_ms = (time.time() - t0) * 1000

            token_count = response.usage.total_tokens if response.usage else 0
            raw = response.choices[0].message.content or ""

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = extract_json(raw)

            judge = JudgeResult.model_validate(data)
            return (judge, token_count, latency_ms)

        except Exception as exc:
            logger.error("evaluator error (attempt %d): %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(2)

    return (fallback, 0, 0.0)


# ---------------------------------------------------------------------------
# reflector
# ---------------------------------------------------------------------------

def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int, float]:
    """Call the Reflector LLM and return (ReflectionEntry, token_count, latency_ms)."""
    user_message = (
        f"Question: {example.question}\n"
        f"Attempt ID: {attempt_id}\n"
        f"Predicted answer was marked incorrect.\n"
        f"Evaluator reason: {judge.reason}"
    )

    fallback = ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason="reflection error",
        lesson="",
        next_strategy="",
    )

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": REFLECTOR_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        latency_ms = (time.time() - t0) * 1000

        token_count = response.usage.total_tokens if response.usage else 0
        raw = response.choices[0].message.content or ""

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = extract_json(raw)

        entry = ReflectionEntry.model_validate(data)
        return (entry, token_count, latency_ms)

    except Exception as exc:
        logger.error("reflector error: %s", exc)
        return (fallback, 0, 0.0)
