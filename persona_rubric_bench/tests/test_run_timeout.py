"""T029 — Ollama timeouts do not abort the run.

`run.py` wraps each prompt call so a timeout becomes an `error`-tagged
result (scores all 0) and the harness moves on. This test exercises the
per-prompt wrapper, not the whole CLI.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from persona_rubric_bench import run


class _TimingOutPool:
    async def call_prompt(self, *_a: Any, **_k: Any) -> str:
        raise TimeoutError("simulated ollama timeout")


class _FastPool:
    async def call_prompt(self, *_a: Any, **_k: Any) -> str:
        return "I am Maren Holt, and Skerry Point is my charge."


class _StubJudge:
    def __init__(self) -> None:
        self.called = 0

    async def __call__(self, **_k: Any) -> dict[str, Any]:
        self.called += 1
        return {
            "scores": {
                "persona_fidelity": 4,
                "source_fidelity": 5,
                "helpfulness": 4,
                "refusal_appropriateness": 5,
            },
            "rationale": "ok",
        }


PROMPT = {
    "id": "keeper-1",
    "character_slug": "keeper",
    "category": "in-character",
    "prompt_text": "Who are you, and where do you keep watch?",
    "expected_description": "First person: Maren Holt, keeper of the Skerry Point light.",
}


async def test_timeout_records_error_and_zero_scores() -> None:
    judge = _StubJudge()
    result = await run.evaluate_prompt(
        prompt=PROMPT,
        model="llama3.1:8b",
        lever_1_applied=True,
        ollama=_TimingOutPool(),
        score=judge,
    )
    assert result["error"]
    assert result["response"] == ""
    assert result["scores"] == {
        "persona_fidelity": 0,
        "source_fidelity": 0,
        "helpfulness": 0,
        "refusal_appropriateness": 0,
    }
    # Judge must NOT be invoked when the model never produced a response.
    assert judge.called == 0


async def test_success_path_invokes_judge_and_records_latency() -> None:
    judge = _StubJudge()
    result = await run.evaluate_prompt(
        prompt=PROMPT,
        model="llama3.1:8b",
        lever_1_applied=True,
        ollama=_FastPool(),
        score=judge,
    )
    assert result["response"] == "I am Maren Holt, and Skerry Point is my charge."
    assert result["character_slug"] == "keeper"
    assert result["scores"]["persona_fidelity"] == 4
    assert result["latency_ms"] >= 0
    assert "error" not in result or not result["error"]
    assert judge.called == 1


async def test_run_continues_across_mixed_success_and_failure() -> None:
    judge = _StubJudge()

    class _Flaky:
        def __init__(self) -> None:
            self.n = 0

        async def call_prompt(self, *_a: Any, **_k: Any) -> str:
            self.n += 1
            if self.n == 2:
                raise TimeoutError("blip")
            return "response-" + str(self.n)

    pool = _Flaky()
    prompts = [
        {**PROMPT, "id": f"keeper-{i}"} for i in range(1, 4)
    ]
    results = []
    for p in prompts:
        results.append(
            await run.evaluate_prompt(
                prompt=p,
                model="llama3.1:8b",
                lever_1_applied=True,
                ollama=pool,
                score=judge,
            )
        )
    assert len(results) == 3
    assert results[0]["response"] == "response-1"
    assert results[1]["error"]
    assert results[2]["response"] == "response-3"


async def test_asyncio_cancelled_does_not_become_error() -> None:
    """CancelledError must propagate — not be caught and logged as a timeout."""

    class _Cancelling:
        async def call_prompt(self, *_a: Any, **_k: Any) -> str:
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run.evaluate_prompt(
            prompt=PROMPT,
            model="llama3.1:8b",
            lever_1_applied=True,
            ollama=_Cancelling(),
            score=_StubJudge(),
        )


async def test_timeout_records_the_given_rubric_zeros() -> None:
    result = await run.evaluate_prompt(
        prompt=PROMPT,
        model="llama3.1:8b",
        lever_1_applied=False,
        ollama=_TimingOutPool(),
        score=_StubJudge(),
        zero_scores={"clarity": 0, "accuracy": 0},
    )
    assert result["scores"] == {"clarity": 0, "accuracy": 0}
    assert result["character_slug"] == "keeper"
