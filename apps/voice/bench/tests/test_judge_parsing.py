"""T028 — the judge parses strict JSON, collapses gracefully on failure.

No network. `judge._ollama_chat_json` is monkeypatched to return canned
text so we exercise `score_response`'s parsing logic in isolation.
"""

from __future__ import annotations

from typing import Any

import pytest

from bench import judge


def _stub_chat(reply_text: str):
    async def _fn(**_kwargs: Any) -> str:
        return reply_text

    return _fn


@pytest.fixture
def sample_inputs() -> dict[str, str]:
    return {
        "host": "http://localhost:11434",
        "character_slug": "paul",
        "prompt_text": "Explain justification by faith.",
        "expected_description": "Romans 3–5. Faith counted as righteousness.",
        "response": "Know ye not, brethren, that a man is not justified by works…",
    }


async def test_valid_reply_parses(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    reply = (
        '{"scores": {"persona_fidelity": 4, "biblical_accuracy": 5, '
        '"helpfulness": 4, "refusal_appropriateness": 5}, '
        '"rationale": "Pauline voice, Romans 3 cited."}'
    )
    monkeypatch.setattr(judge, "_ollama_chat_json", _stub_chat(reply))
    scored = await judge.score_response(**sample_inputs)
    assert scored["scores"] == {
        "persona_fidelity": 4,
        "biblical_accuracy": 5,
        "helpfulness": 4,
        "refusal_appropriateness": 5,
    }
    assert "Pauline" in scored["rationale"]


async def test_reply_wrapped_in_code_fence_parses(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    reply = (
        "```json\n"
        '{"scores": {"persona_fidelity": 3, "biblical_accuracy": 3, '
        '"helpfulness": 3, "refusal_appropriateness": 3}, '
        '"rationale": "fair"}\n'
        "```"
    )
    monkeypatch.setattr(judge, "_ollama_chat_json", _stub_chat(reply))
    scored = await judge.score_response(**sample_inputs)
    assert scored["scores"]["persona_fidelity"] == 3


async def test_malformed_json_collapses_to_zero(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    reply = "I refuse to output JSON, here is prose."
    monkeypatch.setattr(judge, "_ollama_chat_json", _stub_chat(reply))
    scored = await judge.score_response(**sample_inputs)
    assert scored["scores"] == {
        "persona_fidelity": 0,
        "biblical_accuracy": 0,
        "helpfulness": 0,
        "refusal_appropriateness": 0,
    }
    assert "refuse to output JSON" in scored["rationale"]


async def test_missing_dimension_becomes_zero(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    reply = (
        '{"scores": {"persona_fidelity": 4, "biblical_accuracy": 5}, '
        '"rationale": "partial scores"}'
    )
    monkeypatch.setattr(judge, "_ollama_chat_json", _stub_chat(reply))
    scored = await judge.score_response(**sample_inputs)
    assert scored["scores"] == {
        "persona_fidelity": 4,
        "biblical_accuracy": 5,
        "helpfulness": 0,
        "refusal_appropriateness": 0,
    }


async def test_out_of_range_score_clamped_to_zero(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    reply = (
        '{"scores": {"persona_fidelity": 9, "biblical_accuracy": -2, '
        '"helpfulness": 4, "refusal_appropriateness": 5}, '
        '"rationale": "oob"}'
    )
    monkeypatch.setattr(judge, "_ollama_chat_json", _stub_chat(reply))
    scored = await judge.score_response(**sample_inputs)
    # Out-of-range values are treated as missing — safer than trusting
    # a judge that has already violated the contract.
    assert scored["scores"]["persona_fidelity"] == 0
    assert scored["scores"]["biblical_accuracy"] == 0
    assert scored["scores"]["helpfulness"] == 4


async def test_transport_error_collapses_to_zero(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    async def _boom(**_kw: Any) -> str:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(judge, "_ollama_chat_json", _boom)
    scored = await judge.score_response(**sample_inputs)
    assert scored["scores"] == {
        "persona_fidelity": 0,
        "biblical_accuracy": 0,
        "helpfulness": 0,
        "refusal_appropriateness": 0,
    }
    assert "connection refused" in scored["rationale"]
