"""T028 — the judge parses strict JSON, collapses gracefully on failure.

No network. `judge._ollama_chat_json` is monkeypatched to return canned
text so we exercise `score_response`'s parsing logic in isolation.
"""

from __future__ import annotations

from typing import Any

import pytest

from persona_rubric_bench import judge


def _stub_chat(reply_text: str):
    async def _fn(**_kwargs: Any) -> str:
        return reply_text

    return _fn


@pytest.fixture
def sample_inputs() -> dict[str, str]:
    return {
        "host": "http://localhost:11434",
        "character_slug": "keeper",
        "prompt_text": "When was the light first lit?",
        "expected_description": "First person. The logbook: first lit on 1 March 1887.",
        "response": "I lit her myself, the first of March, eighteen eighty-seven…",
    }


async def test_valid_reply_parses(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    reply = (
        '{"scores": {"persona_fidelity": 4, "source_fidelity": 5, '
        '"helpfulness": 4, "refusal_appropriateness": 5}, '
        '"rationale": "In voice; the logbook date is right."}'
    )
    monkeypatch.setattr(judge, "_ollama_chat_json", _stub_chat(reply))
    scored = await judge.score_response(**sample_inputs)
    assert scored["scores"] == {
        "persona_fidelity": 4,
        "source_fidelity": 5,
        "helpfulness": 4,
        "refusal_appropriateness": 5,
    }
    assert "logbook" in scored["rationale"]


async def test_reply_wrapped_in_code_fence_parses(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    reply = (
        "```json\n"
        '{"scores": {"persona_fidelity": 3, "source_fidelity": 3, '
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
        "source_fidelity": 0,
        "helpfulness": 0,
        "refusal_appropriateness": 0,
    }
    assert "refuse to output JSON" in scored["rationale"]


async def test_missing_dimension_becomes_zero(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    reply = (
        '{"scores": {"persona_fidelity": 4, "source_fidelity": 5}, '
        '"rationale": "partial scores"}'
    )
    monkeypatch.setattr(judge, "_ollama_chat_json", _stub_chat(reply))
    scored = await judge.score_response(**sample_inputs)
    assert scored["scores"] == {
        "persona_fidelity": 4,
        "source_fidelity": 5,
        "helpfulness": 0,
        "refusal_appropriateness": 0,
    }


async def test_out_of_range_score_clamped_to_zero(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    reply = (
        '{"scores": {"persona_fidelity": 9, "source_fidelity": -2, '
        '"helpfulness": 4, "refusal_appropriateness": 5}, '
        '"rationale": "oob"}'
    )
    monkeypatch.setattr(judge, "_ollama_chat_json", _stub_chat(reply))
    scored = await judge.score_response(**sample_inputs)
    # Out-of-range values are treated as missing — safer than trusting
    # a judge that has already violated the contract.
    assert scored["scores"]["persona_fidelity"] == 0
    assert scored["scores"]["source_fidelity"] == 0
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
        "source_fidelity": 0,
        "helpfulness": 0,
        "refusal_appropriateness": 0,
    }
    assert "connection refused" in scored["rationale"]


async def test_custom_rubric_drives_prompt_and_parsing(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    from persona_rubric_bench.rubric import Rubric

    rubric = Rubric(name="two", dimensions=("clarity", "accuracy"), context="a tutoring bot")
    seen: dict[str, str] = {}

    async def _capture(**kwargs: Any) -> str:
        seen["system"] = kwargs["system"]
        return '{"scores": {"clarity": 5, "accuracy": 2, "persona_fidelity": 4}, "rationale": "r"}'

    monkeypatch.setattr(judge, "_ollama_chat_json", _capture)
    scored = await judge.score_response(rubric=rubric, **sample_inputs)
    # Only the rubric's dimensions are kept; anything else the judge adds is ignored.
    assert scored["scores"] == {"clarity": 5, "accuracy": 2}
    assert "a tutoring bot" in seen["system"] and '"clarity": <int>' in seen["system"]


async def test_custom_rubric_transport_error_collapses_to_its_zeros(
    monkeypatch: pytest.MonkeyPatch, sample_inputs: dict[str, str]
) -> None:
    from persona_rubric_bench.rubric import Rubric

    async def _boom(**_kw: Any) -> str:
        raise RuntimeError("down")

    monkeypatch.setattr(judge, "_ollama_chat_json", _boom)
    rubric = Rubric(name="one", dimensions=("clarity",), context="a tutoring bot")
    scored = await judge.score_response(rubric=rubric, **sample_inputs)
    assert scored["scores"] == {"clarity": 0}
