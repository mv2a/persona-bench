"""The rubric is data: its dimensions drive the judge prompt, the parser and the zeros."""

from __future__ import annotations

import json

import pytest

from persona_bench.rubric import DEFAULT_RUBRIC, Rubric, load_rubric


def test_default_rubric_keeps_the_original_four_dimension_shape() -> None:
    assert DEFAULT_RUBRIC.dimensions == (
        "persona_fidelity",
        "source_fidelity",
        "helpfulness",
        "refusal_appropriateness",
    )
    assert DEFAULT_RUBRIC.zero_scores() == {d: 0 for d in DEFAULT_RUBRIC.dimensions}


def test_system_prompt_names_every_dimension_and_the_json_shape() -> None:
    prompt = DEFAULT_RUBRIC.system_prompt()
    for dim in DEFAULT_RUBRIC.dimensions:
        assert f'"{dim}": <int>' in prompt
    assert "integer from 1 (very poor) to 5 (excellent)" in prompt
    assert DEFAULT_RUBRIC.context in prompt


def test_load_rubric_from_json(tmp_path) -> None:
    path = tmp_path / "rubric.json"
    path.write_text(
        json.dumps(
            {
                "name": "tutor",
                "dimensions": ["clarity", "accuracy"],
                "context": "a tutoring bot",
                "categories": ["easy", "hard"],
                "descriptions": {"clarity": "is it easy to follow"},
            }
        )
    )
    rubric = load_rubric(path)
    assert rubric.dimensions == ("clarity", "accuracy")
    assert rubric.categories == ("easy", "hard")
    assert "- clarity: is it easy to follow" in rubric.system_prompt()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dimensions": ()}, "at least one dimension"),
        ({"dimensions": ("a_b", "a_b")}, "duplicate"),
        ({"dimensions": ("Clarity",)}, "lower_snake_case"),
        ({"dimensions": ("clarity",), "descriptions": {"other": "x"}}, "unknown dimensions"),
        ({"dimensions": ("clarity",), "context": "  "}, "context"),
    ],
)
def test_invalid_rubrics_are_rejected(kwargs: dict, message: str) -> None:
    args = {"name": "x", "context": "a bot", **kwargs}
    with pytest.raises(ValueError, match=message):
        Rubric(**args)
