"""The shipped examples are internally consistent, and the CLI runs end to end on them offline.

The end-to-end test replaces the Ollama adapter and the judge with stubs, so it
needs no network and no model. It checks that the report the CLI writes
validates against the published schema.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema

from persona_rubric_bench import judge, run
from persona_rubric_bench.rubric import DEFAULT_RUBRIC, load_rubric

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
SCHEMA = json.loads((ROOT / "schemas" / "bench-report.schema.json").read_text(encoding="utf-8"))


def _prompts() -> list[dict[str, Any]]:
    return json.loads((EXAMPLES / "prompts.json").read_text(encoding="utf-8"))


def test_example_rubric_matches_the_built_in_default() -> None:
    rubric = load_rubric(EXAMPLES / "rubric.json")
    assert rubric == DEFAULT_RUBRIC


def test_example_personas_load_and_are_marked_fictional() -> None:
    personas = run.load_personas(EXAMPLES / "personas.json")
    assert personas
    assert all(p.get("fictional") is True for p in personas.values())


def test_example_prompts_are_consistent() -> None:
    prompts = _prompts()
    personas = run.load_personas(EXAMPLES / "personas.json")
    rubric = load_rubric(EXAMPLES / "rubric.json")
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids)), "prompt ids must be unique"
    for p in prompts:
        assert p["character_slug"] in personas
        assert p["id"].startswith(p["character_slug"] + "-")
        assert p["category"] in rubric.categories
        assert p["prompt_text"].strip() and p["expected_description"].strip()
    # Every persona is exercised in every category.
    pairs = Counter((p["character_slug"], p["category"]) for p in prompts)
    for slug in personas:
        for category in rubric.categories:
            assert pairs[(slug, category)] >= 1, (slug, category)


class _StubOllama:
    def __init__(self, host: str, timeout_s: float, personas: dict) -> None:
        self.personas = personas

    async def call_prompt(self, *, character_slug: str, **_kw: Any) -> str:
        if character_slug == "botanist":
            raise TimeoutError("stub timeout")
        return f"a reply from {self.personas[character_slug]['name']}"

    async def aclose(self) -> None:
        return None


def test_cli_runs_end_to_end_offline(monkeypatch, tmp_path) -> None:
    async def _stub_score(**kw: Any) -> dict[str, Any]:
        return {"scores": {d: 4 for d in kw["rubric"].dimensions}, "rationale": "stub"}

    monkeypatch.setattr(run, "_HttpxOllama", _StubOllama)
    monkeypatch.setattr(judge, "score_response", _stub_score)
    out = tmp_path / "report.json"
    code = run.main(
        [
            "--model", "stub-model",
            "--prompts", str(EXAMPLES / "prompts.json"),
            "--personas", str(EXAMPLES / "personas.json"),
            "--rubric", str(EXAMPLES / "rubric.json"),
            "--judge-model", "stub-judge",
            "--output", str(out),
        ]
    )
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    jsonschema.validate(instance=report, schema=SCHEMA)
    assert report["run"]["prompt_count"] == len(_prompts())
    assert report["run"]["rubric"] == "persona-default"
    errors = [r for r in report["results"] if r.get("error")]
    assert {r["character_slug"] for r in errors} == {"botanist"}
    assert all(r["scores"] == DEFAULT_RUBRIC.zero_scores() for r in errors)


def test_cli_refuses_prompts_for_unknown_personas(tmp_path) -> None:
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{**_prompts()[0], "character_slug": "nobody"}]))
    code = run.main(
        [
            "--model", "m",
            "--prompts", str(prompts),
            "--personas", str(EXAMPLES / "personas.json"),
            "--output", str(tmp_path / "r.json"),
        ]
    )
    assert code == 2
