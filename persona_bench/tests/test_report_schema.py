"""T031 — a representative `BenchmarkReport` validates against the schema.

Built entirely in memory — does not invoke `run.py`, does not touch
Ollama or the Anthropic API. The point is to pin the schema itself so
later `report.py` work cannot drift from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "bench-report.schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _sample_result(prompt_id: str, model: str) -> dict:
    return {
        "prompt_id": prompt_id,
        "model": model,
        "lever_1_applied": True,
        "response": "Peace I leave with thee.",
        "scores": {
            "persona_fidelity": 4,
            "biblical_accuracy": 5,
            "helpfulness": 4,
            "refusal_appropriateness": 5,
        },
        "rationale": "Cadence is in-register; John 14:27 cited correctly.",
        "timestamp": "2026-04-23T14:22:00Z",
        "latency_ms": 1840,
    }


def test_valid_report_passes() -> None:
    report = {
        "run": {
            "model": "llama3.1:8b",
            "lever_1_applied": True,
            "started_at": "2026-04-23T14:20:00Z",
            "finished_at": "2026-04-23T14:35:00Z",
            "prompt_count": 2,
            "judge_model": "claude-haiku-4-5-20251001",
            "ollama_host": "http://black-horse:11434",
        },
        "results": [
            _sample_result("jesus-1", "llama3.1:8b"),
            _sample_result("paul-3", "llama3.1:8b"),
        ],
    }
    jsonschema.validate(instance=report, schema=_schema())


def test_result_with_error_field_still_valid() -> None:
    result = _sample_result("daniel-2", "llama3.1:8b")
    result["response"] = ""
    result["scores"] = {
        "persona_fidelity": 0,
        "biblical_accuracy": 0,
        "helpfulness": 0,
        "refusal_appropriateness": 0,
    }
    result["rationale"] = "Ollama timed out after 30s"
    result["error"] = "ollama_timeout"
    report = {
        "run": {
            "model": "llama3.1:8b",
            "lever_1_applied": True,
            "started_at": "2026-04-23T14:20:00Z",
            "finished_at": "2026-04-23T14:35:00Z",
            "prompt_count": 1,
        },
        "results": [result],
    }
    jsonschema.validate(instance=report, schema=_schema())


def test_out_of_range_score_rejected() -> None:
    result = _sample_result("jesus-1", "llama3.1:8b")
    result["scores"]["biblical_accuracy"] = 9  # > 5
    report = {
        "run": {
            "model": "llama3.1:8b",
            "lever_1_applied": True,
            "started_at": "2026-04-23T14:20:00Z",
            "finished_at": "2026-04-23T14:35:00Z",
            "prompt_count": 1,
        },
        "results": [result],
    }
    import pytest

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=report, schema=_schema())
