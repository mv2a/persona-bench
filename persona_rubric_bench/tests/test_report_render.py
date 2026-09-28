"""The renderer reads the dimensions and personas from the report itself."""

from __future__ import annotations

from persona_rubric_bench import report


def _result(pid: str, slug: str | None, scores: dict[str, int]) -> dict:
    r = {"prompt_id": pid, "scores": scores, "latency_ms": 10}
    if slug is not None:
        r["character_slug"] = slug
    return r


def _report(model: str, results: list[dict]) -> dict:
    return {"run": {"model": model, "lever_1_applied": True}, "results": results}


def test_custom_dimensions_and_hyphenated_slugs() -> None:
    rep = _report(
        "m1",
        [
            _result("lighthouse-keeper-1", "lighthouse-keeper", {"clarity": 4, "accuracy": 2}),
            _result("lighthouse-keeper-2", "lighthouse-keeper", {"clarity": 2, "accuracy": 4}),
        ],
    )
    s = report.summarize(rep)
    assert s["per_dimension_mean"] == {"clarity": 3.0, "accuracy": 3.0}
    assert list(s["per_character_mean"]) == ["lighthouse-keeper"]
    md = report.render_markdown(rep)
    assert "| character | clarity | accuracy |" in md


def test_old_reports_without_slug_fall_back_to_the_id_prefix() -> None:
    rep = _report("m1", [_result("keeper-1", None, {"clarity": 5})])
    assert list(report.summarize(rep)["per_character_mean"]) == ["keeper"]


def test_diff_covers_the_union_of_dimensions() -> None:
    a = _report("A", [_result("keeper-1", "keeper", {"clarity": 2})])
    b = _report("B", [_result("keeper-1", "keeper", {"clarity": 4, "accuracy": 3})])
    md = report.render_diff(a, b)
    assert "| clarity | 2.00 | 4.00 | +2.00 |" in md
    assert "| accuracy | 0.00 | 3.00 | +3.00 |" in md
