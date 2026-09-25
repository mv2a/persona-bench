"""Report rendering — spec 026 §2.3.

Two modes:

* default: consume a single JSON report and emit a markdown summary
  (per-dimension means, per-character means, failure counts).
* `--diff A.json B.json`: tabulate the delta between two reports so a
  Lever-1-on vs Lever-1-off (or model-A vs model-B) comparison is one
  command.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .judge import DIMENSIONS


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _per_dimension_means(results: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for dim in DIMENSIONS:
        vals = [r["scores"].get(dim, 0) for r in results]
        out[dim] = round(statistics.mean(vals), 2) if vals else 0.0
    return out


def _per_character_means(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, int]]] = defaultdict(list)
    for r in results:
        slug = r["prompt_id"].split("-", 1)[0]
        buckets[slug].append(r["scores"])
    out: dict[str, dict[str, float]] = {}
    for slug, rows in sorted(buckets.items()):
        per_dim = {}
        for dim in DIMENSIONS:
            vals = [row.get(dim, 0) for row in rows]
            per_dim[dim] = round(statistics.mean(vals), 2) if vals else 0.0
        out[slug] = per_dim
    return out


def _error_count(results: list[dict[str, Any]]) -> int:
    return sum(1 for r in results if r.get("error"))


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results", [])
    return {
        "model": report.get("run", {}).get("model"),
        "lever_1_applied": report.get("run", {}).get("lever_1_applied"),
        "prompt_count": len(results),
        "error_count": _error_count(results),
        "per_dimension_mean": _per_dimension_means(results),
        "per_character_mean": _per_character_means(results),
    }


def _fmt_dim_table(header: str, row: dict[str, float]) -> list[str]:
    lines = [f"### {header}", "", "| dimension | mean |", "|---|---|"]
    for dim in DIMENSIONS:
        lines.append(f"| {dim} | {row.get(dim, 0):.2f} |")
    lines.append("")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    s = summarize(report)
    run = report.get("run", {})
    lines = [
        f"# Bench report — {s['model']}",
        "",
        f"- prompts: **{s['prompt_count']}**, errors: **{s['error_count']}**",
        f"- lever_1_applied: **{s['lever_1_applied']}**",
        f"- started_at: {run.get('started_at', '?')}",
        f"- finished_at: {run.get('finished_at', '?')}",
        f"- judge_model: {run.get('judge_model', '?')}",
        "",
    ]
    lines += _fmt_dim_table("Per-dimension mean", s["per_dimension_mean"])
    lines += ["## Per-character mean", ""]
    lines += ["| character | " + " | ".join(DIMENSIONS) + " |"]
    lines += ["|" + "---|" * (len(DIMENSIONS) + 1)]
    for slug, row in s["per_character_mean"].items():
        cells = [f"{row.get(dim, 0):.2f}" for dim in DIMENSIONS]
        lines.append(f"| {slug} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_diff(report_a: dict[str, Any], report_b: dict[str, Any]) -> str:
    a, b = summarize(report_a), summarize(report_b)
    lines = [
        f"# Bench diff — {a['model']} → {b['model']}",
        "",
        f"- A: {a['model']} (lever_1={a['lever_1_applied']}, n={a['prompt_count']})",
        f"- B: {b['model']} (lever_1={b['lever_1_applied']}, n={b['prompt_count']})",
        "",
        "## Per-dimension mean (B − A)",
        "",
        "| dimension | A | B | Δ |",
        "|---|---|---|---|",
    ]
    for dim in DIMENSIONS:
        av = a["per_dimension_mean"].get(dim, 0.0)
        bv = b["per_dimension_mean"].get(dim, 0.0)
        lines.append(f"| {dim} | {av:.2f} | {bv:.2f} | {bv - av:+.2f} |")
    lines.append("")
    lines += ["## Per-character mean (B − A, averaged across dimensions)", ""]
    lines += ["| character | A | B | Δ |", "|---|---|---|---|"]
    all_slugs = sorted(set(a["per_character_mean"]) | set(b["per_character_mean"]))
    for slug in all_slugs:
        row_a = a["per_character_mean"].get(slug, {})
        row_b = b["per_character_mean"].get(slug, {})
        avg_a = (
            round(statistics.mean([row_a.get(d, 0) for d in DIMENSIONS]), 2)
            if row_a
            else 0.0
        )
        avg_b = (
            round(statistics.mean([row_b.get(d, 0) for d in DIMENSIONS]), 2)
            if row_b
            else 0.0
        )
        lines.append(f"| {slug} | {avg_a:.2f} | {avg_b:.2f} | {avg_b - avg_a:+.2f} |")
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="bible-avatars bench report renderer")
    p.add_argument("report", nargs="?", help="Path to a single report JSON.")
    p.add_argument(
        "--diff",
        nargs=2,
        metavar=("A", "B"),
        help="Render B − A instead of a single summary.",
    )
    p.add_argument(
        "--output",
        help="Write rendered markdown to this path; default prints to stdout.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.diff:
        a, b = _load(Path(args.diff[0])), _load(Path(args.diff[1]))
        text = render_diff(a, b)
    elif args.report:
        text = render_markdown(_load(Path(args.report)))
    else:
        _build_parser().print_help()
        return 2
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
