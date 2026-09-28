# Changelog

## 0.1.0: first public release, 28 September 2026

Licensed under the Apache License 2.0. Extracted on 25 September 2026.

The history up to commit `9c143b6` (27 April 2026) is the historical record of the
harness. Everything below was done on or after 25 September 2026.

### Changed
- Moved the harness from `apps/voice/bench/` to `persona_bench/`, and its report schema
  to `schemas/`, with `git mv`, so the history follows.
- The rubric is data (`rubric.py`, `--rubric`). The default keeps the original four
  dimensions, with the domain-specific accuracy dimension renamed `source_fidelity`.
- Personas come from a JSON file (`--personas`) instead of the original application's
  character modules. `--no-scaffold` is an alias for `--no-lever-1`.
- Results record their `character_slug`, and runs record their `rubric`. The report
  renderer reads dimensions and personas from the report, and the schema keys scores by
  rubric dimension.
- Corrected the runner's docstring, which named a hosted judge. The judge is a local
  Ollama model.

### Added
- Fictional example personas, prompts and a rubric (`examples/`).
- Tests for the rubric, the personas adapter, the renderer and the examples, and an
  offline end-to-end run of the CLI. Pytest now runs the historical async tests, which
  had been skipped: 39 tests in all.
- README with provenance, AI-assistance disclosure and limitations; LICENSE (Apache-2.0) and NOTICE;
  CITATION.cff; .zenodo.json; packaging with console scripts; CI.
