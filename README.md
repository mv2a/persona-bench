# persona-rubric-bench

A small harness for scoring persona chat models. It sends each prompt to a local model
under a persona's system prompt, asks a second local model to score the reply against a
rubric, and writes a JSON report that can be summarised, or compared with another run.

> **Status: research software, version 0.1.2, under the [Apache License 2.0](LICENSE). First
> released publicly as 0.1.0 on 28 September 2026 under the name persona-bench. Renamed the same
> day to avoid confusion with SynthLabs' unrelated
> [PERSONA-bench](https://github.com/SynthLabsAI/PERSONA-bench).**

## How it works

```
prompts.json ──┐
personas.json ─┼─► target model (Ollama) ─► reply ─► judge model (Ollama) ─► scores + rationale
rubric.json ───┘         system prompt from the persona         scored against the rubric
                                                                          │
                                           report.json ◄──────────────────┘
                                                │
                              persona-rubric-bench-report  ─►  markdown summary, or A-versus-B diff
```

- **Personas** (`examples/personas.json`): each has a `system_prompt`, and optionally a
  `scaffold`, extra instructions appended unless `--no-scaffold` is given. That is the
  switch the harness was built to measure: the same persona with and without prompt
  scaffolding.
- **Rubric** (`examples/rubric.json`, which is also the built-in default): the dimensions
  the judge scores from 1 to 5. The default has four: `persona_fidelity`,
  `source_fidelity`, `helpfulness` and `refusal_appropriateness`.
- **Judge** (`persona_rubric_bench/judge.py`): a local Ollama model, `qwen2.5:14b` by default,
  deliberately stronger than a typical target. It is asked for strict JSON. Anything it
  returns that breaks the contract (non-JSON output, missing dimensions, out-of-range
  values, transport errors) collapses to 0 for the affected dimensions rather than being
  guessed at.
- **Runner** (`persona_rubric_bench/run.py`): one prompt at a time, each with its own timeout. A
  timeout or model error becomes an error-tagged result with zero scores, and the run
  continues.
- **Report** (`schemas/bench-report.schema.json`): per-prompt results, with the reply, the
  scores, the rationale, the latency and the persona, plus the run's model, judge,
  rubric and timestamps.

The examples are three **fictional** personas: a lighthouse keeper in 1893, a
clock-museum docent and a field botanist in 1966. Each system prompt defines a small,
closed source of facts, so that `source_fidelity` can be judged. There are twelve
prompts, one per persona in each rubric category: in-character, source-grounded,
adversarial and out-of-scope.

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m pytest            # 39 tests, no network or model needed
```

Running a benchmark needs [Ollama](https://ollama.com) with the target and judge models
pulled:

```bash
persona-rubric-bench --model llama3.1:8b \
  --prompts examples/prompts.json --personas examples/personas.json \
  --output runs/llama31-scaffold.json

persona-rubric-bench --model llama3.1:8b --no-scaffold \
  --prompts examples/prompts.json --personas examples/personas.json \
  --output runs/llama31-baseline.json

persona-rubric-bench-report runs/llama31-scaffold.json
persona-rubric-bench-report --diff runs/llama31-baseline.json runs/llama31-scaffold.json
```

Options: `--rubric` (a rubric JSON), `--characters` (a subset of personas),
`--ollama-host`, `--judge-model`, `--judge-host`, `--timeout-s` and `--judge-timeout-s`.
No results are committed with the repository.

## Provenance

This repository was extracted on **25 September 2026** from a private repository, a
persona voice-chat application whose characters are drawn from a religious text. The
extraction kept only the harness and its report schema, with their original commits.

| | Date |
|---|---|
| The harness written, in private | **27 April 2026** (3 commits, 17:26 to 19:58 US Eastern) |
| Extraction and preparation for release | from 25 September 2026 (commits after `9c143b6`) |
| GitHub repository created (private) | 25 September 2026 |
| First public release | **28 September 2026** (v0.1.0) |

The three historical commits are the harness's first commit; a commit to the same
application that accidentally deleted it through a squash-merge race; and the commit
that restored it. The extraction changed only which paths are kept. Dates, authors,
messages and co-author trailers are unchanged, so the messages still describe the
original application's features. A verified archive of the original repository, the
mapping from each commit here to its original, and the hosting provider's server-side
push log are held privately and can be produced for verification.

Nothing here is the original application's content. Its personas, prompts, rubric and
benchmark results were left out. The history still shows the harness's first form: a
domain-specific accuracy dimension, and imports of the application's character
definitions. Both were replaced after extraction; see [CHANGELOG.md](CHANGELOG.md).

## AI assistance

This software was developed with AI coding assistants. The three historical commits are
squash-merges whose messages carry `Co-Authored-By` trailers naming Claude Sonnet 4.6.
The commits made while preparing this release carry one naming Claude Opus 5.5. The
trailers record which work was machine-assisted, and they measure nothing else. The
author's contribution is the evaluation design: what is measured, the rubric, the
decision that a judge's contract violations score zero rather than being interpreted,
and the choice of a stronger local judge than the model under test.

## Limitations

- **An LLM judges an LLM.** Scores come from one judge model, with no calibration against
  human ratings. A judge from the same family as the target may favour it, and the
  default judge is chosen to be stronger than the default target for that reason. Treat
  scores as relative comparisons between runs, not absolute quality.
- **Averages include failures.** A failed prompt records 0 on every dimension, and the
  per-dimension and per-persona means include those zeros. The report states the error
  count, so read the means alongside it.
- **One sample per prompt.** Target replies use temperature 0.5 and judge calls use 0.1,
  and each prompt runs once. There is no repeated sampling and no confidence interval.
- **Ordinal scores reported as means.** The 1 to 5 scale is ordinal, and means are a
  convenience, not a measurement.
- **source_fidelity is only as good as the persona's defined source.** There is no
  retrieval. Facts are checked against what the system prompt states.
- **Ollama only.** Both the target and the judge are called through Ollama's `/api/chat`,
  one prompt at a time.
- **The examples are small and fictional.** Twelve prompts show the format. They are not
  a benchmark of anything.
- **The history is short.** Three commits on one day. The harness's later use in the
  original application is not part of this repository.

## Citing

See [CITATION.cff](CITATION.cff). The software is archived on Zenodo: [10.5281/zenodo.23029797](https://doi.org/10.5281/zenodo.23029797) covers all versions, and v0.1.2 is [10.5281/zenodo.23029798](https://doi.org/10.5281/zenodo.23029798).
