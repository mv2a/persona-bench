"""Bench harness CLI — spec 026 §2.3.

Runs each prompt in a prompts file through a target Ollama model, with each
persona's system prompt taken from a personas file, scores each reply with the
local Ollama judge against a rubric, and writes a JSON report to `--output`. A
per-prompt timeout becomes an error-tagged result (scores all 0) so a single
stall cannot abort a run.

This module exposes `evaluate_prompt` directly so tests can exercise the
per-prompt wrapper without spinning up the full pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .rubric import DEFAULT_RUBRIC, Rubric, load_rubric

log = logging.getLogger("persona_bench.run")

DEFAULT_TIMEOUT_S = 30.0
ZERO_SCORES = DEFAULT_RUBRIC.zero_scores()


class OllamaLike(Protocol):
    async def call_prompt(
        self,
        *,
        character_slug: str,
        prompt_text: str,
        model: str,
        lever_1_applied: bool,
    ) -> str: ...


JudgeFn = Callable[..., Awaitable[dict[str, Any]]]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def evaluate_prompt(
    *,
    prompt: dict[str, Any],
    model: str,
    lever_1_applied: bool,
    ollama: OllamaLike,
    score: JudgeFn,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    zero_scores: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run one prompt end-to-end. Never raises on model/judge errors.

    `asyncio.CancelledError` is re-raised so shutdown semantics work.
    `zero_scores` is what a failed prompt records; it defaults to the default
    rubric's dimensions and should match the rubric the judge scores against.
    """
    zeros = dict(zero_scores if zero_scores is not None else ZERO_SCORES)
    t0 = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            ollama.call_prompt(
                character_slug=prompt["character_slug"],
                prompt_text=prompt["prompt_text"],
                model=model,
                lever_1_applied=lever_1_applied,
            ),
            timeout=timeout_s,
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "prompt_id": prompt["id"],
            "character_slug": prompt["character_slug"],
            "model": model,
            "lever_1_applied": lever_1_applied,
            "response": "",
            "scores": dict(zeros),
            "rationale": f"ollama_timeout: {exc}",
            "timestamp": _now_iso(),
            "latency_ms": latency_ms,
            "error": "ollama_timeout",
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        log.exception("ollama call failed for %s", prompt["id"])
        return {
            "prompt_id": prompt["id"],
            "character_slug": prompt["character_slug"],
            "model": model,
            "lever_1_applied": lever_1_applied,
            "response": "",
            "scores": dict(zeros),
            "rationale": f"ollama_error: {exc}",
            "timestamp": _now_iso(),
            "latency_ms": latency_ms,
            "error": "ollama_error",
        }

    latency_ms = int((time.perf_counter() - t0) * 1000)
    scored = await score(
        character_slug=prompt["character_slug"],
        prompt_text=prompt["prompt_text"],
        expected_description=prompt["expected_description"],
        response=response,
    )
    return {
        "prompt_id": prompt["id"],
        "character_slug": prompt["character_slug"],
        "model": model,
        "lever_1_applied": lever_1_applied,
        "response": response,
        "scores": scored.get("scores", dict(zeros)),
        "rationale": scored.get("rationale", ""),
        "timestamp": _now_iso(),
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# CLI plumbing — only imported lazily inside `main()` so tests that stub
# `evaluate_prompt` don't need httpx/anthropic on the path.


class _HttpxOllama:
    """Thin adapter over an Ollama `/api/chat` endpoint.

    Each persona supplies a `system_prompt`. When `lever_1_applied` is True and
    the persona also supplies a `scaffold`, the scaffold is appended to the
    system prompt; that is the generic form of the original application's
    "Lever 1" prompt scaffolding, whose effect the bench was built to measure.
    """

    def __init__(
        self,
        host: str,
        timeout_s: float,
        personas: dict[str, dict[str, str]],
        client: Any = None,
    ) -> None:
        if client is None:
            import httpx  # local import

            client = httpx.AsyncClient(timeout=timeout_s)
        self._host = host.rstrip("/")
        self._client = client
        self._personas = personas

    async def aclose(self) -> None:
        await self._client.aclose()

    def system_prompt(self, character_slug: str, lever_1_applied: bool) -> str:
        try:
            persona = self._personas[character_slug]
        except KeyError:
            raise ValueError(f"unknown persona {character_slug!r}") from None
        system = persona["system_prompt"]
        if lever_1_applied and persona.get("scaffold"):
            system = f"{system}\n\n{persona['scaffold']}"
        return system

    async def call_prompt(
        self,
        *,
        character_slug: str,
        prompt_text: str,
        model: str,
        lever_1_applied: bool,
    ) -> str:
        system = self.system_prompt(character_slug, lever_1_applied)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt_text},
            ],
            "stream": False,
            "options": {"temperature": 0.5, "top_p": 0.9, "num_predict": 280},
        }
        r = await self._client.post(f"{self._host}/api/chat", json=payload)
        r.raise_for_status()
        body = r.json()
        return ((body.get("message") or {}).get("content") or "").strip()


def load_personas(path: str | Path) -> dict[str, dict[str, str]]:
    """Load personas from JSON: an object mapping each slug to at least a
    `system_prompt`, and optionally a `name` and a `scaffold`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("personas file must be a non-empty JSON object")
    for slug, persona in data.items():
        if not isinstance(persona, dict) or not str(persona.get("system_prompt", "")).strip():
            raise ValueError(f"persona {slug!r} needs a non-empty system_prompt")
    return data


async def _run_cli(args: argparse.Namespace) -> int:
    from . import judge as judge_mod

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    personas = load_personas(args.personas)
    rubric = load_rubric(args.rubric) if args.rubric else DEFAULT_RUBRIC
    unknown = sorted({p["character_slug"] for p in prompts} - set(personas))
    if unknown:
        log.error("prompts name personas that are not in %s: %s", args.personas, unknown)
        return 2
    if args.characters:
        allowed = {c.strip() for c in args.characters.split(",") if c.strip()}
        prompts = [p for p in prompts if p["character_slug"] in allowed]
    if not prompts:
        log.error("no prompts selected")
        return 2

    ollama = _HttpxOllama(host=args.ollama_host, timeout_s=args.timeout_s, personas=personas)
    judge_host = args.judge_host or args.ollama_host

    async def score(**kw: Any) -> dict[str, Any]:
        return await judge_mod.score_response(
            host=judge_host,
            model=args.judge_model,
            timeout_s=args.judge_timeout_s,
            rubric=rubric,
            **kw,
        )

    started = _now_iso()
    results: list[dict[str, Any]] = []
    try:
        for p in prompts:
            res = await evaluate_prompt(
                prompt=p,
                model=args.model,
                lever_1_applied=not args.no_lever_1,
                ollama=ollama,
                score=score,
                timeout_s=args.timeout_s,
                zero_scores=rubric.zero_scores(),
            )
            results.append(res)
            log.info(
                "%s %s scores=%s latency=%dms",
                args.model,
                p["id"],
                res["scores"],
                res["latency_ms"],
            )
    finally:
        await ollama.aclose()

    finished = _now_iso()
    report = {
        "run": {
            "model": args.model,
            "lever_1_applied": not args.no_lever_1,
            "started_at": started,
            "finished_at": finished,
            "prompt_count": len(results),
            "judge_model": args.judge_model,
            "ollama_host": args.ollama_host,
            "rubric": rubric.name,
        },
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%d results)", out, len(results))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="persona-bench: score persona replies with a local judge")
    p.add_argument("--model", required=True, help="Ollama model tag, e.g. llama3.1:8b")
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON report (parent dirs created).",
    )
    p.add_argument(
        "--prompts",
        required=True,
        help="Path to the prompts JSON (see examples/prompts.json).",
    )
    p.add_argument(
        "--personas",
        required=True,
        help="Path to the personas JSON (see examples/personas.json).",
    )
    p.add_argument(
        "--rubric",
        default=None,
        help="Path to a rubric JSON (default: the built-in persona rubric).",
    )
    p.add_argument(
        "--characters",
        default="",
        help="Comma-separated persona slugs to filter (default: all).",
    )
    p.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Base URL for Ollama (default: http://localhost:11434).",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-prompt timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )
    p.add_argument(
        "--no-lever-1",
        "--no-scaffold",
        dest="no_lever_1",
        action="store_true",
        help="Use each persona's system prompt without its scaffold (baseline measurement).",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Ollama model used as the judge (default: bench.judge.JUDGE_MODEL).",
    )
    p.add_argument(
        "--judge-host",
        default=None,
        help="Ollama host for the judge (default: same as --ollama-host).",
    )
    p.add_argument(
        "--judge-timeout-s",
        type=float,
        default=None,
        help="Per-judge-call timeout in seconds (default: bench.judge.JUDGE_TIMEOUT_S).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    from . import judge as judge_mod

    if args.judge_model is None:
        args.judge_model = judge_mod.JUDGE_MODEL
    if args.judge_timeout_s is None:
        args.judge_timeout_s = judge_mod.JUDGE_TIMEOUT_S
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
