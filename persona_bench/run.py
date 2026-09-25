"""Bench harness CLI — spec 026 §2.3.

Runs each prompt in `bench/prompts.json` through a target Ollama
model, scores each reply with the Claude Haiku judge, and writes a
JSON report to `--output`. A per-prompt timeout becomes an error-tagged
result (scores all 0) so a single stall cannot abort a 30-prompt run.

Typical invocations live in `bench/README` and the spec quickstart;
this module exposes `evaluate_prompt` directly so tests can exercise
the per-prompt wrapper without spinning up the full pipeline.
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

log = logging.getLogger("bible-avatars.bench.run")

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS = BENCH_DIR / "prompts.json"
DEFAULT_REPORT_DIR = BENCH_DIR / "reports"
DEFAULT_TIMEOUT_S = 30.0
ZERO_SCORES = {
    "persona_fidelity": 0,
    "biblical_accuracy": 0,
    "helpfulness": 0,
    "refusal_appropriateness": 0,
}


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
) -> dict[str, Any]:
    """Run one prompt end-to-end. Never raises on model/judge errors.

    `asyncio.CancelledError` is re-raised so shutdown semantics work.
    """
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
            "model": model,
            "lever_1_applied": lever_1_applied,
            "response": "",
            "scores": dict(ZERO_SCORES),
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
            "model": model,
            "lever_1_applied": lever_1_applied,
            "response": "",
            "scores": dict(ZERO_SCORES),
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
        "model": model,
        "lever_1_applied": lever_1_applied,
        "response": response,
        "scores": scored.get("scores", dict(ZERO_SCORES)),
        "rationale": scored.get("rationale", ""),
        "timestamp": _now_iso(),
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# CLI plumbing — only imported lazily inside `main()` so tests that stub
# `evaluate_prompt` don't need httpx/anthropic on the path.


class _HttpxOllama:
    """Thin adapter over an Ollama `/api/chat` endpoint.

    Uses `app.llm.build_system_prompt` to keep the bench and the runtime
    in lock-step on prompt composition (when `lever_1_applied` is True).
    """

    def __init__(self, host: str, timeout_s: float) -> None:
        import httpx  # local import

        self._host = host.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def call_prompt(
        self,
        *,
        character_slug: str,
        prompt_text: str,
        model: str,
        lever_1_applied: bool,
    ) -> str:
        from app.characters import get_character
        from app.llm import build_system_prompt

        character = get_character(character_slug)
        if lever_1_applied:
            system = build_system_prompt(character, rag_context=None)
        else:
            system = character.core_prompt
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


async def _run_cli(args: argparse.Namespace) -> int:
    from . import judge as judge_mod

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if args.characters:
        allowed = {c.strip() for c in args.characters.split(",") if c.strip()}
        prompts = [p for p in prompts if p["character_slug"] in allowed]
    if not prompts:
        log.error("no prompts selected")
        return 2

    ollama = _HttpxOllama(host=args.ollama_host, timeout_s=args.timeout_s)
    judge_host = args.judge_host or args.ollama_host

    async def score(**kw: Any) -> dict[str, Any]:
        return await judge_mod.score_response(
            host=judge_host,
            model=args.judge_model,
            timeout_s=args.judge_timeout_s,
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
        },
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%d results)", out, len(results))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="bible-avatars bench harness")
    p.add_argument("--model", required=True, help="Ollama model tag, e.g. llama3.1:8b")
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON report (parent dirs created).",
    )
    p.add_argument(
        "--prompts",
        default=str(DEFAULT_PROMPTS),
        help="Path to prompts.json (default: bench/prompts.json).",
    )
    p.add_argument(
        "--characters",
        default="",
        help="Comma-separated character slugs to filter (default: all).",
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
        action="store_true",
        help="Disable Lever 1 prompt scaffolding (baseline measurement).",
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
