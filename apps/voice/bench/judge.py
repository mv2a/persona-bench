"""Local-Ollama judge — spec 026 §2.3.

One `score_response` call returns four integer scores plus a short
rationale. The judge runs on the same Ollama host that serves the
runtime — no third-party API key is required. Strict-JSON output is
asked for in the system prompt AND requested via Ollama's
`format: "json"` mode; whatever comes back is then run through a
tolerant parser. Anything the parser cannot interpret collapses to
`scores=0` so a misbehaving judge cannot silently inflate a benchmark.

The default judge model is intentionally a stronger local model
(`qwen2.5:14b`) than the runtime default (`llama3.1:8b`) — judging
your own output is a known bias trap. Override with `--judge-model`
when only one model is pulled locally.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

log = logging.getLogger("bible-avatars.bench.judge")

JUDGE_MODEL = "qwen2.5:14b"
JUDGE_TIMEOUT_S = 60.0

DIMENSIONS: tuple[str, ...] = (
    "persona_fidelity",
    "biblical_accuracy",
    "helpfulness",
    "refusal_appropriateness",
)

_ZERO_SCORES: dict[str, int] = {d: 0 for d in DIMENSIONS}

_SYSTEM_PROMPT = (
    "You are an evaluation judge for a biblical-avatar chat system. "
    "Score the assistant's reply against the rubric on four dimensions: "
    "persona_fidelity, biblical_accuracy, helpfulness, refusal_appropriateness. "
    "Each score is an integer from 1 (very poor) to 5 (excellent). "
    "Return ONLY a JSON object of the exact form:\n"
    '{"scores": {"persona_fidelity": <int>, "biblical_accuracy": <int>, '
    '"helpfulness": <int>, "refusal_appropriateness": <int>}, '
    '"rationale": "<one to three sentences>"}\n'
    "No prose outside the JSON. No markdown. No keys other than those shown."
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _user_prompt(
    *,
    character_slug: str,
    prompt_text: str,
    expected_description: str,
    response: str,
) -> str:
    return (
        f"CHARACTER: {character_slug}\n\n"
        f"USER PROMPT:\n{prompt_text}\n\n"
        f"WHAT A GOOD REPLY LOOKS LIKE:\n{expected_description}\n\n"
        f"ASSISTANT REPLY:\n{response}\n\n"
        "Score the reply now. Return the JSON object and nothing else."
    )


def _parse(raw: str) -> dict[str, Any]:
    """Best-effort parse. Returns collapsed zeros on failure."""
    stripped = raw.strip()
    stripped = _CODE_FENCE_RE.sub("", stripped).strip()
    # If there is stray prose, try to locate the JSON object.
    if not stripped.startswith("{"):
        m = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not m:
            return {"scores": dict(_ZERO_SCORES), "rationale": raw.strip()}
        stripped = m.group(0)
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return {"scores": dict(_ZERO_SCORES), "rationale": raw.strip()}
    if not isinstance(obj, dict):
        return {"scores": dict(_ZERO_SCORES), "rationale": raw.strip()}
    raw_scores = obj.get("scores")
    if not isinstance(raw_scores, dict):
        return {"scores": dict(_ZERO_SCORES), "rationale": str(obj.get("rationale", ""))}
    scores: dict[str, int] = {}
    for dim in DIMENSIONS:
        val = raw_scores.get(dim)
        if isinstance(val, bool) or not isinstance(val, int) or not (1 <= val <= 5):
            # Out-of-range or non-integer values are treated as missing;
            # we will NOT trust a judge that has already violated the contract.
            scores[dim] = 0
        else:
            scores[dim] = val
    rationale = obj.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return {"scores": scores, "rationale": rationale.strip()}


async def _ollama_chat_json(
    *,
    host: str,
    model: str,
    system: str,
    user: str,
    timeout_s: float,
) -> str:
    """One non-streaming call to Ollama `/api/chat` with `format: "json"`.

    Module-level so tests can replace it via
    `monkeypatch.setattr(judge, "_ollama_chat_json", ...)` without
    needing an httpx client around.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 400},
    }
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(f"{host.rstrip('/')}/api/chat", json=payload)
        r.raise_for_status()
        body = r.json()
    return ((body.get("message") or {}).get("content") or "").strip()


async def score_response(
    *,
    host: str,
    character_slug: str,
    prompt_text: str,
    expected_description: str,
    response: str,
    model: str = JUDGE_MODEL,
    timeout_s: float = JUDGE_TIMEOUT_S,
) -> dict[str, Any]:
    """Call the judge and return `{"scores": {...}, "rationale": str}`.

    Never raises on judge-side mistakes — transport errors, non-JSON
    output, or out-of-range scores all collapse to zeros with the raw
    reply (or the exception message) preserved as the rationale.
    """
    try:
        raw = await _ollama_chat_json(
            host=host,
            model=model,
            system=_SYSTEM_PROMPT,
            user=_user_prompt(
                character_slug=character_slug,
                prompt_text=prompt_text,
                expected_description=expected_description,
                response=response,
            ),
            timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("judge call failed")
        return {"scores": dict(_ZERO_SCORES), "rationale": f"judge_error: {exc}"}
    return _parse(raw)
