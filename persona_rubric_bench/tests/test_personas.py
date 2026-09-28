"""Personas come from a file, and the adapter sends each one's system prompt to Ollama.

The HTTP call goes through httpx's MockTransport: no network, no Ollama.
"""

from __future__ import annotations

import json

import httpx
import pytest

from persona_rubric_bench.run import _HttpxOllama, load_personas

PERSONAS = {
    "keeper": {
        "name": "Maren Holt",
        "system_prompt": "You are Maren Holt, keeper of the Skerry Point light.",
        "scaffold": "Answer in two sentences at most.",
    },
    "docent": {"system_prompt": "You are a docent at a clock museum."},
}


def _adapter(handler) -> _HttpxOllama:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return _HttpxOllama(host="http://ollama.test/", timeout_s=5, personas=PERSONAS, client=client)


def test_system_prompt_with_and_without_scaffold() -> None:
    adapter = _HttpxOllama(host="http://x", timeout_s=1, personas=PERSONAS, client=object())
    assert adapter.system_prompt("keeper", lever_1_applied=False) == PERSONAS["keeper"]["system_prompt"]
    scaffolded = adapter.system_prompt("keeper", lever_1_applied=True)
    assert scaffolded.startswith(PERSONAS["keeper"]["system_prompt"])
    assert scaffolded.endswith("Answer in two sentences at most.")
    # A persona without a scaffold is unaffected by the flag.
    assert adapter.system_prompt("docent", lever_1_applied=True) == PERSONAS["docent"]["system_prompt"]


def test_unknown_persona_is_an_error() -> None:
    adapter = _HttpxOllama(host="http://x", timeout_s=1, personas=PERSONAS, client=object())
    with pytest.raises(ValueError, match="unknown persona"):
        adapter.system_prompt("nobody", lever_1_applied=False)


async def test_call_prompt_posts_the_persona_and_returns_the_reply() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "  The light is lit.  "}})

    adapter = _adapter(handler)
    reply = await adapter.call_prompt(
        character_slug="keeper", prompt_text="Is the light lit?", model="m", lever_1_applied=True
    )
    await adapter.aclose()
    assert reply == "The light is lit."
    assert seen["url"] == "http://ollama.test/api/chat"
    assert seen["body"]["model"] == "m"
    assert seen["body"]["messages"][0]["role"] == "system"
    assert "Skerry Point" in seen["body"]["messages"][0]["content"]
    assert seen["body"]["messages"][1] == {"role": "user", "content": "Is the light lit?"}


async def test_http_error_propagates_for_evaluate_prompt_to_record() -> None:
    adapter = _adapter(lambda request: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.call_prompt(
            character_slug="docent", prompt_text="hi", model="m", lever_1_applied=False
        )
    await adapter.aclose()


def test_load_personas_validates(tmp_path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(PERSONAS))
    assert set(load_personas(good)) == {"keeper", "docent"}
    for bad in ([], {}, {"x": {"system_prompt": "  "}}, {"x": "not an object"}):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_personas(path)
