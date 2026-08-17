"""Tests for the HTTP backend.

Only the parts that can be wrong *quietly* -- transport retries and connection limits fail loudly:

* **Thinking is a per-request override.** If the kwarg fails to reach the body, the reasoning arm
  runs with thinking *off* and reports a null that reads as "reasoning changes nothing".
* **Token usage is the arm's covariate.** `max_tokens` truncation returns `content: null` for every
  choice, which `summarise` counts as a parse failure; recorded usage is what makes that visible.

`asyncio.run` rather than pytest-asyncio: these are the only async tests in the suite.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from critxer.core import backend

EP = backend.Endpoint("qwen", "Qwen/Qwen3.6-35B-A3B", "http://127.0.0.1:9021")
MSGS = [{"role": "user", "content": "audit this"}]


def call(*, content: str = "{}", completion_tokens: int = 7, replies: int = 1,
         finish: str = "stop", **kwargs):
    """Run one `sample` against a mock transport; return (result, request body)."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": finish}] * replies,
            "usage": {"prompt_tokens": 100, "completion_tokens": completion_tokens},
        })

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await backend.sample(http, EP, MSGS, temperature=0.0, **kwargs)

    return asyncio.run(go()), bodies[0]


def test_thinking_is_off_unless_asked_for():
    """The main study depends on it being off; nothing may turn it on by default."""
    _, body = call(n=1, max_tokens=512)
    assert "chat_template_kwargs" not in body


def test_thinking_reaches_the_request_body_as_a_per_request_override():
    _, body = call(n=1, max_tokens=4096, thinking=True)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_usage_is_recorded_into_the_sink_when_one_is_given():
    sink: list[dict] = []
    call(n=1, max_tokens=4096, completion_tokens=897, usage_sink=sink)
    assert sink == [{"prompt_tokens": 100, "completion_tokens": 897, "n": 1,
                     "finish_reasons": ["stop"]}]


def test_the_sink_records_n_so_per_choice_tokens_can_be_recovered():
    """`completion_tokens` is the SUM over the n choices of one request, not the per-choice count.

    Dividing by n is the whole reason `n` is recorded. A smoke run reported a p95 of 5,363 tokens
    against a 4,096 cap and the number looked like a broken cap rather than two choices summed.
    """
    sink: list[dict] = []
    call(n=2, max_tokens=4096, completion_tokens=5363, replies=2, usage_sink=sink)
    assert sink[0]["n"] == 2
    assert sink[0]["completion_tokens"] / sink[0]["n"] == pytest.approx(2681.5)


def test_truncation_is_recorded_per_choice_not_inferred_from_token_arithmetic():
    """`finish_reason == "length"` is the only honest truncation signal, for the reason above."""
    sink: list[dict] = []
    call(n=2, max_tokens=4096, replies=2, finish="length", usage_sink=sink)
    assert sink[0]["finish_reasons"] == ["length", "length"]


def test_no_sink_means_usage_is_simply_not_collected():
    """Every other caller passes no sink and must keep working unchanged."""
    out, _ = call(n=1, max_tokens=512)
    assert out == ["{}"]


def test_truncated_choices_still_pad_to_n():
    """A short choice list divided by n is how a truncated arm becomes a plausible number."""
    out, _ = call(n=4, max_tokens=512, replies=2)
    assert out == ["{}", "{}", None, None]


def test_a_missing_usage_block_is_not_an_error():
    """vLLM sends usage on chat completions, but a sink must not make the call depend on it."""
    sink: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await backend.sample(http, EP, MSGS, n=1, temperature=0.0, max_tokens=512,
                                        usage_sink=sink)

    assert asyncio.run(go()) == ["{}"]
    assert sink == []


def test_a_non_200_still_raises_after_its_retries():
    """The retry path must not be masked by anything added for the reasoning arm."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await backend.sample(http, EP, MSGS, n=1, temperature=0.0, max_tokens=512,
                                        attempts=1, thinking=True)

    with pytest.raises(RuntimeError, match="500"):
        asyncio.run(go())
