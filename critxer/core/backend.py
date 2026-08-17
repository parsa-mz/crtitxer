"""HTTP backend for auditor models.

One interface for both providers: a local vLLM server now, a hosted API later. Project code
never talks to a provider SDK, so switching backends does not mean a second harness.

``trust_env=False`` is deliberate. The sandbox exports HTTP_PROXY with an empty NO_PROXY, so
requests to a *local* vLLM port would otherwise be routed through the egress proxy and fail.
The same pattern is used elsewhere in this project for presigned S3 downloads.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Endpoint:
    """A served model. ``label`` is the short name used in result files."""

    label: str
    model: str
    base_url: str
    revision: str = ""


def parse_endpoint(spec: str) -> Endpoint:
    """``label=model@url``, the form every runner takes on the command line.

    Kept here rather than in each command so a served model is described one way. Hardcoding the
    auditor pair instead is how the auditors got out of sync with the spec before, and gate 0 is a
    model-*selection* criterion rather than a go/no-go, so candidates change.
    """
    label, _, rest = spec.partition("=")
    model, _, url = rest.partition("@")
    if not (label and model and url):
        raise ValueError(f"expected label=model@url, got {spec!r}")
    return Endpoint(label, model, url)


def client(timeout: float = 600.0) -> httpx.AsyncClient:
    """An AsyncClient configured for local vLLM endpoints."""
    return httpx.AsyncClient(
        trust_env=False,
        timeout=timeout,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
    )


async def sample(
    http: httpx.AsyncClient,
    endpoint: Endpoint,
    messages: list[dict],
    n: int,
    temperature: float,
    max_tokens: int,
    schema: dict | None = None,
    seed: int | None = None,
    attempts: int = 3,
    thinking: bool = False,
    usage_sink: list[dict] | None = None,
) -> list[str | None]:
    """Draw ``n`` completions for one prompt. Returns raw strings, None per failed choice.

    ``n`` is served in one request so an item's samples share a scheduling batch, which is what
    makes the T=0 determinism probe meaningful under a different batch composition.

    ``thinking`` re-enables reasoning per request, overriding the server-wide default. A trace that
    hits ``max_tokens`` returns ``content: None`` for every choice and reads downstream as a parse
    failure, so budget accordingly.

    ``usage_sink`` collects per-request token counts, the only way truncation is visible rather than
    inferred. ``n`` and ``finish_reasons`` go alongside because the raw usage misleads:
    ``completion_tokens`` is the **sum** over choices, and ``length`` is the only honest truncation
    signal.
    """
    body: dict = {
        "model": endpoint.model,
        "messages": messages,
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    if seed is not None:
        body["seed"] = seed
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "audit", "schema": schema, "strict": True},
        }

    last = ""
    for attempt in range(attempts):
        try:
            resp = await http.post(f"{endpoint.base_url}/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:  # transient transport failure
            last = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                payload = resp.json()
                choices = payload.get("choices", [])
                if usage_sink is not None and (usage := payload.get("usage")):
                    usage_sink.append({
                        **usage, "n": n,
                        "finish_reasons": [c.get("finish_reason") for c in choices],
                    })
                out = [c.get("message", {}).get("content") for c in choices]
                # Pad rather than silently returning a short list: callers divide by n.
                return out + [None] * (n - len(out))
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
        await asyncio.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{endpoint.label}: {attempts} attempts failed; last = {last}")


async def map_prompts(
    http: httpx.AsyncClient,
    endpoint: Endpoint,
    prompts: list[list[dict]],
    concurrency: int,
    **kwargs,
) -> list[list[str | None]]:
    """Run many prompts against one endpoint with bounded concurrency, order preserved."""
    gate = asyncio.Semaphore(concurrency)

    async def one(messages: list[dict]) -> list[str | None]:
        async with gate:
            return await sample(http, endpoint, messages, **kwargs)

    return await asyncio.gather(*(one(m) for m in prompts))
