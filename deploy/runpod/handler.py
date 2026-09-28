"""Runpod Serverless handler for pagedserve: a proxy to the real server.

`main.py` starts `pagedserve serve` on 127.0.0.1:PAGEDSERVE_PORT (the same OpenAI-compatible
server the benchmarks ran against, engine-core process and all) and then the Runpod job
loop; every job is forwarded to it. This is the layout Runpod's own `worker-vllm` uses,
and it means the endpoint speaks OpenAI at `https://api.runpod.ai/v2/<id>/openai/v1/...`:
the platform wraps such a request as a job and this handler unwraps it.

Accepted job input shapes (all under job["input"]):

1. Runpod's OpenAI passthrough (what /openai/v1/... sends):
       {"openai_route": "/v1/chat/completions", "openai_input": {...}}
2. Any server route:
       {"route": "/v1/completions", "body": {...}, "method": "POST"}   (bodies POST, else GET)
3. Shorthand:
       {"prompt": "...", "sampling_params": {...}, "stream": false}
       {"messages": [...], "sampling_params": {...}, "stream": true}

With "stream": true the server's SSE bytes are yielded as they arrive (the platform
relays them to the caller); otherwise the parsed JSON response is yielded once.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

import httpx

DEFAULT_CHAT_ROUTE = "/v1/chat/completions"
DEFAULT_COMPLETION_ROUTE = "/v1/completions"
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT", "3600"))


def normalize_job_input(job_input: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None]:
    """`(route, method, body)` for any accepted job input shape."""
    if job_input.get("openai_input"):
        return job_input.get("openai_route") or DEFAULT_CHAT_ROUTE, "POST", job_input["openai_input"]
    if job_input.get("openai_route"):
        return job_input["openai_route"], "GET", None  # e.g. /v1/models
    if job_input.get("route"):
        body = job_input.get("body")
        method = (job_input.get("method") or ("POST" if body else "GET")).upper()
        return job_input["route"], method, body
    messages, prompt = job_input.get("messages"), job_input.get("prompt")
    if messages is None and prompt is None:
        raise ValueError("job input needs one of: openai_input (+openai_route), route (+body), "
                         "or prompt / messages")
    body = {**dict(job_input.get("sampling_params") or {}), "stream": bool(job_input.get("stream", False))}
    if messages is not None:
        return DEFAULT_CHAT_ROUTE, "POST", {**body, "messages": messages}
    return DEFAULT_COMPLETION_ROUTE, "POST", {**body, "prompt": prompt}


def _error(message: str, error_type: str = "worker_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": None}}


def make_handler(client: httpx.AsyncClient, served_model: str | None = None,
                 alive=lambda: True):
    """The job handler, closed over an HTTP client for the server (an ASGI-transport client
    in the tests, a real one in the worker). `served_model` fills in a missing "model"
    field; `alive()` reports whether the server process is still there."""

    async def handler(job: dict[str, Any]) -> AsyncIterator[Any]:
        try:
            route, method, body = normalize_job_input(job.get("input") or {})
        except ValueError as exc:
            yield _error(str(exc))
            return
        if not alive():
            yield _error("pagedserve server process is not running; worker is unhealthy")
            return
        if body is not None and served_model and "model" not in body:
            body = {**body, "model": served_model}
        wants_stream = isinstance(body, dict) and body.get("stream") is True
        try:
            async with client.stream(method, route, json=body if method != "GET" else None,
                                     timeout=REQUEST_TIMEOUT_S) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", errors="replace")
                    yield _error(f"pagedserve returned HTTP {resp.status_code}: {detail}")
                    return
                if wants_stream:
                    async for chunk in resp.aiter_text():
                        if chunk:
                            yield chunk
                else:
                    raw = await resp.aread()
                    yield httpx.Response(200, content=raw).json()
        except httpx.HTTPError as exc:
            yield _error(f"request to pagedserve failed: {type(exc).__name__}: {exc}")

    return handler
