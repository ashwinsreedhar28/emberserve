"""Async Poisson load generator for any OpenAI-compatible `/v1/completions` endpoint.

Works unchanged against pagedserve, vLLM, or a Runpod endpoint. Each trace request is
sent at its `arrival_s` offset (relative to the run start); the response is streamed
and TTFT is the time to the first content chunk.

    records = asyncio.run(run_http_benchmark("http://localhost:8000", "Qwen/Qwen2.5-0.5B-Instruct", trace))
    print(summarize(records).one_line())
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

import httpx

from pagedserve.bench.metrics import RequestRecord
from pagedserve.bench.trace import TraceRequest, render_prompt


def _parse_sse_line(line: str) -> dict | None:
    """Return the JSON payload of a `data:` line, `{"done": True}` for the sentinel,
    or None for anything else (comments, blank keep-alives, event: lines)."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload:
        return None
    if payload == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _chunk_text(obj: dict) -> str | None:
    """Text delta from a completions or chat-completions stream chunk; None if the chunk
    carries no choice (e.g. a usage-only trailer)."""
    choices = obj.get("choices") or []
    if not choices:
        return None
    c = choices[0]
    if "text" in c:
        return c["text"] or ""
    delta = c.get("delta") or {}
    return delta.get("content") or ""


async def _one_request(client: httpx.AsyncClient, model: str, req: TraceRequest,
                       prompt: str | list[int], stream: bool, ignore_eos: bool | None,
                       extra_body: dict[str, Any] | None, path: str = "/v1/completions") -> RequestRecord:
    body: dict[str, Any] = {
        "model": model, "prompt": prompt, "max_tokens": req.output_len,
        "temperature": 0.0, "stream": stream,
    }
    if ignore_eos is not None:  # a vLLM/pagedserve extension; None omits it (hosted APIs)
        body["ignore_eos"] = ignore_eos
    if stream:
        body["stream_options"] = {"include_usage": True}
    if extra_body:
        body.update(extra_body)
    t_send = time.perf_counter()
    first: float | None = None
    last: float | None = None
    n_chunks = 0
    usage_out: int | None = None
    usage_in: int | None = None
    unparsed = ""  # non-SSE text seen before any token (an empty stream's reason)
    try:
        if stream:
            async with client.stream("POST", path, json=body) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode(errors="replace")[:200]
                    raise RuntimeError(f"HTTP {resp.status_code}: {text}")
                async for line in resp.aiter_lines():
                    obj = _parse_sse_line(line)
                    if obj is None:
                        # Keep what a 200 that is not SSE said (a gateway's own error
                        # body, say) so an empty stream carries its reason.
                        if last is None and line.strip() and len(unparsed) < 300:
                            unparsed += line.strip()[:300 - len(unparsed)]
                        continue
                    if obj.get("done"):
                        break
                    now = time.perf_counter()
                    usage = obj.get("usage")
                    if usage:
                        usage_out = usage.get("completion_tokens", usage_out)
                        usage_in = usage.get("prompt_tokens", usage_in)
                    if _chunk_text(obj) is None:
                        continue
                    n_chunks += 1
                    if first is None:
                        first = now
                    last = now
        else:
            resp = await client.post(path, json=body)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            now = time.perf_counter()
            first = last = now
            usage = resp.json().get("usage") or {}
            usage_out = usage.get("completion_tokens")
            usage_in = usage.get("prompt_tokens")
    except Exception as e:  # noqa: BLE001 - any failure is a failed record
        return RequestRecord(req.request_id, t_send, first, None, req.prompt_len, n_chunks,
                             success=False, error=f"{type(e).__name__}: {e}")
    out_tokens = usage_out if usage_out is not None else n_chunks
    in_tokens = usage_in if usage_in is not None else req.prompt_len
    if last is None:
        return RequestRecord(req.request_id, t_send, first, None, in_tokens, out_tokens,
                             success=False,
                             error="empty stream" + (f": {unparsed!r}" if unparsed else ""))
    return RequestRecord(req.request_id, t_send, first, last, in_tokens, out_tokens)


async def run_http_benchmark(base_url: str, model: str, trace: list[TraceRequest],
                             stream: bool = True, max_concurrency: int | None = None,
                             timeout_s: float = 600.0, api_key: str = "x",
                             transport: httpx.AsyncBaseTransport | None = None,
                             tokenizer: Any = None, ignore_eos: bool | None = True,
                             extra_body: dict[str, Any] | None = None,
                             progress: bool = True, path: str = "/v1/completions") -> list[RequestRecord]:
    """Replay `trace` against `base_url` honoring each request's `arrival_s` offset.

    `max_concurrency=None` means unbounded (the endpoint's own admission control is
    what's under test). `transport` lets tests inject `httpx.ASGITransport(app=...)`.
    `tokenizer` (optional) renders text prompts; without it token ids are sent.
    `ignore_eos=None` leaves the field out of the request (hosted OpenAI-compatible APIs
    do not know it); `path` is the completions route relative to `base_url`.
    """
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    prompts = [render_prompt(r, tokenizer) for r in trace]
    records: list[RequestRecord | None] = [None] * len(trace)
    done = 0
    t_start = time.perf_counter()

    async def worker(i: int, client: httpx.AsyncClient) -> None:
        nonlocal done
        req = trace[i]
        delay = req.arrival_s - (time.perf_counter() - t_start)
        if delay > 0:
            await asyncio.sleep(delay)
        if sem is not None:
            async with sem:
                rec = await _one_request(client, model, req, prompts[i], stream, ignore_eos,
                                         extra_body, path)
        else:
            rec = await _one_request(client, model, req, prompts[i], stream, ignore_eos,
                                     extra_body, path)
        records[i] = rec
        done += 1
        if progress and (done % 10 == 0 or done == len(trace)):
            elapsed = time.perf_counter() - t_start
            print(f"\r[load] {done}/{len(trace)} done  {elapsed:6.1f}s", end="",
                  file=sys.stderr, flush=True)

    limits = httpx.Limits(max_connections=None, max_keepalive_connections=256)
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport,
                                 limits=limits, headers=headers) as client:
        await asyncio.gather(*(worker(i, client) for i in range(len(trace))))
    if progress:
        print(file=sys.stderr)
    return [r for r in records if r is not None]


# vLLM's Prometheus names -> the keys pagedserve's JSON /metrics uses, so a sweep can
# record server-side latency sums for either engine.
_PROM_KEYS = {
    "vllm:time_to_first_token_seconds_sum": "ttft_s_sum",
    "vllm:time_to_first_token_seconds_count": "ttft_count",
    "vllm:time_per_output_token_seconds_sum": "tpot_s_sum",
    "vllm:time_per_output_token_seconds_count": "tpot_count",
    "vllm:inter_token_latency_seconds_sum": "tpot_s_sum",
    "vllm:inter_token_latency_seconds_count": "tpot_count",
    "vllm:e2e_request_latency_seconds_sum": "e2e_s_sum",
    "vllm:e2e_request_latency_seconds_count": "e2e_count",
    "vllm:request_success_total": "requests_finished_total",
    "vllm:generation_tokens_total": "generated_tokens_total",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
}


def parse_prometheus(text: str) -> dict[str, float]:
    """The metrics of `_PROM_KEYS` out of a Prometheus text exposition (labels summed)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition("{") if "{" in line.split(" ")[0] else (line.split(" ")[0], "", "")
        if rest:
            value = rest.rpartition("}")[2].strip().split(" ")[0]
        else:
            parts = line.split()
            name, value = parts[0], parts[1] if len(parts) > 1 else "0"
        key = _PROM_KEYS.get(name)
        if key is None:
            continue
        try:
            out[key] = out.get(key, 0.0) + float(value)
        except ValueError:
            continue
    return out


def _bench_worker(conn, base_url: str, model: str, shard: list[TraceRequest], t0: float,
                  kwargs: dict[str, Any]) -> None:
    """One load-generator process: waits for the shared start instant, replays its shard."""
    delay = t0 - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
    try:
        recs = asyncio.run(run_http_benchmark(base_url, model, shard, progress=False, **kwargs))
        conn.send(("ok", recs))
    except Exception as e:  # noqa: BLE001 - reported to the parent
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def run_http_benchmark_procs(base_url: str, model: str, trace: list[TraceRequest], procs: int,
                             **kwargs: Any) -> list[RequestRecord]:
    """`run_http_benchmark` split across `procs` processes, each with its own event loop.

    One Python process parsing SSE lines tops out around 20k events/s, which a 200-stream
    burst on a fast server reaches: the client then queues, and TTFT/TPOT measure the
    client. The trace is dealt round-robin (arrival offsets kept) and every process starts
    at the same instant; `perf_counter` is the system monotonic clock, so the records merge.
    `tokenizer` is not sent to the workers: real-text traces carry their text and synthetic
    ones send ids."""
    import multiprocessing as mp

    if procs <= 1:
        kwargs.pop("progress", None)
        return asyncio.run(run_http_benchmark(base_url, model, trace, **kwargs))
    kwargs = {k: v for k, v in kwargs.items() if k not in ("tokenizer", "progress", "transport")}
    shards = [trace[i::procs] for i in range(procs)]
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter() + 2.0  # time for the workers to import and connect
    conns, workers = [], []
    for shard in shards:
        parent, child = ctx.Pipe(duplex=False)
        w = ctx.Process(target=_bench_worker, args=(child, base_url, model, shard, t0, kwargs))
        w.start()
        child.close()
        conns.append(parent)
        workers.append(w)
    records: list[RequestRecord] = []
    errors: list[str] = []
    for conn in conns:
        kind, payload = conn.recv()
        if kind == "ok":
            records.extend(payload)
        else:
            errors.append(payload)
    for w in workers:
        w.join()
    if errors:
        raise RuntimeError("load-generator worker failed: " + "; ".join(errors))
    order = {r.request_id: i for i, r in enumerate(trace)}
    records.sort(key=lambda r: order.get(r.request_id, 0))
    return records


def _auth(api_key: str | None) -> dict[str, str]:
    """Bearer header for endpoints behind a gateway (Runpod's load balancer rejects
    unauthenticated /health and /metrics polls with 401)."""
    return {"Authorization": f"Bearer {api_key}"} if api_key and api_key != "x" else {}


async def fetch_metrics(base_url: str, path: str = "/metrics",
                        transport: httpx.AsyncBaseTransport | None = None,
                        api_key: str | None = None) -> dict | None:
    """The server's metrics as a flat dict: pagedserve's `/metrics` JSON as is, vLLM's
    Prometheus text reduced to the keys of `_PROM_KEYS`; None when there is no such route
    (hosted APIs) or nothing recognizable in it."""
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0, transport=transport,
                                     headers=_auth(api_key)) as c:
            r = await c.get(path)
            if r.status_code != 200:
                return None
            try:
                data = r.json()
                return data if isinstance(data, dict) else None
            except ValueError:
                parsed = parse_prometheus(r.text)
                return parsed or None
    except httpx.HTTPError:
        return None


async def wait_for_health(base_url: str, timeout_s: float = 600.0, path: str = "/health",
                          transport: httpx.AsyncBaseTransport | None = None,
                          api_key: str | None = None) -> bool:
    """Poll `base_url + path` until it answers 200 or the timeout elapses."""
    deadline = time.perf_counter() + timeout_s
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0, transport=transport,
                                 headers=_auth(api_key)) as c:
        while time.perf_counter() < deadline:
            try:
                r = await c.get(path)
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    return False
