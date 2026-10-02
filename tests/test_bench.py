"""Benchmark harness tests: traces, metrics, offline driver, ablation CLI, HTTP load
generator (against an in-process ASGI app), and plots. No GPU, no weights, no network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

from emberserve.bench import ablation, plot
from emberserve.bench.load import run_http_benchmark, wait_for_health
from emberserve.bench.metrics import RequestRecord, Stat, summarize
from emberserve.bench.offline import run_offline_benchmark
from emberserve.bench.trace import (MIN_LEN, LogNormal, TraceRequest, from_json, generate_trace,
                                    render_prompt, to_json)
from emberserve.config import ModelConfig
from tests.test_engine import make_engine

torch.set_num_threads(2)
VOCAB = ModelConfig.tiny().vocab_size


def tiny_trace(n: int, seed: int = 0, rate: float | None = None, max_out: int = 16,
               **kw) -> list[TraceRequest]:
    return generate_trace(n, seed=seed, request_rate=rate, vocab_size=VOCAB,
                          prompt_len_dist=LogNormal(12, 0.4), output_len_dist=LogNormal(8, 0.4),
                          max_prompt_len=24, max_output_len=max_out, **kw)


# ---- trace ---------------------------------------------------------------------------
def test_trace_deterministic_and_roundtrip() -> None:
    a = generate_trace(50, seed=3, request_rate=2.0)
    b = generate_trace(50, seed=3, request_rate=2.0)
    assert a == b
    assert generate_trace(50, seed=4, request_rate=2.0) != a
    assert from_json(to_json(a)) == a
    assert all(len(r.prompt_ids) == r.prompt_len for r in a)
    assert a[0].arrival_s == 0.0
    assert all(x.arrival_s <= y.arrival_s for x, y in zip(a, a[1:], strict=False))


def test_trace_clipping_and_all_at_zero() -> None:
    t = generate_trace(300, seed=0, prompt_len_dist=LogNormal(1, 3.0),
                       output_len_dist=LogNormal(5000, 0.1), max_prompt_len=100,
                       max_output_len=64)
    assert min(r.prompt_len for r in t) >= MIN_LEN
    assert max(r.prompt_len for r in t) <= 100
    assert all(r.output_len == 64 for r in t)
    assert all(r.arrival_s == 0.0 for r in t)


def test_trace_poisson_interarrival_mean() -> None:
    rate = 5.0
    t = generate_trace(2001, seed=11, request_rate=rate)
    gaps = np.diff([r.arrival_s for r in t])
    assert len(gaps) == 2000
    assert abs(gaps.mean() - 1 / rate) / (1 / rate) < 0.15


def test_trace_shared_prefix_and_render() -> None:
    t = generate_trace(5, seed=0, shared_prefix_len=8, vocab_size=VOCAB, max_prompt_len=32)
    first = t[0].prompt_ids[:8]
    assert all(r.prompt_ids[:8] == first for r in t)
    assert render_prompt(t[0]) == t[0].prompt_ids

    class WordTok:
        def encode(self, s: str) -> list[int]:
            return [hash(w) % 1000 for w in s.split()]

        def decode(self, ids: list[int]) -> str:
            return " ".join("w" for _ in ids)

    txt = render_prompt(t[1], WordTok())
    assert isinstance(txt, str)
    assert len(WordTok().encode(txt)) == t[1].prompt_len
    txt0 = render_prompt(t[0], WordTok())
    assert txt.split()[:8] == txt0.split()[:8]


def test_rendered_text_prompts_follow_the_seed() -> None:
    """Synthetic text bodies used to depend only on the request index modulo 40, so 100
    equal-length requests rendered 40 distinct prompts, identical for every seed: a
    prefix-caching server served repeats. They now come from the seeded token ids."""
    class WordTok:
        def encode(self, s: str) -> list[int]:
            return [hash(w) % 1000 for w in s.split()]

        def decode(self, ids: list[int]) -> str:
            return " ".join("w" for _ in ids)

    def render(seed: int) -> list[str]:
        t = generate_trace(100, seed=seed, vocab_size=VOCAB, max_prompt_len=64)
        for r in t:  # equal lengths, as in the reviewer's repro
            r.prompt_len, r.prompt_ids = 48, (r.prompt_ids * 48)[:48]
        return [render_prompt(r, WordTok()) for r in t]

    a, b = render(0), render(1000)
    assert len(set(a)) == 100 and len(set(b)) == 100
    assert not set(a) & set(b)


# ---- metrics -------------------------------------------------------------------------
def test_summarize_percentiles_and_goodput() -> None:
    recs = []
    ttfts = [0.010, 0.020, 0.030, 0.040, 0.100]
    for i, tt in enumerate(ttfts):
        # 11 output tokens, 5 ms per token after the first -> e2e = ttft + 50 ms
        recs.append(RequestRecord(f"r{i}", arrival_s=float(i), first_token_s=i + tt,
                                  finish_s=i + tt + 0.050, prompt_tokens=10, output_tokens=11))
    recs.append(RequestRecord("bad", 0.0, None, None, 10, 0, success=False, error="x"))
    s = summarize(recs, slo_ttft_ms=35.0, slo_tpot_ms=6.0, wall_s=10.0)
    assert s.num_requests == 6 and s.completed == 5 and s.failed == 1
    assert s.ttft_ms.p50 == pytest.approx(30.0)
    assert s.ttft_ms.p90 == pytest.approx(float(np.percentile([10, 20, 30, 40, 100], 90)))
    assert s.ttft_ms.p99 == pytest.approx(float(np.percentile([10, 20, 30, 40, 100], 99)))
    assert s.ttft_ms.mean == pytest.approx(40.0)
    assert s.tpot_ms.p50 == pytest.approx(5.0)
    assert s.e2e_ms.p50 == pytest.approx(80.0)
    assert s.throughput_tok_s == pytest.approx(55 / 10.0)
    assert s.total_throughput_tok_s == pytest.approx(105 / 10.0)
    assert s.requests_per_s == pytest.approx(0.5)
    assert s.goodput_rps == pytest.approx(3 / 10.0)  # ttft <= 35 ms: 10, 20, 30
    d = s.to_dict()
    assert d["ttft_ms"]["p99"] == s.ttft_ms.p99
    json.dumps(d)
    assert np.isnan(Stat.of([]).p50)
    assert summarize([]).completed == 0


def test_summarize_default_wall() -> None:
    recs = [RequestRecord("a", 1.0, 1.1, 2.0, 5, 4), RequestRecord("b", 1.5, 1.6, 3.0, 5, 6)]
    s = summarize(recs)
    assert s.duration_s == pytest.approx(2.0)
    assert s.throughput_tok_s == pytest.approx(10 / 2.0)


# ---- offline driver ------------------------------------------------------------------
def test_offline_benchmark_records() -> None:
    eng = make_engine("paged_torch")
    trace = tiny_trace(12)
    recs, steps = run_offline_benchmark(eng, trace)
    assert len(recs) == 12 and len(steps) > 0
    for rec, req in zip(recs, trace, strict=True):
        assert rec.success and rec.error is None
        assert rec.output_tokens == req.output_len
        assert rec.prompt_tokens == req.prompt_len
        assert rec.first_token_s is not None and rec.finish_s is not None
        assert rec.ttft_s <= rec.e2e_s
        assert rec.ttft_s > 0
    assert all(s.kv_utilization >= 0 for s in steps)
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks


def test_offline_benchmark_honors_arrivals() -> None:
    eng = make_engine("paged_torch")
    trace = tiny_trace(4, rate=40.0)  # ~25 ms apart
    recs, _ = run_offline_benchmark(eng, trace)
    t0 = recs[0].arrival_s
    for rec, req in zip(recs, trace, strict=True):
        assert rec.arrival_s - t0 >= req.arrival_s - 1e-3


def test_static_vs_continuous_steps() -> None:
    trace = generate_trace(24, seed=5, vocab_size=VOCAB, prompt_len_dist=LogNormal(10, 0.3),
                           output_len_dist=LogNormal(20, 0.5), max_prompt_len=16,
                           max_output_len=32)
    lens = {r.output_len for r in trace}
    assert len(lens) > 1, "need varied output lengths for static batching to lose"
    cont = make_engine("paged_torch", max_num_seqs=8)
    stat = make_engine("paged_torch", max_num_seqs=8)
    recs_c, steps_c = run_offline_benchmark(cont, trace)
    recs_s, steps_s = run_offline_benchmark(stat, trace, static_batching=True)
    assert all(r.success for r in recs_c) and all(r.success for r in recs_s)
    assert [r.output_tokens for r in recs_c] == [r.output_tokens for r in recs_s]
    assert len(steps_c) < len(steps_s)
    # Static batching never had more than one batch in flight.
    assert max(s.num_seqs for s in steps_s) <= 8


def test_offline_latency_counts_from_the_offer() -> None:
    """Arrival used to be the admission time, so requests waiting for an earlier static
    batch had their wait left out of TTFT (a reviewer's repro: three requests offered at
    t=0 got three different arrival times)."""
    trace = tiny_trace(3)  # all offered at t=0
    stat = make_engine("paged_torch", max_num_seqs=1)
    recs, _ = run_offline_benchmark(stat, trace, static_batching=True)
    assert len({r.arrival_s for r in recs}) == 1
    ttft = [r.ttft_s for r in recs]
    assert ttft[0] < ttft[1] < ttft[2]  # the queued ones waited, and it shows


# ---- ablation CLI --------------------------------------------------------------------
def test_ablation_tiny(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "ablation.json"
    rc = ablation.main(["--tiny", "--configs", "naive,paged_torch,static", "--trace-n", "8",
                        "--out", str(out), "--block-size", "4", "--no-warmup"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert [r["config"] for r in data["results"]] == ["naive", "paged_torch", "static"]
    for r in data["results"]:
        assert r["summary"]["completed"] == 8
        assert r["summary"]["throughput_tok_s"] > 0
        assert 0.0 <= r["kv_utilization_mean"] <= 1.0
    assert data["results"][2]["static_batching"] is True
    table = capsys.readouterr().out
    assert "| config |" in table and "| static |" in table
    assert ablation.AblationConfig.parse("paged_flash+graphs").enable_cuda_graphs
    assert ablation.AblationConfig.parse("paged_torch+prefix").enable_prefix_caching
    with pytest.raises(ValueError):
        ablation.AblationConfig.parse("bogus")


def test_ablation_chunked_config(tmp_path: Path) -> None:
    cfg = ablation.AblationConfig.parse("paged_flash+graphs+chunked")
    assert cfg.enable_cuda_graphs and cfg.enable_chunked_prefill and not cfg.enable_prefix_caching
    args = ablation.build_parser().parse_args(["--tiny"])
    assert cfg.max_num_batched_tokens(args) == ablation.CHUNKED_DEFAULT_BUDGET
    assert ablation.AblationConfig.parse("paged_torch").max_num_batched_tokens(args) == \
        ablation.DEFAULT_BUDGET
    args = ablation.build_parser().parse_args(["--tiny", "--max-num-batched-tokens", "24"])
    assert cfg.max_num_batched_tokens(args) == 24

    out = tmp_path / "abl.json"
    rc = ablation.main(["--tiny", "--configs", "paged_torch,paged_torch+chunked", "--trace-n",
                        "8", "--out", str(out), "--block-size", "4", "--no-warmup",
                        "--max-num-batched-tokens", "64"])
    assert rc == 0
    data = json.loads(out.read_text())
    plain, chunked = data["results"]
    assert chunked["enable_chunked_prefill"] and chunked["max_num_batched_tokens"] == 64
    assert chunked["max_step_tokens"] <= 64
    assert chunked["summary"]["completed"] == plain["summary"]["completed"] == 8


# ---- HTTP load generator -------------------------------------------------------------
def _fake_app(n_chunks: int = 5, delay_s: float = 0.01) -> Starlette:
    async def completions(request: Request):
        body = await request.json()
        assert body["stream"] is True
        assert body["max_tokens"] >= 1

        async def gen():
            await asyncio.sleep(delay_s)
            for i in range(n_chunks):
                chunk = {"id": "cmpl-1", "object": "text_completion",
                         "choices": [{"index": 0, "text": f"t{i} ", "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                await asyncio.sleep(delay_s)
            trailer = {"id": "cmpl-1", "choices": [],
                       "usage": {"prompt_tokens": len(body["prompt"]),
                                 "completion_tokens": n_chunks}}
            yield f"data: {json.dumps(trailer)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def health(_: Request):
        return JSONResponse({"status": "ok"})

    async def metrics(_: Request):
        return JSONResponse({"requests_finished_total": 3, "spec_drafted_total": 40,
                             "spec_accepted_total": 25, "num_running": 0})

    return Starlette(routes=[Route("/v1/completions", completions, methods=["POST"]),
                             Route("/health", health), Route("/metrics", metrics)])


def test_http_load_generator_asgi() -> None:
    app = _fake_app(n_chunks=5, delay_s=0.01)
    trace = tiny_trace(6, rate=50.0)
    transport = httpx.ASGITransport(app=app)
    assert asyncio.run(wait_for_health("http://test", timeout_s=5, transport=transport))
    recs = asyncio.run(run_http_benchmark("http://test", "m", trace, transport=transport,
                                          max_concurrency=3, progress=False))
    assert len(recs) == 6
    for rec, req in zip(recs, trace, strict=True):
        assert rec.success, rec.error
        assert rec.output_tokens == 5
        assert rec.prompt_tokens == req.prompt_len
        assert rec.ttft_s is not None and rec.e2e_s is not None
        assert 0 < rec.ttft_s < rec.e2e_s
    s = summarize(recs)
    assert s.completed == 6 and s.throughput_tok_s > 0
    assert s.tpot_ms.p50 > 0


def test_fetch_metrics_json_or_none() -> None:
    from emberserve.bench.load import fetch_metrics

    transport = httpx.ASGITransport(app=_fake_app())
    m = asyncio.run(fetch_metrics("http://test", transport=transport))
    assert m["spec_drafted_total"] == 40 and m["spec_accepted_total"] == 25
    assert asyncio.run(fetch_metrics("http://test", path="/nope", transport=transport)) is None


def test_prometheus_metrics_are_reduced_to_the_json_keys() -> None:
    from emberserve.bench.load import fetch_metrics, parse_prometheus

    text = """# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="m"} 0.0
vllm:time_to_first_token_seconds_sum{model_name="m"} 12.5
vllm:time_to_first_token_seconds_count{model_name="m"} 200.0
vllm:inter_token_latency_seconds_sum{model_name="m"} 3.0
vllm:inter_token_latency_seconds_count{model_name="m"} 1000.0
vllm:e2e_request_latency_seconds_sum{model_name="m"} 400.0
vllm:e2e_request_latency_seconds_count{model_name="m"} 200.0
vllm:request_success_total{finished_reason="length",model_name="m"} 150.0
vllm:request_success_total{finished_reason="stop",model_name="m"} 50.0
vllm:num_requests_running{model_name="m"} 0.0
"""
    got = parse_prometheus(text)
    assert got == {"ttft_s_sum": 12.5, "ttft_count": 200.0, "tpot_s_sum": 3.0, "tpot_count": 1000.0,
                   "e2e_s_sum": 400.0, "e2e_count": 200.0, "requests_finished_total": 200.0}

    async def metrics(_: Request):
        return PlainTextResponse(text)

    app = Starlette(routes=[Route("/metrics", metrics)])
    m = asyncio.run(fetch_metrics("http://test", transport=httpx.ASGITransport(app=app)))
    assert m["ttft_count"] == 200.0 and m["requests_finished_total"] == 200.0


def test_http_load_generator_records_errors() -> None:
    async def boom(_: Request):
        return JSONResponse({"error": "no"}, status_code=500)

    app = Starlette(routes=[Route("/v1/completions", boom, methods=["POST"])])
    recs = asyncio.run(run_http_benchmark("http://test", "m", tiny_trace(3),
                                          transport=httpx.ASGITransport(app=app),
                                          progress=False))
    assert len(recs) == 3 and all(not r.success for r in recs)
    assert "500" in recs[0].error
    assert summarize(recs).failed == 3


# ---- plots ---------------------------------------------------------------------------
def _fake_summary(scale: float) -> dict:
    return summarize([RequestRecord(f"r{i}", i * 0.1, i * 0.1 + 0.02 * scale,
                                    i * 0.1 + 0.5 * scale, 50, 20) for i in range(5)]).to_dict()


def test_plot_functions_write_pngs(tmp_path: Path) -> None:
    rates = [1.0, 4.0, 16.0, None]
    sweeps = {
        name: {"kind": "sweep", "system": name,
               "runs": [{"request_rate": r, "summary": _fake_summary(k * (i + 1))}
                        for i, r in enumerate(rates)]}
        for k, name in ((1.0, "vllm"), (1.4, "emberserve"))
    }
    abl = {"kind": "ablation", "model": "tiny", "gpu": None, "device": "cpu",
           "trace": {"n": 8, "request_rate": None},
           "results": [{"config": c, "summary": _fake_summary(1 + i),
                        "kv_utilization_mean": 0.1 * (i + 1)}
                       for i, c in enumerate(["naive", "static", "paged_torch"])]}
    written = plot.plot_all(sweeps, abl, tmp_path)
    assert len(written) == 4
    for p in written:
        assert p.exists() and p.stat().st_size > 1000
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # CLI path on the same data.
    (tmp_path / "vllm.json").write_text(json.dumps(sweeps["vllm"]))
    (tmp_path / "ablation.json").write_text(json.dumps(abl))
    assert plot.main([str(tmp_path / "vllm.json"), "--ablation", str(tmp_path / "ablation.json"),
                      "--out-dir", str(tmp_path / "p2")]) == 0
    assert (tmp_path / "p2" / "ablation.png").exists()


def test_http_load_generator_real_stream() -> None:
    """ASGITransport buffers the body; a real socket proves TTFT is measured at the
    first chunk, not at the end of the response."""
    import socket
    import threading

    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(_fake_app(n_chunks=8, delay_s=0.03), host="127.0.0.1",
                                           port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert asyncio.run(wait_for_health(base, timeout_s=10))
        recs = asyncio.run(run_http_benchmark(base, "m", tiny_trace(3), progress=False))
        for rec in recs:
            assert rec.success, rec.error
            assert rec.output_tokens == 8
            assert rec.ttft_s < 0.5 * rec.e2e_s
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_multiprocess_client_merges_records_in_trace_order() -> None:
    import socket
    import threading

    import uvicorn

    from emberserve.bench.load import run_http_benchmark_procs

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(_fake_app(n_chunks=4, delay_s=0.01), host="127.0.0.1",
                                           port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert asyncio.run(wait_for_health(base, timeout_s=10))
        trace = tiny_trace(9, rate=100.0)
        recs = run_http_benchmark_procs(base, "m", trace, 3, timeout_s=30, tokenizer=object())
        assert [r.request_id for r in recs] == [r.request_id for r in trace]
        assert all(r.success and r.output_tokens == 4 for r in recs), [r.error for r in recs]
        s = summarize(recs)
        assert s.completed == 9 and s.tpot_ms.p50 > 0
        # the shards started together: no request waited for a whole other shard first
        spread = max(r.arrival_s for r in recs) - min(r.arrival_s for r in recs)
        assert spread < 1.5, spread
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_hosted_request_omits_ignore_eos_and_uses_path():
    """`ignore_eos=None` leaves the vLLM extension out of the body and `path` picks the
    route: what `run_vllm_baseline --hosted` sends to an OpenAI-compatible API."""
    import json as _json

    import httpx

    from emberserve.bench.load import run_http_benchmark
    from emberserve.bench.trace import generate_trace

    seen: list[tuple[str, dict]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, _json.loads(request.content)))
        lines = ['data: {"choices":[{"text":"hi"}]}', 'data: {"choices":[{"text":"!"}],"usage":'
                 '{"prompt_tokens":3,"completion_tokens":2}}', "data: [DONE]"]
        return httpx.Response(200, content="\n\n".join(lines).encode(),
                              headers={"content-type": "text/event-stream"})

    trace = generate_trace(2, seed=0, request_rate=None, max_prompt_len=8, max_output_len=4,
                           vocab_size=100)
    records = asyncio.run(run_http_benchmark(
        "https://api.example", "m", trace, transport=httpx.MockTransport(handler),
        ignore_eos=None, path="/completions", progress=False))
    assert all(r.success for r in records) and len(seen) == 2
    for path, body in seen:
        assert path == "/completions"
        assert "ignore_eos" not in body and body["stream"] is True


def test_sharegpt_trace_samples_and_filters(tmp_path):
    from emberserve.bench.trace import render_prompt, sharegpt_trace

    class Tok:  # one token per character
        def encode(self, s):
            return [ord(c) for c in s]

    convs = []
    for i in range(30):
        prompt = "p" * (5 + i * 3)          # 5..92 chars
        reply = "r" * (4 + (i * 7) % 40)     # 4..43
        convs.append({"conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": reply}]})
    convs.append({"conversations": [{"from": "gpt", "value": "starts with the model"}]})
    convs.append({"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "x"}]})  # too short
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(convs))
    tr = sharegpt_trace(str(path), 10, Tok(), seed=1, request_rate=2.0, max_prompt_len=50)
    assert len(tr) == 10
    assert all(4 <= r.prompt_len <= 50 and r.output_len >= 4 for r in tr)
    assert all(r.prompt_len == len(r.prompt_text) and r.output_len == (4 + (int((r.prompt_len - 5) / 3) * 7) % 40)
               for r in tr)
    assert render_prompt(tr[0]) == tr[0].prompt_text
    assert tr[0].arrival_s == 0.0 and tr[-1].arrival_s > 0
    # same seed -> same sample, different rate -> same requests, different arrivals
    tr2 = sharegpt_trace(str(path), 10, Tok(), seed=1, request_rate=None, max_prompt_len=50)
    assert [r.prompt_text for r in tr2] == [r.prompt_text for r in tr]
    assert all(r.arrival_s == 0.0 for r in tr2)
    with pytest.raises(ValueError):
        sharegpt_trace(str(path), 100, Tok(), seed=1)


# ---- SSE accounting (review fixes) -------------------------------------------------------
def _sse_transport(events: list[tuple[float, str]]) -> httpx.MockTransport:
    """A server that streams `events` ((delay before, data payload) pairs) as SSE."""
    async def body():
        for delay, data in events:
            await asyncio.sleep(delay)
            yield f"data: {data}\n\n".encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def _one(events: list[tuple[float, str]], path: str = "/v1/chat/completions"):
    from emberserve.bench.trace import TraceRequest

    trace = [TraceRequest("r0", prompt_len=3, output_len=2, arrival_s=0.0, prompt_text="hi")]
    return asyncio.run(run_http_benchmark("http://test", "m", trace, transport=_sse_transport(events),
                                          progress=False, path=path))[0]


def test_role_only_chunk_is_not_the_first_token() -> None:
    role = '{"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}'
    tok = '{"choices":[{"delta":{"content":"%s"},"finish_reason":null}]}'
    fin = '{"choices":[{"delta":{},"finish_reason":"length"}]}'
    r = _one([(0.0, role), (0.15, tok % "hello"), (0.01, tok % " there"), (0.0, fin), (0.0, "[DONE]")])
    assert r.success and r.ttft_s is not None and r.ttft_s >= 0.14  # the content, not the role
    assert r.output_tokens == 2  # no usage: two content chunks, not role + 2 + finish


def test_error_event_and_cut_off_stream_are_failures() -> None:
    tok = '{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}'
    err = '{"error":{"message":"engine died","type":"server_error"}}'
    r = _one([(0.0, tok), (0.01, err)])
    assert not r.success and "engine died" in r.error and r.end_s is not None
    r = _one([(0.0, tok)])  # no finish reason, no [DONE]
    assert not r.success and "before completion" in r.error


def test_failures_count_in_the_run_duration() -> None:
    """A failure that ends after the last success used to fall out of the duration
    (results/openrouter_kimi_k3_r05.json: 46.04 s of 61.37 s wall, 141 vs ~106 tok/s)."""
    recs = [RequestRecord("a", arrival_s=0.0, first_token_s=0.5, finish_s=10.0,
                          prompt_tokens=5, output_tokens=100),
            RequestRecord("b", arrival_s=1.0, first_token_s=None, finish_s=None,
                          prompt_tokens=5, output_tokens=0, success=False, error="timeout",
                          end_s=20.0)]
    s = summarize(recs)
    assert s.duration_s == 20.0 and s.throughput_tok_s == 5.0 and s.completed == 1


def test_usage_trailer_does_not_move_the_last_token() -> None:
    """vLLM's completions stream ends with a separate usage-only chunk; emberserve puts usage
    on its last token. The trailer must not become the "last token" (a reviewer's repro:
    two tokens 11 ms apart and a trailer 150 ms later read as 162 ms TPOT)."""
    tok = '{"choices":[{"text":"%s","finish_reason":%s}]}'
    usage = '{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}'
    r = _one([(0.0, tok % ("a", "null")), (0.011, tok % ("b", '"length"')), (0.15, usage),
              (0.0, "[DONE]")], path="/v1/completions")
    assert r.success and r.output_tokens == 2
    assert r.tpot_s is not None and r.tpot_s < 0.05


def test_final_token_with_empty_text_is_a_token() -> None:
    """emberserve sends its last token with the finish reason, and its text can be empty
    (suppressed by the detokenizer): a one-token completion is not an empty stream."""
    fin = '{"choices":[{"text":"","finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}'
    r = _one([(0.02, fin), (0.0, "[DONE]")], path="/v1/completions")
    assert r.success and r.output_tokens == 1 and r.ttft_s is not None and r.ttft_s >= 0.015
