"""Engine core in a subprocess (`server/engine_core.py`) driven through
`AsyncEngineCoreClient`: outputs identical to the in-process engine, stop strings applied on
the client, cancellation aborts in the core, a bad prompt fails only its own stream."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import torch

from pagedserve.config import EngineConfig
from pagedserve.llm import LLM
from pagedserve.sched.request import FinishReason, RequestOutput, SamplingParams
from pagedserve.server.async_engine import AsyncEngineCoreClient
from pagedserve.server.engine_core import EngineSpec
from tests.stub_tokenizer import StubTokenizer, install
from tests.test_engine import CFG, make_engine, prompts

pytestmark = pytest.mark.anyio
TINY = dict(num_hidden_layers=CFG.num_hidden_layers, num_attention_heads=CFG.num_attention_heads,
            num_key_value_heads=CFG.num_key_value_heads, hidden_size=CFG.hidden_size,
            intermediate_size=CFG.intermediate_size, vocab_size=CFG.vocab_size)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def spec(**kw) -> EngineSpec:
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=4, num_gpu_blocks=256,
                        max_num_seqs=64, max_num_batched_tokens=512, max_model_len=256, **kw)
    return EngineSpec(ecfg, tiny=True, tiny_seed=0, tiny_overrides=TINY)


@pytest.fixture
async def client():
    c = AsyncEngineCoreClient(spec(), StubTokenizer())
    c.start()
    try:
        yield c
    finally:
        c.stop()


async def collect(c: AsyncEngineCoreClient, rid: str, prompt, sp: SamplingParams) -> list[RequestOutput]:
    return [out async for out in c.generate(rid, prompt, sp)]


async def test_matches_in_process_engine(client: AsyncEngineCoreClient) -> None:
    ps = prompts(8, seed=5)
    sp = SamplingParams.greedy(12, ignore_eos=True)
    ref = LLM.from_engine(install(make_engine())).generate(ps, sp)
    outs = await asyncio.gather(*(collect(client, f"r{i}", p, sp) for i, p in enumerate(ps)))
    for i, (r, chunks) in enumerate(zip(ref, outs)):
        assert chunks[-1].finished and chunks[-1].finish_reason is FinishReason.LENGTH
        assert chunks[-1].output_token_ids == r.output_token_ids, i
        assert [c.new_token_ids[0] for c in chunks] == r.output_token_ids
        assert "".join(c.text_delta for c in chunks) == r.text
        assert chunks[-1].metrics["num_prompt_tokens"] == len(ps[i])
    m = client.metrics()
    assert m["requests_finished_total"] == 8 and m["kv_blocks_used"] == 0
    # server-side latency sums (from the request's arrival at the API process)
    assert m["ttft_count"] == 8 and m["tpot_count"] == 8 and m["e2e_count"] == 8
    assert 0 < m["ttft_s_sum"] < m["e2e_s_sum"] and m["tpot_s_sum"] > 0


async def test_stop_string_is_applied_on_the_client(client: AsyncEngineCoreClient) -> None:
    """The core cannot see stop strings; the client truncates, finishes the stream and
    aborts the core-side request. Find a stop string from the greedy output itself."""
    p = prompts(1, seed=9)[0]
    full = await collect(client, "full", p, SamplingParams.greedy(24, ignore_eos=True))
    text = "".join(c.text_delta for c in full)
    assert len(text) > 6
    stop = text[3:6]
    outs = await collect(client, "stopped", p, SamplingParams.greedy(24, ignore_eos=True, stop=[stop]))
    got = "".join(c.text_delta for c in outs)
    assert outs[-1].finished and outs[-1].finish_reason is FinishReason.STOP
    assert got == text[:text.index(stop)]
    # The core-side request is gone: metrics show nothing running, all blocks free.
    for _ in range(50):
        m = client.metrics()
        if m["requests_running"] == 0 and m["kv_blocks_used"] == 0:
            break
        await asyncio.sleep(0.02)
    assert m["requests_running"] == 0 and m["kv_blocks_used"] == 0


async def test_cancel_aborts_in_core(client: AsyncEngineCoreClient) -> None:
    p = prompts(1, seed=3)[0]
    gen = client.generate("c", p, SamplingParams.greedy(200, ignore_eos=True))
    first = await gen.__anext__()
    assert first.new_token_ids
    await gen.aclose()
    for _ in range(50):
        m = client.metrics()
        if m["requests_running"] == 0 and m["requests_aborted_total"] == 1:
            break
        await asyncio.sleep(0.02)
    assert m["requests_running"] == 0 and m["requests_aborted_total"] == 1
    assert m["kv_blocks_used"] == 0


async def test_bad_prompt_fails_only_its_stream(client: AsyncEngineCoreClient) -> None:
    ok = asyncio.create_task(collect(client, "ok", prompts(1, seed=4)[0],
                                     SamplingParams.greedy(5, ignore_eos=True)))
    with pytest.raises(RuntimeError, match="too long|max_model_len|prompt"):
        await collect(client, "bad", list(range(2, 2 + 300)), SamplingParams.greedy(5))
    outs = await ok
    assert outs[-1].finished


async def test_served_over_http_with_engine_process() -> None:
    """The FastAPI app on top of the core client: streaming completion end to end."""
    from pagedserve.server.app import create_app

    c = AsyncEngineCoreClient(spec(), StubTokenizer())
    app = create_app(c, "tiny", manage_lifespan=True)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            r = await http.post("/v1/completions", json={"model": "tiny", "prompt": "hello there",
                                                         "max_tokens": 6, "temperature": 0,
                                                         "ignore_eos": True})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["usage"]["completion_tokens"] == 6
            assert body["choices"][0]["finish_reason"] == "length"
            m = (await http.get("/metrics")).json()
            assert m["engine_running"] and m["requests_finished_total"] == 1


async def test_async_scheduling_through_the_core_process() -> None:
    """The one-step-late outputs of async scheduling flow through the core process and the
    client unchanged; stop tokens end streams exactly where the sync engine does."""
    ps = prompts(6, seed=8)
    sp = SamplingParams.greedy(10, ignore_eos=True)
    ref = LLM.from_engine(install(make_engine())).generate(ps, sp)
    stop_sps = [SamplingParams.greedy(10, stop_token_ids=[r.output_token_ids[4]]) for r in ref]
    c = AsyncEngineCoreClient(spec(async_scheduling=True), StubTokenizer())
    c.start()
    try:
        outs = await asyncio.gather(*(collect(c, f"r{i}", p, sp) for i, p in enumerate(ps)))
        stopped = await asyncio.gather(*(collect(c, f"s{i}", p, s) for i, (p, s) in enumerate(zip(ps, stop_sps))))
    finally:
        c.stop()
    for r, chunks in zip(ref, outs, strict=True):
        assert [ch.new_token_ids[0] for ch in chunks] == r.output_token_ids
        assert chunks[-1].finished and chunks[-1].finish_reason is FinishReason.LENGTH
    for r, sp_stop, chunks in zip(ref, stop_sps, stopped, strict=True):
        stop = sp_stop.stop_token_ids[0]
        expect = r.output_token_ids[: r.output_token_ids.index(stop) + 1]
        assert [ch.new_token_ids[0] for ch in chunks] == expect
        assert chunks[-1].finish_reason is FinishReason.STOP


async def test_speculative_decoding_through_the_core_process() -> None:
    """Multi-token step rows (accepted drafts + the bonus token) stream through the core
    pipe and the client as one delta each, token-identical to plain greedy."""
    ps = prompts(5, seed=12) + [prompts(1, seed=9)[0] * 3]
    sp = SamplingParams.greedy(16, ignore_eos=True)
    ref = LLM.from_engine(install(make_engine())).generate(ps, sp)
    c = AsyncEngineCoreClient(spec(speculative_ngram=3, num_speculative_tokens=4), StubTokenizer())
    c.start()
    try:
        outs = await asyncio.gather(*(collect(c, f"r{i}", p, sp) for i, p in enumerate(ps)))
    finally:
        c.stop()
    multi = 0
    for r, chunks in zip(ref, outs, strict=True):
        got = [t for ch in chunks for t in ch.new_token_ids]
        assert got == r.output_token_ids
        assert chunks[-1].output_token_ids == r.output_token_ids and chunks[-1].finished
        assert "".join(ch.text_delta for ch in chunks) == r.text
        multi += sum(len(ch.new_token_ids) > 1 for ch in chunks)
    assert multi > 0, "no step delivered more than one token"
    assert c.metrics()["generated_tokens_total"] == 16 * len(ps)


def test_command_writer_never_blocks_on_a_full_pipe() -> None:
    """`send` used to write into the pipe on the caller's thread: with the core blocked on
    a full output pipe, the output reader (sending a stop-string abort) and the event loop
    (sending a large add) blocked with it, and nothing drained the outputs. The writer
    thread takes the blocking write; order is kept."""
    import time
    from multiprocessing import Pipe

    from pagedserve.server.engine_core import _CommandWriter

    r, w = Pipe(duplex=False)
    writer = _CommandWriter(w)
    big = ("add", b"x" * (4 << 20))  # far more than a pipe buffer holds
    t0 = time.monotonic()
    writer.put(big)
    writer.put(("abort", "a"))
    writer.put(("abort", "b"))
    assert time.monotonic() - t0 < 0.5  # returned with nobody reading
    assert r.recv() == big and r.recv() == ("abort", "a") and r.recv() == ("abort", "b")
    writer.close()


def test_command_writer_reports_a_closed_pipe() -> None:
    import time
    from multiprocessing import Pipe

    import pytest

    from pagedserve.server.engine_core import _CommandWriter

    r, w = Pipe(duplex=False)
    writer = _CommandWriter(w)
    r.close()
    writer.put(("abort", "x"))
    deadline = time.monotonic() + 5
    while writer.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(BrokenPipeError):
        writer.put(("abort", "y"))
