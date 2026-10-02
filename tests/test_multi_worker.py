"""Several API workers sharing one engine core (`serve --api-workers N`, server/multi.py):
requests route back to the worker that added them with the same tokens the in-process
engine produces, aborts and ownership stay per worker, a closed worker's requests are
aborted while the other keeps serving, `/metrics` sums every worker, and over HTTP the
kernel spreads connections across the workers."""

from __future__ import annotations

import asyncio
import socket
import time

import httpx
import pytest

from emberserve.llm import LLM
from emberserve.sched.request import SamplingParams
from emberserve.server.async_engine import AsyncEngineCoreClient, SharedCounters
from emberserve.server.engine_core import AttachedCore, spawn_core
from emberserve.server.multi import MultiServer
from tests.stub_tokenizer import StubTokenizer, install
from tests.test_engine import make_engine, prompts
from tests.test_engine_core import collect, spec

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _clients(n: int, **kw):
    proc, chans = spawn_core(spec(**kw), n)
    shared = SharedCounters(n)
    clients = [AsyncEngineCoreClient(spec(**kw), StubTokenizer(),
                                     core=AttachedCore(c[0], c[1], proc.pid), shared=(shared, i))
               for i, c in enumerate(chans)]
    return proc, clients


def _stop(proc, clients) -> None:
    for c in clients:
        c.stop()
    proc.join(10)
    if proc.is_alive():
        proc.kill()


async def test_two_workers_get_their_own_tokens_and_shared_metrics() -> None:
    proc, (a, b) = _clients(2)
    try:
        a.start()
        b.start()
        ps = prompts(8, seed=5)
        sp = SamplingParams.greedy(10, ignore_eos=True)
        ref = LLM.from_engine(install(make_engine())).generate(ps, sp)
        jobs = [collect(a if i % 2 == 0 else b, f"r{i}", p, sp) for i, p in enumerate(ps)]
        outs = await asyncio.gather(*jobs)
        for i, (r, chunks) in enumerate(zip(ref, outs)):
            assert chunks[-1].finished and chunks[-1].output_token_ids == r.output_token_ids, i
        for c in (a, b):  # either worker reports the server-wide totals
            m = c.metrics()
            assert m["api_workers"] == 2
            assert m["requests_finished_total"] == 8 and m["requests_received_total"] == 8
            assert m["generated_tokens_total"] == 80
            assert m["ttft_count"] == 8 and m["ttft_s_sum"] > 0
        assert {a.metrics()["api_worker"], b.metrics()["api_worker"]} == {0, 1}
    finally:
        _stop(proc, (a, b))


async def test_closed_worker_is_aborted_other_keeps_serving() -> None:
    proc, (a, b) = _clients(2)
    try:
        a.start()
        b.start()
        long = SamplingParams.greedy(200, ignore_eos=True)
        gen = a.generate("a0", prompts(1, seed=2)[0], long)
        first = await gen.__anext__()  # a's request is running in the core
        assert first.new_token_ids
        a.stop()  # its pipes close: the core must abort a0, not keep decoding it forever
        sp = SamplingParams.greedy(6, ignore_eos=True)
        out = await collect(b, "b0", prompts(1, seed=3)[0], sp)
        assert out[-1].finished and len(out[-1].output_token_ids) == 6
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            m = b.metrics()
            if m["requests_running"] == 0 and m["requests_waiting"] == 0:
                break
            await asyncio.sleep(0.1)
        assert b.metrics()["requests_running"] == 0  # a0 is gone from the core
        await gen.aclose()
    finally:
        _stop(proc, (b,))


def test_core_routes_rows_and_enforces_ownership() -> None:
    """Pipe-level: worker 1 cannot abort worker 0's request, and each worker's step
    messages carry only its own request ids."""
    from emberserve.server.engine_core import CoreRequest

    proc, chans = spawn_core(spec(), 2)
    try:
        for _, out in chans:
            assert out.recv()[0] == "ready"
        sp = SamplingParams.greedy(5, ignore_eos=True)
        now = time.perf_counter()
        chans[0][0].send(("add", CoreRequest("x", [5, 6, 7], sp, now)))
        chans[1][0].send(("add", CoreRequest("y", [8, 9], sp, now)))
        chans[1][0].send(("abort", "x"))  # not y's owner: ignored
        chans[1][0].send(("add", CoreRequest("x", [1, 2], sp, now)))  # duplicate id: refused
        seen = {0: {}, 1: {}}
        failed = []
        deadline = time.monotonic() + 30
        while (seen[0].get("x", 0) < 5 or seen[1].get("y", 0) < 5) and time.monotonic() < deadline:
            for w, (_, out) in enumerate(chans):
                while out.poll(0.05):
                    msg = out.recv()
                    if msg[0] == "step":
                        for rid, toks, _fin, _r in msg[1]:
                            seen[w][rid] = seen[w].get(rid, 0) + len(toks)
                    elif msg[0] == "failed":
                        failed.append((w, msg[1]))
        assert seen[0] == {"x": 5} and seen[1] == {"y": 5}
        assert failed == [(1, ["x"])]
    finally:
        for cmd, out in chans:
            cmd.close()
            out.close()
        proc.join(10)
        if proc.is_alive():
            proc.kill()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_two_workers_share_the_port() -> None:
    port = _free_port()
    srv = MultiServer(spec(), StubTokenizer(), "tiny", "127.0.0.1", port, 2, log_level="warning")
    srv.start()
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() < deadline, "server did not come up"
            time.sleep(0.3)

        async def fire() -> list[dict]:
            async with httpx.AsyncClient(base_url=base, timeout=60) as c:
                async def one(i: int) -> dict:
                    r = await c.post("/v1/completions", json={
                        "model": "tiny", "prompt": f"hello {i}", "max_tokens": 8,
                        "temperature": 0, "ignore_eos": True})
                    assert r.status_code == 200, r.text
                    return r.json()
                return await asyncio.gather(*(one(i) for i in range(12)))

        outs = asyncio.run(fire())
        assert all(o["usage"]["completion_tokens"] == 8 for o in outs)
        answered = set()
        for _ in range(100):  # a fresh connection each time: whichever worker accepts
            with httpx.Client(base_url=base, timeout=5) as c:
                m = c.get("/metrics").json()
            answered.add(m["api_worker"])
            assert m["requests_finished_total"] == 12
        assert answered == {0, 1}
        assert srv.alive()
    finally:
        srv.stop()
    assert not srv.workers and srv.core is None
