"""AsyncLLMEngine: worker-thread loop, concurrent streams, cancellation, error propagation."""

from __future__ import annotations

import asyncio

import pytest

from pagedserve.llm import LLM
from pagedserve.sched.request import RequestOutput, SamplingParams
from pagedserve.server.async_engine import AsyncLLMEngine, EngineNotRunningError
from tests.stub_tokenizer import install
from tests.test_engine import make_engine, prompts

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def collect(aeng: AsyncLLMEngine, rid: str, prompt: list[int],
                  sp: SamplingParams) -> list[RequestOutput]:
    return [out async for out in aeng.generate(rid, prompt, sp)]


async def test_concurrent_generate_matches_offline() -> None:
    ps = prompts(8, seed=5)
    sp = SamplingParams.greedy(12, ignore_eos=True)
    ref = LLM.from_engine(install(make_engine())).generate(ps, sp)
    aeng = AsyncLLMEngine(install(make_engine()))
    aeng.start()
    try:
        outs = await asyncio.gather(*(collect(aeng, f"r{i}", p, sp) for i, p in enumerate(ps)))
    finally:
        aeng.stop()
    for r, chunks in zip(ref, outs, strict=True):
        assert [c.new_token_ids[0] for c in chunks] == r.output_token_ids
        assert chunks[-1].finished and chunks[-1].output_token_ids == r.output_token_ids
        assert "".join(c.text_delta for c in chunks) == r.text
    m = aeng.metrics()
    assert m["requests_finished_total"] == 8 and m["generated_tokens_total"] == 8 * 12
    assert m["prefill_steps_total"] >= 1 and m["decode_steps_total"] >= 11


async def test_cancel_aborts_and_frees_blocks() -> None:
    eng = install(make_engine())
    aeng = AsyncLLMEngine(eng)
    aeng.start()
    ps = prompts(2, seed=9)
    try:
        long = SamplingParams.greedy(200, ignore_eos=True)
        gen = aeng.generate("victim", ps[0], long)
        other = asyncio.create_task(collect(aeng, "other", ps[1], SamplingParams.greedy(20, ignore_eos=True)))
        await gen.__anext__()
        await gen.__anext__()
        await gen.aclose()
        chunks = await other
        assert chunks[-1].finished and len(chunks) == 20
        for _ in range(50):
            if eng.block_manager.num_free_blocks == eng.block_manager.num_blocks:
                break
            await asyncio.sleep(0.02)
        assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks
        assert aeng.metrics()["requests_aborted_total"] == 1
        assert aeng.metrics()["requests_running"] == 0
    finally:
        aeng.stop()


async def test_step_exception_reaches_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = install(make_engine())
    calls = {"n": 0}
    real_step = eng.step

    def boom():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("kaboom")
        return real_step()

    monkeypatch.setattr(eng, "step", boom)
    aeng = AsyncLLMEngine(eng)
    aeng.start()
    try:
        ps = prompts(2)
        sp = SamplingParams.greedy(8, ignore_eos=True)
        results = await asyncio.gather(collect(aeng, "a", ps[0], sp), collect(aeng, "b", ps[1], sp),
                                       return_exceptions=True)
        assert all(isinstance(r, RuntimeError) and "kaboom" in str(r) for r in results)
        assert aeng.is_running  # the worker survived
        assert aeng.metrics()["step_errors_total"] == 1
        # And it still serves afterwards.
        out = await collect(aeng, "c", ps[0], sp)
        assert out[-1].finished
    finally:
        aeng.stop()


async def test_add_request_error_surfaces() -> None:
    aeng = AsyncLLMEngine(install(make_engine()))
    aeng.start()
    try:
        with pytest.raises(ValueError, match="empty prompt"):
            await collect(aeng, "e", [], SamplingParams.greedy(4))
    finally:
        aeng.stop()


async def test_stop_is_clean() -> None:
    aeng = AsyncLLMEngine(install(make_engine()))
    aeng.start()
    gen = aeng.generate("x", prompts(1)[0], SamplingParams.greedy(500, ignore_eos=True))
    await gen.__anext__()
    aeng.stop()
    assert not aeng.is_running
    with pytest.raises(EngineNotRunningError):
        await gen.__anext__()
    with pytest.raises(EngineNotRunningError):
        await collect(aeng, "y", prompts(1)[0], SamplingParams.greedy(4))
    aeng.stop()  # idempotent
    assert aeng.metrics()["requests_running"] == 0


async def test_async_scheduling_in_process_matches_offline() -> None:
    from tests.test_async_scheduling import make_engine as make_async_engine

    ps = prompts(8, seed=5)
    sp = SamplingParams.greedy(12, ignore_eos=True)
    ref = LLM.from_engine(install(make_engine())).generate(ps, sp)
    aeng = AsyncLLMEngine(install(make_async_engine(True)))
    aeng.start()
    try:
        outs = await asyncio.gather(*(collect(aeng, f"r{i}", p, sp) for i, p in enumerate(ps)))
    finally:
        aeng.stop()
    for r, chunks in zip(ref, outs, strict=True):
        assert [c.new_token_ids[0] for c in chunks] == r.output_token_ids
        assert chunks[-1].finished and chunks[-1].output_token_ids == r.output_token_ids
    assert aeng.metrics()["requests_finished_total"] == 8
