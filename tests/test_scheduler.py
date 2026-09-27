"""Scheduler behaviour driven end to end through the tensor-free FakeEngineLoop."""

from __future__ import annotations

import random

import pytest

from pagedserve.sched.request import FinishReason, RequestState
from pagedserve.sched.scheduler import SchedulerOutput
from tests.fakes import FakeEngineLoop, StaticBatchLoop, expected_output, poisson_arrivals

BLOCK = 4


def prompt(n: int, base: int = 10) -> list[int]:
    return [base + i for i in range(n)]


# ---- 1. admission limits ------------------------------------------------------------

def test_prefill_respects_max_num_seqs():
    loop = FakeEngineLoop(num_blocks=64, block_size=BLOCK, max_num_seqs=2)
    ids = [loop.add(prompt(8, base=i), max_tokens=4) for i in range(3)]
    out = loop.step()
    assert out.is_prefill and out.query_lens == [8, 8]
    assert [r.request_id for r in out.scheduled] == ids[:2]
    assert loop.scheduler.num_waiting == 1 and loop.scheduler.num_running == 2


def test_prefill_respects_token_budget_and_is_fifo():
    loop = FakeEngineLoop(num_blocks=64, block_size=BLOCK, max_num_batched_tokens=20)
    ids = [loop.add(prompt(8), 4), loop.add(prompt(16), 4), loop.add(prompt(2), 4)]
    out = loop.step()
    # 8 + 16 > 20 stops the batch at r1; the 2-token r2 must NOT be pulled ahead of it.
    assert [r.request_id for r in out.scheduled] == [ids[0]]
    out = loop.step()
    assert out.is_prefill and [r.request_id for r in out.scheduled] == [ids[1], ids[2]]
    assert out.num_tokens == 18


def test_prefill_respects_free_blocks():
    loop = FakeEngineLoop(num_blocks=3, block_size=BLOCK)
    ids = [loop.add(prompt(8), 2), loop.add(prompt(8), 2)]  # 2 blocks each
    out = loop.step()
    assert [r.request_id for r in out.scheduled] == [ids[0]]
    assert loop.block_manager.num_free_blocks == 1


def test_schedule_is_empty_when_idle():
    loop = FakeEngineLoop(num_blocks=8, block_size=BLOCK)
    out = loop.scheduler.schedule()
    assert isinstance(out, SchedulerOutput) and out.is_empty and out.num_tokens == 0
    assert not loop.scheduler.has_unfinished_requests()


# ---- 2. continuous batching ------------------------------------------------------------

def test_late_arrival_joins_running_batch():
    loop = FakeEngineLoop(num_blocks=64, block_size=BLOCK)
    p0, p1, p2 = prompt(5, 10), prompt(6, 40), prompt(3, 90)
    r0, r1 = loop.add(p0, 10), loop.add(p1, 10)
    first = loop.step()
    assert first.is_prefill and first.query_lens == [5, 6]
    second = loop.step()
    assert not second.is_prefill and second.query_lens == [1, 1]

    r2 = loop.add(p2, 10)
    third = loop.step()
    assert third.is_prefill and [r.request_id for r in third.scheduled] == [r2]
    assert third.query_lens == [3]
    fourth = loop.step()
    assert not fourth.is_prefill
    assert [r.request_id for r in fourth.scheduled] == [r0, r1, r2]
    assert len(loop.outputs[r0]) == 3 and len(loop.outputs[r2]) == 2

    loop.run_until_done(max_steps=50)
    assert loop.outputs[r0] == expected_output(p0, 10)
    assert loop.outputs[r1] == expected_output(p1, 10)
    assert loop.outputs[r2] == expected_output(p2, 10)
    assert loop.block_manager.num_free_blocks == 64
    for rid in (r0, r1, r2):
        req = loop.scheduler.get_request(rid)
        assert req is None  # retired from the scheduler's index
    assert all(r.finish_reason == FinishReason.LENGTH for r in loop.requests.values())


# ---- 3 + 7. preemption --------------------------------------------------------------

def preemption_loop() -> tuple[FakeEngineLoop, list[str], list[list[int]]]:
    """4 requests that each need 4 blocks to finish, on an 8-block cache."""
    loop = FakeEngineLoop(num_blocks=8, block_size=BLOCK)
    prompts = [prompt(4, base=20 * i) for i in range(4)]
    ids = [loop.add(p, max_tokens=12) for p in prompts]
    return loop, ids, prompts


def test_preemption_recovers_and_finishes_everything():
    loop, ids, prompts = preemption_loop()
    loop.run_until_done(max_steps=200)
    preempted = [rid for rec in loop.log for rid in rec.preempted]
    assert preempted, "the cache is too small for four requests; something must be evicted"
    for rid, p in zip(ids, prompts):
        assert loop.outputs[rid] == expected_output(p, 12)
    assert loop.block_manager.num_free_blocks == 8
    assert loop.block_manager.stats().used_token_slots == 0


def test_preemption_evicts_youngest_and_keeps_fifo_order():
    loop, ids, _ = preemption_loop()
    out = loop.step()  # prefill all four
    assert out.is_prefill and len(out.scheduled) == 4
    while not out.preempted:
        out = loop.step()
    assert not out.is_prefill
    # Youngest first: r3 was evicted, then r2 could not be extended and evicted itself.
    assert [r.request_id for r in out.preempted] == [ids[3], ids[2]]
    assert [r.request_id for r in out.scheduled] == [ids[0], ids[1]]
    assert [r.request_id for r in loop.scheduler.waiting] == [ids[2], ids[3]]
    for rid in ids[2:]:
        req = loop.scheduler.get_request(rid)
        assert req.state == RequestState.PREEMPTED and req.num_computed_tokens == 0
        assert not loop.block_manager.has_sequence(req.seq_id)


def test_preempted_request_re_prefills_full_history_and_continues():
    loop, ids, prompts = preemption_loop()
    out = loop.step()
    while not out.preempted:
        out = loop.step()
    victim = loop.scheduler.get_request(ids[3])
    kept = list(victim.output_token_ids)
    assert len(kept) == 5 and kept == expected_output(prompts[3], 5)

    # Step until the victim is re-admitted; its query must cover prompt + kept outputs.
    while not (out.is_prefill and victim in out.scheduled):
        out = loop.step()
    idx = out.scheduled.index(victim)
    rec = loop.log[-1]
    assert out.query_lens[idx] == rec.num_tokens_at_schedule[idx] == 4 + 5
    assert victim.num_computed_tokens == 9, "engine advanced by the full re-prefill"
    # Generation continued from the kept history rather than restarting.
    assert victim.output_token_ids[:5] == kept and len(victim.output_token_ids) == 6
    loop.run_until_done(max_steps=200)
    assert victim.output_token_ids == expected_output(prompts[3], 12)


# ---- 4. abort ---------------------------------------------------------------------

def test_abort_waiting_request():
    loop = FakeEngineLoop(num_blocks=16, block_size=BLOCK, max_num_seqs=1)
    loop.add(prompt(4), 4)
    r1 = loop.add(prompt(4), 4)
    loop.step()
    assert loop.scheduler.num_waiting == 1
    req = loop.scheduler.abort_request(r1)
    assert req is not None and req.request_id == r1
    assert req.state == RequestState.FINISHED and req.finish_reason == FinishReason.ABORT
    assert req.finished_time is not None
    assert loop.scheduler.num_waiting == 0 and loop.scheduler.get_request(r1) is None
    assert loop.block_manager.num_free_blocks == 15  # only r0's block is held
    loop.run_until_done()
    assert loop.block_manager.num_free_blocks == 16
    assert loop.scheduler.abort_request(r1) is None, "already gone"


def test_abort_running_request_frees_blocks():
    loop = FakeEngineLoop(num_blocks=16, block_size=BLOCK)
    r0, r1 = loop.add(prompt(8), 6), loop.add(prompt(8), 6)
    loop.step()
    loop.step()
    assert loop.block_manager.num_free_blocks == 16 - 2 * 3
    req = loop.scheduler.abort_request(r0)
    assert req.state == RequestState.FINISHED and req.finish_reason == FinishReason.ABORT
    assert loop.scheduler.num_running == 1
    assert not loop.block_manager.has_sequence(req.seq_id)
    assert loop.block_manager.num_free_blocks == 16 - 3
    out = loop.step()
    assert [r.request_id for r in out.scheduled] == [r1]
    loop.run_until_done()
    assert loop.block_manager.num_free_blocks == 16
    assert loop.outputs[r1] == expected_output(prompt(8), 6)


def test_abort_preempted_request():
    loop, ids, prompts = preemption_loop()
    out = loop.step()
    while not out.preempted:
        out = loop.step()
    free_before = loop.block_manager.num_free_blocks
    req = loop.scheduler.abort_request(ids[2])
    assert req.state == RequestState.FINISHED and req.finish_reason == FinishReason.ABORT
    assert [r.request_id for r in loop.scheduler.waiting] == [ids[3]]
    assert loop.block_manager.num_free_blocks == free_before  # it held nothing
    loop.run_until_done(max_steps=200)
    for rid in (ids[0], ids[1], ids[3]):
        assert loop.outputs[rid] == expected_output(prompts[int(rid[1:])], 12)
    assert loop.outputs[ids[2]] == expected_output(prompts[2], 5)  # frozen at abort
    assert loop.block_manager.num_free_blocks == 8


def test_abort_unknown_request_returns_none():
    loop = FakeEngineLoop(num_blocks=8, block_size=BLOCK)
    assert loop.scheduler.abort_request("nope") is None


# ---- 5. continuous vs static batching under random arrivals ------------------------------

def test_continuous_beats_static_batching_with_identical_outputs():
    rng = random.Random(2026)
    arrivals = poisson_arrivals(40, rate=1.0, rng=rng)
    jobs = [(prompt(rng.randint(5, 30), base=rng.randint(0, 200)), rng.randint(8, 40))
            for _ in range(40)]
    kw = dict(num_blocks=256, block_size=BLOCK, max_num_seqs=8, max_num_batched_tokens=256)

    def run(loop_cls):
        loop = loop_cls(**kw)
        ids = [loop.add_at(step, p, mt) for step, (p, mt) in zip(arrivals, jobs)]
        steps = loop.run_until_done(max_steps=5000)
        assert loop.block_manager.num_free_blocks == 256
        return steps, [loop.outputs[i] for i in ids]

    cont_steps, cont_out = run(FakeEngineLoop)
    static_steps, static_out = run(StaticBatchLoop)
    assert cont_out == static_out == [expected_output(p, mt) for p, mt in jobs]
    assert cont_steps < static_steps, (cont_steps, static_steps)


# ---- 6. rejection -----------------------------------------------------------------

def test_add_request_rejects_oversized_prompts():
    loop = FakeEngineLoop(num_blocks=64, block_size=BLOCK, max_model_len=16,
                          max_num_batched_tokens=12)
    with pytest.raises(ValueError):
        loop.add(prompt(16), 1)  # == max_model_len
    with pytest.raises(ValueError):
        loop.add(prompt(13), 1)  # > max_num_batched_tokens
    with pytest.raises(ValueError):
        loop.add([], 1)
    rid = loop.add(prompt(12), 1)
    assert loop.scheduler.get_request(rid).state == RequestState.WAITING
    assert loop.scheduler.num_waiting == 1


def test_add_request_requires_assigned_seq_id():
    from pagedserve.sched.request import Request, SamplingParams

    loop = FakeEngineLoop(num_blocks=8, block_size=BLOCK)
    req = Request(request_id="x", prompt_token_ids=[1, 2], sampling_params=SamplingParams())
    with pytest.raises(AssertionError):
        loop.scheduler.add_request(req)
