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


# ---- 8. chunked prefill --------------------------------------------------------------

def chunked_loop(**kw) -> FakeEngineLoop:
    kw.setdefault("num_blocks", 256)
    kw.setdefault("block_size", BLOCK)
    return FakeEngineLoop(enable_chunked_prefill=True, **kw)


def test_chunked_long_prompt_alone_is_split_exactly():
    loop = chunked_loop(max_num_batched_tokens=32)
    p = prompt(100)
    rid = loop.add(p, max_tokens=1)  # 100 > 32 is accepted with chunking on
    req = loop.scheduler.get_request(rid)
    recs = []
    for _ in range(4):
        out = loop.step()
        recs.append(loop.log[-1])
        assert out.num_tokens <= 32
    assert [r.query_lens for r in recs] == [[32], [32], [32], [4]]
    assert [r.prefill_complete for r in recs] == [[False], [False], [False], [True]]
    assert [r.is_prefill for r in recs] == [True, True, True, True]
    assert all(r.num_decode_tokens == 0 for r in recs)
    assert loop.outputs[rid] == expected_output(p, 1)  # sampled exactly once
    assert req.is_finished and loop.is_done() and loop.num_steps == 4
    assert loop.block_manager.num_free_blocks == 256


def test_chunked_decodes_keep_flowing_while_long_prompt_is_chunked():
    budget = 16
    loop = chunked_loop(max_num_batched_tokens=budget)
    shorts = [prompt(4, base=10 * i) for i in range(3)]
    short_ids = [loop.add(p, max_tokens=30) for p in shorts]
    first = loop.step()
    assert first.is_prefill and first.query_lens == [4, 4, 4]
    loop.step()  # a pure decode step
    assert not loop.log[-1].is_prefill and loop.log[-1].num_decode_tokens == 3
    long_p = prompt(50, base=200)
    long_id = loop.add(long_p, max_tokens=3)
    long_req = loop.scheduler.get_request(long_id)
    steps_while_chunking = 0
    while long_req.num_computed_tokens < len(long_p):
        loop.step()
        rec = loop.log[-1]
        steps_while_chunking += 1
        assert rec.num_tokens <= budget
        # every decoding request got exactly one token this step, in the leading rows
        assert rec.scheduled[:3] == short_ids and rec.query_lens[:3] == [1, 1, 1]
        assert rec.num_decode_tokens == 3
        assert rec.scheduled[3] == long_id and rec.query_lens[3] <= budget - 3
        assert rec.is_prefill
    # 50 tokens at 13 per step: 3 partial chunks + one 11-token completing chunk.
    assert steps_while_chunking == 4
    assert [r.query_lens[3] for r in loop.log[-4:]] == [13, 13, 13, 11]
    assert loop.log[-1].prefill_complete == [True, True, True, True]
    assert len(loop.outputs[long_id]) == 1
    loop.run_until_done(max_steps=100)
    for rid, p in zip(short_ids, shorts):
        assert loop.outputs[rid] == expected_output(p, 30)
    assert loop.outputs[long_id] == expected_output(long_p, 3)
    assert all(r.num_tokens <= budget for r in loop.log)
    assert loop.block_manager.num_free_blocks == 256


def test_chunked_is_fifo_across_waiting_long_prompts():
    loop = chunked_loop(max_num_batched_tokens=10)
    pa, pb = prompt(24, base=0), prompt(7, base=100)
    ra, rb = loop.add(pa, 2), loop.add(pb, 2)
    out = loop.step()
    assert [r.request_id for r in out.scheduled] == [ra] and out.query_lens == [10]
    out = loop.step()
    assert [r.request_id for r in out.scheduled] == [ra] and out.query_lens == [10]
    out = loop.step()  # ra's last 4 tokens complete, then rb gets a partial 6-token chunk
    assert [r.request_id for r in out.scheduled] == [ra, rb]
    assert out.query_lens == [4, 6] and out.prefill_complete == [True, False]
    assert loop.scheduler.num_waiting == 0
    out = loop.step()  # ra decodes (1 slot), rb finishes its 1 remaining token
    assert [r.request_id for r in out.scheduled] == [ra, rb]
    assert out.query_lens == [1, 1] and out.prefill_complete == [True, True]
    assert out.num_decode_tokens == 1 and not out.is_prefill  # 1-token completing chunk
    loop.run_until_done(max_steps=50)
    assert loop.outputs[ra] == expected_output(pa, 2)
    assert loop.outputs[rb] == expected_output(pb, 2)


def test_chunked_preemption_mid_prefill_recovers():
    # 8 blocks of 4 = 32 slots. Two short decoders plus a 20-token prompt (5 blocks) make
    # the cache overflow while the long prompt is still being chunked in.
    loop = chunked_loop(num_blocks=8, max_num_batched_tokens=6)
    shorts = [prompt(4, base=10), prompt(4, base=50)]
    sids = [loop.add(p, max_tokens=10) for p in shorts]
    loop.step()  # both prefilled (2 blocks)
    long_p = prompt(20, base=100)
    lid = loop.add(long_p, max_tokens=4)
    loop.run_until_done(max_steps=300)
    preempted = [rid for rec in loop.log for rid in rec.preempted]
    assert preempted, "expected preemption on an 8-block cache"
    assert all(rec.num_tokens <= 6 for rec in loop.log)
    for rid, p in zip(sids, shorts):
        assert loop.outputs[rid] == expected_output(p, 10)
    assert loop.outputs[lid] == expected_output(long_p, 4)
    assert loop.block_manager.num_free_blocks == 8
    assert loop.block_manager.stats().used_token_slots == 0


def test_chunked_flag_off_is_unchanged():
    rng = random.Random(7)
    arrivals = poisson_arrivals(30, rate=1.0, rng=rng)
    jobs = [(prompt(rng.randint(5, 30), base=rng.randint(0, 200)), rng.randint(4, 20))
            for _ in range(30)]
    kw = dict(num_blocks=64, block_size=BLOCK, max_num_seqs=6, max_num_batched_tokens=40)

    def run(chunked: bool):
        loop = FakeEngineLoop(enable_chunked_prefill=chunked, **kw)
        ids = [loop.add_at(step, p, mt) for step, (p, mt) in zip(arrivals, jobs)]
        steps = loop.run_until_done(max_steps=5000)
        log = [(r.scheduled, r.query_lens, r.is_prefill, r.preempted) for r in loop.log]
        return steps, [loop.outputs[i] for i in ids], log

    off_steps, off_out, off_log = run(False)
    assert off_out == [expected_output(p, mt) for p, mt in jobs]
    # The flag-off path is the pre-existing scheduler byte for byte: no mixed steps.
    assert all((qls == [1] * len(qls)) != pre for _, qls, pre, _ in off_log)
    on_steps, on_out, _ = run(True)
    assert on_out == off_out
    assert on_steps <= off_steps


def test_chunked_add_request_accepts_prompt_over_budget():
    loop = chunked_loop(max_num_batched_tokens=8, max_model_len=64)
    rid = loop.add(prompt(40), 1)
    assert loop.scheduler.get_request(rid).state == RequestState.WAITING
    with pytest.raises(ValueError):
        loop.add(prompt(64), 1)  # max_model_len still applies
