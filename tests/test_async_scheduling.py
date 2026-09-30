"""Async scheduling (`EngineConfig.async_scheduling`): the engine launches step N+1 before
reading step N's tokens back. Every scenario must produce exactly the tokens the
synchronous engine produces, step for step, on the tiny deterministic model (CPU; on CUDA
the same tests run through tests/test_cuda_graphs_gpu.py-style GPU engines)."""

from __future__ import annotations

import pytest
import torch

from pagedserve.config import EngineConfig, ModelConfig
from pagedserve.engine import LLMEngine
from pagedserve.llm import LLM
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from pagedserve.sched.request import FinishReason, SamplingParams
from tests.test_engine import prompts

torch.set_num_threads(2)
CFG = ModelConfig.tiny()


def make_engine(async_scheduling: bool, backend: str = "paged_torch", num_blocks: int = 256,
                block_size: int = 4, max_model_len: int = 256, chunked: bool = False,
                max_batched: int = 512, prefix: bool = False, max_num_seqs: int = 64) -> LLMEngine:
    model = Qwen2ForCausalLM(CFG)
    reset_parameters_deterministic(model, 0)
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=block_size,
                        num_gpu_blocks=num_blocks, max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=max_batched, max_model_len=max_model_len,
                        attn_backend=backend, enable_chunked_prefill=chunked,
                        enable_prefix_caching=prefix, async_scheduling=async_scheduling)
    return LLMEngine(model, CFG, ecfg, tokenizer=None)


def gen(eng: LLMEngine, ps, sp):
    return LLM.from_engine(eng).generate(ps, sp)


def same(a, b) -> None:
    assert [r.output_token_ids for r in a] == [r.output_token_ids for r in b]
    assert [r.finish_reason for r in a] == [r.finish_reason for r in b]


@pytest.mark.parametrize("backend", ["naive", "paged_torch"])
def test_greedy_batch_matches_sync(backend: str) -> None:
    ps = prompts(6)
    sp = SamplingParams.greedy(16, ignore_eos=True)
    sync = make_engine(False, backend)
    asyn = make_engine(True, backend)
    same(gen(asyn, ps, sp), gen(sync, ps, sp))
    # length-ended requests are anticipated: no extra step was computed
    assert asyn._step_count == sync._step_count
    assert asyn._pending is None and not asyn.has_unfinished_requests()
    assert asyn.block_manager.num_free_blocks == asyn.block_manager.num_blocks


def test_outputs_arrive_one_step_late_and_stuck_flag() -> None:
    eng = make_engine(True)
    ps = prompts(2)
    sp = SamplingParams.greedy(4, ignore_eos=True)
    eng.add_request("a", ps[0], sp)
    eng.add_request("b", ps[1], sp)
    assert eng.step() == [] and eng.last_step_scheduled  # prefill launched, nothing pending
    assert eng.has_unfinished_requests()
    out = eng.step()  # decode launched; prefill's tokens come back now
    assert sorted(o.request_id for o in out) == ["a", "b"]
    assert all(len(o.output_token_ids) == 1 for o in out)
    while eng.has_unfinished_requests():
        eng.step()
    # nothing scheduled and nothing pending: a step is a no-op that says so
    assert eng.step() == [] and not eng.last_step_scheduled


def test_eos_finish_matches_sync_and_discards_the_extra_token() -> None:
    ps = prompts(3, seed=5)
    ref = gen(make_engine(False), ps, SamplingParams.greedy(12, ignore_eos=True))
    # use each request's 4th greedy token as its stop token: they end at different steps
    sps = [SamplingParams.greedy(12, stop_token_ids=[r.output_token_ids[3]]) for r in ref]
    sync_eng, async_eng = make_engine(False), make_engine(True)
    for i, (p, sp) in enumerate(zip(ps, sps)):
        sync_eng.add_request(str(i), p, sp)
        async_eng.add_request(str(i), p, sp)

    def drain(eng):
        finished = {}
        seen = []
        while eng.has_unfinished_requests():
            for o in eng.step():
                seen.append((o.request_id, o.new_token_ids[0]))
                if o.finished:
                    assert o.request_id not in finished, "finished twice"
                    finished[o.request_id] = (o.output_token_ids, o.finish_reason)
        return finished, seen

    fs, seen_s = drain(sync_eng)
    fa, seen_a = drain(async_eng)
    assert fa == fs
    assert seen_a == seen_s  # the discarded post-EOS token never surfaces
    assert all(r == FinishReason.STOP for _, r in fa.values())
    assert async_eng.block_manager.num_free_blocks == async_eng.block_manager.num_blocks


def test_preemption_matches_sync() -> None:
    ps = prompts(5, seed=3)
    sp = SamplingParams.greedy(24, ignore_eos=True)
    sync = make_engine(False, num_blocks=20)
    asyn = make_engine(True, num_blocks=20)
    same(gen(asyn, ps, sp), gen(sync, ps, sp))
    assert any(s.num_preempted > 0 for s in asyn.stats)
    assert asyn.block_manager.num_free_blocks == 20


@pytest.mark.parametrize("backend", ["naive", "paged_torch"])
def test_chunked_prefill_matches_sync(backend: str) -> None:
    ps = prompts(6, seed=9)
    sp = SamplingParams.greedy(10, ignore_eos=True)
    sync = make_engine(False, backend, chunked=True, max_batched=12)
    asyn = make_engine(True, backend, chunked=True, max_batched=12)
    same(gen(asyn, ps, sp), gen(sync, ps, sp))
    assert asyn._step_count == sync._step_count


def test_prefix_caching_matches_sync() -> None:
    base = prompts(1, seed=2)[0] * 3  # a long shared prefix
    ps = [base + p for p in prompts(4, seed=4)]
    sp = SamplingParams.greedy(8, ignore_eos=True)
    same(gen(make_engine(True, prefix=True), ps, sp), gen(make_engine(False, prefix=True), ps, sp))


def test_continuous_admission_and_abort() -> None:
    eng = make_engine(True, num_blocks=64)
    ps = prompts(4, seed=11)
    sp = SamplingParams.greedy(14, ignore_eos=True)
    eng.add_request("a", ps[0], sp)
    eng.add_request("b", ps[1], sp)
    eng.step()
    eng.step()
    eng.add_request("c", ps[2], sp)
    eng.step()
    eng.abort_request("a")  # its token from the step just launched must be dropped
    eng.add_request("d", ps[3], sp)
    got = {}
    while eng.has_unfinished_requests():
        for o in eng.step():
            assert o.request_id != "a"
            if o.finished:
                got[o.request_id] = o.output_token_ids
    ref = {k: r.output_token_ids for k, r in zip("bcd", gen(make_engine(False), ps[1:], sp))}
    assert got == ref
    assert eng.block_manager.num_free_blocks == 64


def test_max_model_len_matches_sync() -> None:
    p = prompts(1)[0]
    sp = SamplingParams.greedy(100, ignore_eos=True)
    a = gen(make_engine(True, max_model_len=32), [p], sp)[0]
    b = gen(make_engine(False, max_model_len=32), [p], sp)[0]
    assert a.output_token_ids == b.output_token_ids and a.finish_reason == FinishReason.LENGTH
    assert len(p) + len(a.output_token_ids) == 32


def test_seeded_sampling_matches_sync() -> None:
    ps = prompts(3, seed=6)
    sp = SamplingParams(max_tokens=12, temperature=0.8, top_p=0.9, seed=123, ignore_eos=True)
    same(gen(make_engine(True), ps, sp), gen(make_engine(False), ps, sp))


def test_reset_drops_pending() -> None:
    eng = make_engine(True)
    eng.add_request("a", prompts(1)[0], SamplingParams.greedy(4, ignore_eos=True))
    eng.step()
    assert eng._pending is not None
    eng.reset()
    assert eng._pending is None and not eng.has_unfinished_requests()
    same(gen(eng, prompts(2), SamplingParams.greedy(5, ignore_eos=True)),
         gen(make_engine(False), prompts(2), SamplingParams.greedy(5, ignore_eos=True)))


def _untied_engine(async_scheduling: bool) -> LLMEngine:
    cfg = ModelConfig.tiny(tie_word_embeddings=False)
    model = Qwen2ForCausalLM(cfg)
    reset_parameters_deterministic(model, 3)
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=4, num_gpu_blocks=256,
                        max_num_seqs=64, max_num_batched_tokens=512, max_model_len=256,
                        attn_backend="paged_torch", async_scheduling=async_scheduling)
    return LLMEngine(model, cfg, ecfg, tokenizer=None)


def test_repetition_penalty_sees_the_pending_token() -> None:
    """The penalty reads `all_token_ids` on the host; under async scheduling the token the
    previous step sampled was not in it yet, so it escaped the penalty (this model and
    prompt: sync 51, 202, 176, 247, 137, 0, 214 vs async ..., 0, 0). Penalized requests
    now resolve before the next launch; an unpenalized one in the same batch is unchanged."""
    p = [23, 88, 230, 29, 77]
    pen = SamplingParams.greedy(16, ignore_eos=True)
    pen.repetition_penalty = 1.2
    plain = SamplingParams.greedy(16, ignore_eos=True)
    want = gen(_untied_engine(False), [p], pen)[0].output_token_ids
    assert want[:7] == [51, 202, 176, 247, 137, 0, 214]
    assert gen(_untied_engine(True), [p], pen)[0].output_token_ids == want
    eng = _untied_engine(True)
    eng.add_request("p", p, pen)
    eng.add_request("q", prompts(1)[0], plain)
    outs: dict[str, list[int]] = {}
    while eng.has_unfinished_requests():
        for o in eng.step():
            outs[o.request_id] = o.output_token_ids
    assert outs["p"] == want
    assert outs["q"] == gen(_untied_engine(False), [prompts(1)[0]], plain)[0].output_token_ids
