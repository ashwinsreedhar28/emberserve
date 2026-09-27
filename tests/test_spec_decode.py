"""Speculative decoding by n-gram lookup: exactness against plain greedy decoding, the
verification arithmetic, cache accounting, stop conditions, and the proposer itself."""

from __future__ import annotations

import pytest
import torch

from pagedserve import spec as specmod
from pagedserve.config import EngineConfig, ModelConfig
from pagedserve.engine import LLMEngine
from pagedserve.llm import LLM
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from pagedserve.sched.request import FinishReason, SamplingParams
from pagedserve.spec import accepted_prefix, propose_ngram
from tests.test_engine import prompts

torch.set_num_threads(2)
CFG = ModelConfig.tiny()


def make_engine(spec: bool, k: int = 4, ngram: int = 3, backend: str = "paged_torch",
                num_blocks: int = 256, block_size: int = 4, max_model_len: int = 256,
                chunked: bool = False, max_batched: int = 512, prefix: bool = False,
                async_scheduling: bool = False) -> LLMEngine:
    model = Qwen2ForCausalLM(CFG)
    reset_parameters_deterministic(model, 0)
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=block_size,
                        num_gpu_blocks=num_blocks, max_num_seqs=64, max_num_batched_tokens=max_batched,
                        max_model_len=max_model_len, attn_backend=backend,
                        enable_chunked_prefill=chunked, enable_prefix_caching=prefix,
                        async_scheduling=async_scheduling,
                        speculative_ngram=ngram if spec else 0, num_speculative_tokens=k if spec else 0)
    return LLMEngine(model, CFG, ecfg, tokenizer=None)


def gen(eng, ps, sp):
    return LLM.from_engine(eng).generate(ps, sp)


def same(a, b):
    assert [r.output_token_ids for r in a] == [r.output_token_ids for r in b]
    assert [r.finish_reason for r in a] == [r.finish_reason for r in b]


# ---- proposer ----------------------------------------------------------------------------
def test_propose_ngram_finds_the_most_recent_earlier_occurrence():
    toks = [1, 2, 3, 9, 9, 1, 2, 3, 7, 7, 7, 1, 2, 3]
    assert propose_ngram(toks, 3, 4) == [7, 7, 7, 1]   # after the (1,2,3) at index 5, cut at k
    assert propose_ngram(toks, 3, 2) == [7, 7]
    assert propose_ngram([1, 2, 3, 4, 1, 2], 3, 3) == [3, 4, 1]  # falls back to a 2-gram
    assert propose_ngram([5, 6, 7, 8], 3, 3) == []  # no repeat anywhere
    assert propose_ngram([4, 4], 3, 3) == [4]  # what followed the earlier 4 was... the last 4
    assert propose_ngram([4, 4, 4], 3, 5) == [4]  # (4,4) at 0 is followed by one token only
    assert propose_ngram([], 3, 3) == [] and propose_ngram([1, 2, 1], 3, 0) == []


def test_accepted_prefix():
    assert accepted_prefix([1, 2, 3], [1, 2, 3, 9]) == 3
    assert accepted_prefix([1, 2, 3], [1, 5, 3, 9]) == 1
    assert accepted_prefix([1, 2, 3], [4, 2, 3, 9]) == 0
    assert accepted_prefix([], [7]) == 0


# ---- engine: exactness --------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["naive", "paged_torch"])
@pytest.mark.parametrize("chunked", [False, True])
def test_spec_matches_greedy(backend, chunked):
    """Whatever the proposer guesses, the output equals plain greedy decoding. The tiny
    random model repeats itself a lot, so the n-gram lookup does fire."""
    ps = prompts(6, seed=4) + [prompts(1, seed=9)[0] * 3]  # a prompt with an obvious repeat
    sp = SamplingParams.greedy(24, ignore_eos=True)
    ref = gen(make_engine(False, backend=backend, chunked=chunked, max_batched=32 if chunked else 512), ps, sp)
    eng = make_engine(True, backend=backend, chunked=chunked, max_batched=32 if chunked else 512)
    same(gen(eng, ps, sp), ref)
    assert eng.spec_drafted > 0, "the proposer never fired"
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks
    assert not eng.async_scheduling


def test_perfect_drafts_cut_steps_and_junk_drafts_change_nothing(monkeypatch):
    """Feed the proposer the true greedy continuation: every draft is accepted and the run
    takes ~1/(k+1) of the decode steps. Feed it junk: nothing accepted, same output."""
    ps = prompts(3, seed=5)
    sp = SamplingParams.greedy(20, ignore_eos=True)
    ref = gen(make_engine(False), ps, sp)
    truth = {tuple(p): r.output_token_ids for p, r in zip(ps, ref)}

    def oracle(tokens, max_ngram, k, min_ngram=1):
        for p, out in truth.items():
            if tokens[:len(p)] == list(p):
                done = len(tokens) - len(p)
                return out[done:done + k]
        return []

    monkeypatch.setattr(specmod, "propose_ngram", oracle)
    eng = make_engine(True, k=4)
    same(gen(eng, ps, sp), ref)
    assert eng.spec_accepted == eng.spec_drafted > 0
    # 20 tokens: prefill step + ceil(19 / 5) verification steps (each yields up to 5)
    assert eng._step_count <= 1 + 4 + 1

    monkeypatch.setattr(specmod, "propose_ngram", lambda tokens, n, k, min_ngram=1: [CFG.vocab_size - 1] * k)
    eng = make_engine(True, k=4)
    same(gen(eng, ps, sp), ref)
    assert eng.spec_drafted > 0 and eng.spec_accepted == 0
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks


def test_eos_inside_accepted_drafts_and_max_tokens(monkeypatch):
    """A stop token that arrives as an accepted draft ends the request exactly there; a
    max_tokens limit is never overrun by drafts."""
    ps = prompts(2, seed=6)
    ref = gen(make_engine(False), ps, SamplingParams.greedy(30, ignore_eos=True))
    truth = {tuple(p): r.output_token_ids for p, r in zip(ps, ref)}

    def oracle(tokens, max_ngram, k, min_ngram=1):
        for p, out in truth.items():
            if tokens[:len(p)] == list(p):
                done = len(tokens) - len(p)
                return out[done:done + k]
        return []

    monkeypatch.setattr(specmod, "propose_ngram", oracle)
    stop = [r.output_token_ids[7] for r in ref]  # the 8th token of each
    sps = [SamplingParams.greedy(30, stop_token_ids=[s]) for s in stop]
    got = gen(make_engine(True, k=5), ps, sps)
    for r, g, s in zip(ref, got, stop):
        assert g.output_token_ids == r.output_token_ids[: r.output_token_ids.index(s) + 1]
        assert g.finish_reason == FinishReason.STOP
    got = gen(make_engine(True, k=5), ps, SamplingParams.greedy(13, ignore_eos=True))
    for r, g in zip(ref, got):
        assert g.output_token_ids == r.output_token_ids[:13] and g.finish_reason == FinishReason.LENGTH


def test_preemption_and_prefix_caching_with_drafts():
    ps = prompts(5, seed=3)
    sp = SamplingParams.greedy(24, ignore_eos=True)
    ref = gen(make_engine(False, num_blocks=20), ps, sp)
    eng = make_engine(True, num_blocks=20)
    same(gen(eng, ps, sp), ref)
    assert any(s.num_preempted > 0 for s in eng.stats)
    assert eng.block_manager.num_free_blocks == 20
    base = prompts(1, seed=2)[0] * 2
    ps2 = [base + p for p in prompts(3, seed=8)]
    same(gen(make_engine(True, prefix=True), ps2, sp), gen(make_engine(False, prefix=True), ps2, sp))


def test_sampled_requests_are_not_drafted():
    ps = prompts(3, seed=6)
    sp = SamplingParams(max_tokens=12, temperature=0.8, top_p=0.9, seed=123, ignore_eos=True)
    eng = make_engine(True)
    same(gen(eng, ps, sp), gen(make_engine(False), ps, sp))
    assert eng.spec_drafted == 0


def test_max_model_len_with_drafts():
    p = prompts(1)[0]
    sp = SamplingParams.greedy(100, ignore_eos=True)
    a = gen(make_engine(True, max_model_len=32), [p], sp)[0]
    b = gen(make_engine(False, max_model_len=32), [p], sp)[0]
    assert a.output_token_ids == b.output_token_ids and a.finish_reason == FinishReason.LENGTH


def test_async_is_turned_off_with_a_warning():
    with pytest.warns(UserWarning):
        eng = make_engine(True, async_scheduling=True)
    assert not eng.async_scheduling


def test_block_manager_truncate():
    from pagedserve.kv.block_manager import BlockManager

    bm = BlockManager(16, 4)
    bm.allocate(0, 6)  # 2 blocks
    bm.append_slots(0, 5)  # 11 tokens -> 3 blocks
    assert bm.get_num_tokens(0) == 11 and len(bm.get_block_table(0)) == 3
    released = bm.truncate(0, 7)  # back to 2 blocks
    assert bm.get_num_tokens(0) == 7 and len(bm.get_block_table(0)) == 2 and len(released) == 1
    assert bm.num_free_blocks == 14
    assert bm.truncate(0, 7) == []
