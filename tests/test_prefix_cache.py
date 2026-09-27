"""Automatic prefix caching: block-manager invariants and end-to-end engine equality."""

from __future__ import annotations

import random

import pytest
import torch

from pagedserve.config import EngineConfig
from pagedserve.engine import LLMEngine
from pagedserve.kv.block_manager import BlockManager, OutOfBlocksError
from pagedserve.kv.prefix_cache import PrefixCache
from pagedserve.llm import LLM
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from pagedserve.sched.request import SamplingParams
from tests.test_engine import CFG, make_engine

BLOCK = 4


def make_cached_engine(num_blocks: int = 256, max_batched: int = 512,
                       max_num_seqs: int = 64) -> LLMEngine:
    model = Qwen2ForCausalLM(CFG)
    reset_parameters_deterministic(model, 0)
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=BLOCK,
                        num_gpu_blocks=num_blocks, max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=max_batched, max_model_len=256,
                        attn_backend="paged_torch", enable_prefix_caching=True)
    return LLMEngine(model, CFG, ecfg, tokenizer=None)


def rand_ids(n: int, g: torch.Generator) -> list[int]:
    return torch.randint(2, CFG.vocab_size, (n,), generator=g).tolist()


def greedy(n: int) -> SamplingParams:
    return SamplingParams.greedy(n, ignore_eos=True)


def uncached_outputs(ps: list[list[int]], max_tokens: int, **kw) -> list[list[int]]:
    eng = make_engine("paged_torch", **kw)
    return [r.output_token_ids for r in LLM.from_engine(eng).generate(ps, greedy(max_tokens))]


def prefill_tokens(eng: LLMEngine) -> int:
    return sum(s.num_tokens for s in eng.stats if s.is_prefill)


def check_invariants(bm: BlockManager) -> None:
    """Refcounts equal table references; free <-> (ref 0, no hash); evictable <-> (ref 0, hash)."""
    refs = [0] * bm.num_blocks
    for table in bm._tables.values():
        for b in table:
            refs[b] += 1
    pc = bm._prefix
    free = set(bm._free)
    assert len(free) == len(bm._free), "duplicate block on the free list"
    for b in range(bm.num_blocks):
        assert bm.ref_count(b) == refs[b], (b, bm.ref_count(b), refs[b])
        hashed = pc.hash_of(b) is not None
        evictable = pc.is_evictable(b)
        if b in free:
            assert refs[b] == 0 and not hashed and not evictable
        elif evictable:
            assert refs[b] == 0 and hashed
        else:
            assert refs[b] > 0
    assert len(free) + pc.num_evictable + sum(1 for r in refs if r > 0) == bm.num_blocks


# ---- PrefixCache / BlockManager unit tests ---------------------------------------------

def test_block_hash_chains_on_prefix():
    h0 = PrefixCache.block_hash(None, (1, 2, 3, 4))
    assert h0 == PrefixCache.block_hash(None, (1, 2, 3, 4))
    assert PrefixCache.block_hash(h0, (5, 6, 7, 8)) != PrefixCache.block_hash(None, (5, 6, 7, 8))
    assert PrefixCache.block_hash(h0, (5, 6, 7, 8)) != PrefixCache.block_hash(h0, (5, 6, 7, 9))


def test_lru_eviction_order():
    pc = PrefixCache()
    for b in (0, 1, 2):
        pc.insert(100 + b, b)
        pc.mark_evictable(b)
    pc.unmark(0)  # 0 is referenced again; 1 is now the oldest evictable
    assert pc.evict_one() == 1 and pc.lookup(101) is None
    assert pc.evict_one() == 2 and pc.evict_one() is None
    assert pc.lookup(100) == 0 and pc.evictions == 2


def test_allocate_with_prefix_shares_and_caps_exact_multiple():
    bm = BlockManager(num_blocks=8, block_size=BLOCK, enable_prefix_caching=True)
    ids = list(range(10, 22))  # 12 tokens = 3 full blocks
    table, cached = bm.allocate_with_prefix(0, ids)
    assert cached == 0 and len(table) == 3
    bm.register_full_blocks(0, ids)
    # Exact multiple: the last block must still be computed, so only 2 blocks hit.
    m = bm.match_prefix(ids)
    assert m.num_hits == 2 and m.num_cached_tokens == 8 and m.num_new_blocks == 1
    table1, cached = bm.allocate_with_prefix(1, ids)
    assert cached == 8 and table1[:2] == table[:2] and table1[2] != table[2]
    assert bm.ref_count(table[0]) == 2 and bm.ref_count(table[2]) == 1
    # A longer prompt with the same 12-token prefix hits all 3 blocks.
    table2, cached = bm.allocate_with_prefix(2, ids + [99])
    assert cached == 12 and table2[:3] == table and len(table2) == 4
    # Seq 1 finishing registers nothing new (its 3rd block hashes to an owned prefix).
    bm.register_full_blocks(1, ids)
    assert bm._prefix.hash_of(table1[2]) is None
    bm.free(1)
    assert table1[2] in bm._free
    check_invariants(bm)


def test_free_parks_hashed_blocks_as_evictable_and_pop_evicts_lru():
    bm = BlockManager(num_blocks=3, block_size=BLOCK, enable_prefix_caching=True)
    a = list(range(8))
    bm.allocate_with_prefix(0, a + [50])  # 3 blocks, 9 tokens
    bm.register_full_blocks(0, a + [50])
    bm.free(0)
    assert bm.num_evictable_blocks == 2 and len(bm._free) == 1 and bm.num_free_blocks == 3
    check_invariants(bm)
    # Re-use: 2 hits on evictable blocks, 1 fresh block from the free list.
    m = bm.match_prefix(a + [51, 52])
    assert m.num_hits == 2 and m.num_evictable_hits == 2 and bm.can_allocate_with_prefix(a, m)
    table, cached = bm.allocate_with_prefix(1, a + [51, 52])
    assert cached == 8 and bm.num_evictable_blocks == 0 and bm.num_free_blocks == 0
    bm.free(1)
    # Free list is now empty; a disjoint prompt must evict cached blocks (LRU first).
    stats_before = bm.stats().prefix_cache.evictions
    bm.allocate_with_prefix(2, list(range(100, 112)))
    assert bm.stats().prefix_cache.evictions == stats_before + 2
    with pytest.raises(OutOfBlocksError):
        bm.append_slots(2, 1)
    check_invariants(bm)


def test_disabled_manager_is_unchanged():
    bm = BlockManager(num_blocks=4, block_size=BLOCK)
    assert not bm.prefix_caching_enabled and bm.num_evictable_blocks == 0
    bm.allocate(0, 8)
    bm.free(0)
    assert bm.num_free_blocks == 4 and bm.stats().prefix_cache is None


# ---- engine-level tests ---------------------------------------------------------------

def test_shared_system_prefix_reduces_second_prefill():
    g = torch.Generator().manual_seed(5)
    sys_prefix = rand_ids(16, g)
    a, b = sys_prefix + rand_ids(5, g), sys_prefix + rand_ids(7, g)
    ref = uncached_outputs([a, b], 8)

    eng = make_cached_engine()
    sp = greedy(8)
    eng.add_request("a", a, sp)
    eng.step()  # prefill a alone: 21 tokens, nothing cached yet
    assert eng.stats[0].is_prefill and eng.stats[0].num_tokens == 21
    eng.step()  # decode; at its start a's 4 full blocks get registered
    eng.add_request("b", b, sp)
    eng.step()  # prefill b with the 16-token prefix cached
    assert eng.stats[2].is_prefill and eng.stats[2].num_tokens == 23 - 16
    outs = {}
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                outs[o.request_id] = o.output_token_ids
    assert [outs["a"], outs["b"]] == ref
    assert eng.block_manager.stats().prefix_cache.hits == 4
    check_invariants(eng.block_manager)


def test_many_requests_shared_prefix_hit_rate():
    g = torch.Generator().manual_seed(9)
    shared = rand_ids(32, g)
    rng = random.Random(3)
    ps = [shared + rand_ids(rng.randint(1, 10), g) for _ in range(20)]
    ref = uncached_outputs(ps, 6)
    # A 48-token budget admits one prompt per prefill step, so every request after the
    # first sees the shared prefix already registered by its predecessor.
    eng = make_cached_engine(max_batched=48)
    res = LLM.from_engine(eng).generate(ps, greedy(6))
    assert [r.output_token_ids for r in res] == ref

    un = make_engine("paged_torch")
    LLM.from_engine(un).generate(ps, greedy(6))
    assert prefill_tokens(eng) < prefill_tokens(un)
    bm = eng.block_manager
    assert bm.num_free_blocks == bm.num_blocks
    assert bm.stats().prefix_cache.hit_rate > 0.5
    check_invariants(bm)


def test_sequential_reuse_after_finish():
    g = torch.Generator().manual_seed(21)
    p = rand_ids(19, g)  # 4 full blocks + 3 tokens
    ref = uncached_outputs([p], 5)[0]
    eng = make_cached_engine()
    a = LLM.from_engine(eng).generate([p], greedy(5))[0].output_token_ids
    assert a == ref
    n_stats = len(eng.stats)
    b = LLM.from_engine(eng).generate([p], greedy(5))[0].output_token_ids
    assert b == ref
    second_prefill = eng.stats[n_stats]
    assert second_prefill.is_prefill and second_prefill.num_tokens <= BLOCK
    assert second_prefill.num_tokens == 19 - 16
    check_invariants(eng.block_manager)


def test_eviction_under_tiny_budget_keeps_outputs_correct():
    g = torch.Generator().manual_seed(33)
    ps = [rand_ids(int(torch.randint(5, 13, (1,), generator=g)), g) for _ in range(12)]
    ref = uncached_outputs(ps, 6, num_blocks=8)
    eng = make_cached_engine(num_blocks=8)
    outs = []
    for i, p in enumerate(ps):
        outs.append(LLM.from_engine(eng).generate([p], greedy(6))[0].output_token_ids)
        check_invariants(eng.block_manager)
    assert outs == ref
    assert eng.block_manager.stats().prefix_cache.evictions > 0
    # Run the whole set concurrently too: preemption + eviction interleave.
    eng2 = make_cached_engine(num_blocks=8)
    res = LLM.from_engine(eng2).generate(ps, greedy(6))
    assert [r.output_token_ids for r in res] == ref
    assert any(s.num_preempted > 0 for s in eng2.stats)
    check_invariants(eng2.block_manager)


def test_preempted_request_hits_its_own_blocks_on_readmission():
    """6 blocks: r0 (8 prompt + 12 out -> 5 blocks) and r1 (8 + 8 -> 4 blocks) cannot both
    finish. r1 is preempted at r0's 13th token with 3 hashed blocks parked evictable; r0
    evicts the deepest ones (LRU-first) as it grows, finishes, and r1 re-prefills with
    its surviving leading block(s) as hits."""
    g = torch.Generator().manual_seed(44)
    p0, p1 = rand_ids(8, g), rand_ids(8, g)
    ref0, ref1 = uncached_outputs([p0], 12)[0], uncached_outputs([p1], 8)[0]
    eng = make_cached_engine(num_blocks=6)
    eng.add_request("r0", p0, greedy(12))
    eng.add_request("r1", p1, greedy(8))
    r1 = eng.scheduler.get_request("r1")
    outs = {}
    readmit = None
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                outs[o.request_id] = o.output_token_ids
        st = eng.stats[-1]
        if st.is_prefill and st.step > 1:
            readmit = (st, r1.num_tokens - 1)  # history re-prefilled (before its new token)
    assert any(s.num_preempted > 0 for s in eng.stats)
    assert readmit is not None
    pc = eng.block_manager.stats().prefix_cache
    assert pc.hits >= 1 and pc.evictions >= 1
    assert readmit[0].num_tokens == readmit[1] - BLOCK * pc.hits
    assert outs["r0"] == ref0 and outs["r1"] == ref1
    check_invariants(eng.block_manager)


def test_exact_multiple_prompt_fully_cached_still_computes_last_token():
    g = torch.Generator().manual_seed(55)
    p = rand_ids(8, g)
    ref = uncached_outputs([p], 3)[0]
    eng = make_cached_engine()
    assert LLM.from_engine(eng).generate([p], greedy(3))[0].output_token_ids == ref
    n = len(eng.stats)
    assert LLM.from_engine(eng).generate([p], greedy(3))[0].output_token_ids == ref
    st = eng.stats[n]
    assert st.is_prefill and st.num_tokens == BLOCK  # last block recomputed, first block hit
    assert eng.block_manager.stats().prefix_cache.hits == 1
    check_invariants(eng.block_manager)
