"""PagedTorchAttentionBackend vs the dense causal reference, driven by a BlockManager.

The harness mirrors what the engine will do each step: allocate/append blocks, build the
slot mapping and block tables, call the backend per layer, and keep every sequence's
k/v history in plain lists so each output can be checked against
`causal_softmax_attention` over the full history.
"""

from collections import defaultdict

import pytest
import torch

from pagedserve.attn.base import AttnMetadata, causal_softmax_attention
from pagedserve.attn.paged_torch import (PagedTorchAttentionBackend, build_block_tables_tensor,
                                         build_slot_mapping)
from pagedserve.config import ModelConfig
from pagedserve.kv.block_manager import BlockManager
from pagedserve.kv.cache import PagedKVCache

CFG = ModelConfig.tiny(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                       hidden_size=64)  # H=4, Hkv=2, D=16
H, HKV, D, L = 4, 2, 16, 2
BLOCK = 4
DTYPES = [pytest.param(torch.float32, 1e-5, id="fp32"),
          pytest.param(torch.float16, 2e-2, id="fp16")]


class Harness:
    """Drives the backend like the engine and checks every output against the reference."""

    def __init__(self, cache_dtype: torch.dtype, atol: float, num_blocks: int, seed: int = 0):
        self.bm = BlockManager(num_blocks, BLOCK)
        self.cache = PagedKVCache(CFG, num_blocks, BLOCK, device="cpu", dtype=cache_dtype)
        self.backend = PagedTorchAttentionBackend(CFG, self.cache)
        self.atol = atol
        self.gen = torch.Generator().manual_seed(seed)
        self.hist_k: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        self.hist_v: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)

    def rand(self, n: int, heads: int) -> torch.Tensor:
        return torch.randn(n, heads, D, generator=self.gen)

    def step(self, seq_ids: list[int], query_lens: list[int], is_prefill: bool) -> AttnMetadata:
        """Run one step (all layers) and assert every sequence matches the reference."""
        starts = []
        for sid, n in zip(seq_ids, query_lens):
            if self.bm.has_sequence(sid):
                starts.append(self.bm.get_num_tokens(sid))
                self.bm.append_slots(sid, n)
            else:
                assert is_prefill
                starts.append(0)
                self.bm.allocate(sid, n)
        context_lens = [self.bm.get_num_tokens(s) for s in seq_ids]
        positions = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, query_lens)])
        meta = AttnMetadata(
            is_prefill=is_prefill, seq_ids=seq_ids, query_lens=query_lens,
            context_lens=context_lens, positions=positions,
            slot_mapping=build_slot_mapping(self.bm, seq_ids, starts, query_lens, "cpu"),
            block_tables=build_block_tables_tensor(
                [self.bm.get_block_table(s) for s in seq_ids], "cpu"),
            block_size=BLOCK,
            num_cached_tokens=starts if is_prefill else [],
        )
        n_tok = sum(query_lens)
        for layer in range(L):
            q, k, v = self.rand(n_tok, H), self.rand(n_tok, HKV), self.rand(n_tok, HKV)
            out = self.backend.forward(layer, q, k, v, meta)
            assert out.shape == (n_tok, H, D) and out.dtype == q.dtype
            off = 0
            for sid, n in zip(seq_ids, query_lens):
                self.hist_k[sid, layer].append(k[off:off + n])
                self.hist_v[sid, layer].append(v[off:off + n])
                k_all = torch.cat(self.hist_k[sid, layer])
                v_all = torch.cat(self.hist_v[sid, layer])
                assert k_all.shape[0] == context_lens[seq_ids.index(sid)]
                ref = causal_softmax_attention(q[off:off + n], k_all, v_all, n)
                torch.testing.assert_close(
                    out[off:off + n], ref, atol=self.atol, rtol=0,
                    msg=lambda m, s=sid, lyr=layer: f"seq {s} layer {lyr}: {m}")
                off += n
        return meta

    def free(self, seq_id: int) -> None:
        self.bm.free(seq_id)
        self.backend.free_sequence(seq_id)
        for layer in range(L):
            self.hist_k.pop((seq_id, layer), None)
            self.hist_v.pop((seq_id, layer), None)


def scramble_free_list(h: Harness, keep: tuple[int, ...] = (103, 110)) -> None:
    """Fill every block with a one-block dummy, then free them in a shuffled order so
    later allocations get non-contiguous tables. `keep` stay resident with live data."""
    dummies = list(range(100, 100 + h.bm.num_blocks))
    for sid in dummies:
        h.bm.allocate(sid, BLOCK)
        slots = torch.tensor(h.bm.slot_mapping(sid, 0, BLOCK))
        for layer in range(L):
            h.cache.write(layer, h.rand(BLOCK, HKV), h.rand(BLOCK, HKV), slots)
    order = torch.randperm(len(dummies), generator=h.gen).tolist()
    for i in order:
        if dummies[i] not in keep:
            h.bm.free(dummies[i])
    assert h.bm.num_free_blocks == h.bm.num_blocks - len(keep)


def is_contiguous(table: list[int]) -> bool:
    return all(b == a + 1 for a, b in zip(table, table[1:]))


@pytest.mark.parametrize("cache_dtype,atol", DTYPES)
def test_prefill_then_decode_with_free_and_admit(cache_dtype, atol):
    # 11 blocks: 2 resident dummies + exactly the 9 the three prompts need.
    h = Harness(cache_dtype, atol, num_blocks=11)
    scramble_free_list(h)

    # (a) one packed prefill of three prompts onto scrambled block tables.
    seqs, lens = [1, 2, 3], [5, 9, 13]
    h.step(seqs, lens, is_prefill=True)
    tables = {s: list(h.bm.get_block_table(s)) for s in seqs}
    assert [len(tables[s]) for s in seqs] == [2, 3, 4]
    assert not all(is_contiguous(t) for t in tables.values()), tables
    assert h.bm.num_free_blocks == 0

    # 6 decode steps; the batch order rotates so block-table rows must follow seq order.
    running = list(seqs)
    for t in range(6):
        if t == 3:
            # (b) seq 2 and the two dummies finish; seq 4 is admitted and lands on
            # seq 2's freed block ids while their stale contents are still in the cache.
            freed = set(h.bm.get_block_table(2))
            h.free(2)
            h.bm.free(103)
            h.bm.free(110)
            running.remove(2)
            h.step([4], [7], is_prefill=True)
            assert set(h.bm.get_block_table(4)) <= freed, "seq 4 should reuse freed blocks"
            running.append(4)
        order = running[t % len(running):] + running[:t % len(running)]
        h.step(order, [1] * len(order), is_prefill=False)

    assert [h.bm.get_num_tokens(s) for s in running] == [11, 19, 10]
    assert h.bm.num_free_blocks == 0  # 3 + 5 + 3 blocks: the pool is exactly full


@pytest.mark.parametrize("cache_dtype,atol", DTYPES)
def test_prefill_with_cached_prefix(cache_dtype, atol):
    """(c) first 4 tokens written in an earlier step; then prefill the remaining 6 with
    context_lens = 10 and num_cached_tokens = 4, packed with a fresh 3-token prompt."""
    h = Harness(cache_dtype, atol, num_blocks=8, seed=1)
    h.step([5], [4], is_prefill=True)
    assert h.bm.get_num_tokens(5) == 4
    meta = h.step([5, 6], [6, 3], is_prefill=True)
    assert meta.num_cached_tokens == [4, 0]
    assert meta.context_lens == [10, 3] and meta.query_lens == [6, 3]
    assert meta.slot_mapping.shape == (9,)
    # Decode still works after a cached-prefix prefill.
    h.step([6, 5], [1, 1], is_prefill=False)


def test_single_sequence_long_decode_crosses_blocks():
    h = Harness(torch.float32, 1e-5, num_blocks=6, seed=2)
    h.step([0], [3], is_prefill=True)
    for _ in range(10):  # 3 -> 13 tokens: crosses three block boundaries
        h.step([0], [1], is_prefill=False)
    assert len(h.bm.get_block_table(0)) == 4


def test_reset_clears_cache_and_free_sequence_is_noop():
    h = Harness(torch.float32, 1e-5, num_blocks=4)
    h.step([0], [5], is_prefill=True)
    h.backend.free_sequence(0)  # must not touch the cache or the block manager
    assert h.bm.has_sequence(0) and h.cache.k_cache[0].abs().sum() > 0
    h.backend.reset()
    assert all(t.abs().sum() == 0 for t in h.cache.k_cache + h.cache.v_cache)


def test_forward_requires_paged_metadata():
    h = Harness(torch.float32, 1e-5, num_blocks=4)
    meta = AttnMetadata(is_prefill=True, seq_ids=[0], query_lens=[2], context_lens=[2],
                        positions=torch.arange(2))
    q, k, v = h.rand(2, H), h.rand(2, HKV), h.rand(2, HKV)
    with pytest.raises(AssertionError):
        h.backend.forward(0, q, k, v, meta)


def test_build_helpers():
    bm = BlockManager(num_blocks=6, block_size=BLOCK)
    bm.allocate(0, 5)  # [0, 1]
    bm.allocate(1, 2)  # [2]
    bm.allocate(2, 9)  # [3, 4, 5]
    tables = build_block_tables_tensor([bm.get_block_table(s) for s in (0, 1, 2)], "cpu")
    assert tables.dtype == torch.int32 and tables.shape == (3, 3)
    assert tables.tolist() == [[0, 1, -1], [2, -1, -1], [3, 4, 5]]
    assert build_block_tables_tensor([], "cpu").shape == (0, 0)

    slots = build_slot_mapping(bm, [0, 1, 2], [0, 0, 0], [5, 2, 9], "cpu")
    assert slots.dtype == torch.int64 and slots.shape == (16,)
    assert slots.tolist() == [0, 1, 2, 3, 4, 8, 9] + list(range(12, 21))
    # Decode: one token per sequence at each sequence's current length.
    for s in (0, 1, 2):
        bm.append_slots(s, 1)
    dec = build_slot_mapping(bm, [2, 0, 1], [9, 5, 2], 1, "cpu")
    assert dec.tolist() == [21, 5, 10]
