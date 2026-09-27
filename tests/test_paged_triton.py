"""Triton paged-decode kernel + backend on the CPU via Triton's interpreter.

`TRITON_INTERPRET=1` must be set BEFORE triton is imported, so this module sets it at the
top. The interpreter executes the kernel program-by-program with numpy, which is slow but
exact, so shapes are kept tiny (a few sequences, contexts < 64) to stay well under 30 s.

Mirrors tests/test_paged_attn.py: a real BlockManager hands out scrambled block tables,
the backend is driven like the engine, and every output is checked against
`causal_softmax_attention` over the sequence's full history.
"""

from __future__ import annotations

import os

os.environ["TRITON_INTERPRET"] = "1"  # noqa: E402  (must precede the triton import)

from collections import defaultdict  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton")

from pagedserve.attn.base import AttnMetadata, causal_softmax_attention  # noqa: E402
from pagedserve.attn.paged_torch import (  # noqa: E402
    PagedTorchAttentionBackend, build_block_tables_tensor, build_slot_mapping)
from pagedserve.attn.paged_triton import (  # noqa: E402
    PagedTritonAttentionBackend, default_num_splits, is_available, paged_attention_decode)
from pagedserve.config import ModelConfig  # noqa: E402
from pagedserve.kv.block_manager import BlockManager  # noqa: E402
from pagedserve.kv.cache import PagedKVCache  # noqa: E402

H, HKV, D, L = 4, 2, 64, 2
CFG = ModelConfig.tiny(num_hidden_layers=L, num_attention_heads=H, num_key_value_heads=HKV,
                       hidden_size=H * D)
BLOCK = 16
ATOL = 1e-4
DEV = "cpu"


class Harness:
    """Drives the backend like the engine and checks every output against the reference."""

    def __init__(self, num_blocks: int, seed: int = 0, dtype=torch.float32,
                 backend_cls=PagedTritonAttentionBackend, **backend_kw):
        self.bm = BlockManager(num_blocks, BLOCK)
        self.cache = PagedKVCache(CFG, num_blocks, BLOCK, device=DEV, dtype=dtype)
        self.backend = backend_cls(CFG, self.cache, **backend_kw)
        self.gen = torch.Generator().manual_seed(seed)
        self.hist_k: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        self.hist_v: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)

    def rand(self, n: int, heads: int) -> torch.Tensor:
        return torch.randn(n, heads, D, generator=self.gen)

    def build_meta(self, seq_ids, query_lens, is_prefill) -> AttnMetadata:
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
        return AttnMetadata(
            is_prefill=is_prefill, seq_ids=seq_ids, query_lens=query_lens,
            context_lens=context_lens, positions=positions,
            slot_mapping=build_slot_mapping(self.bm, seq_ids, starts, query_lens, DEV),
            block_tables=build_block_tables_tensor(
                [self.bm.get_block_table(s) for s in seq_ids], DEV),
            block_size=BLOCK, num_cached_tokens=starts if is_prefill else [])

    def step(self, seq_ids, query_lens, is_prefill, atol=ATOL) -> AttnMetadata:
        meta = self.build_meta(seq_ids, query_lens, is_prefill)
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
                assert k_all.shape[0] == meta.context_lens[seq_ids.index(sid)]
                ref = causal_softmax_attention(q[off:off + n], k_all, v_all, n)
                torch.testing.assert_close(
                    out[off:off + n], ref, atol=atol, rtol=0,
                    msg=lambda m, s=sid, lyr=layer: f"seq {s} layer {lyr}: {m}")
                off += n
        return meta

    def free(self, seq_id: int) -> None:
        self.bm.free(seq_id)
        self.backend.free_sequence(seq_id)
        for layer in range(L):
            self.hist_k.pop((seq_id, layer), None)
            self.hist_v.pop((seq_id, layer), None)


def scramble_free_list(h: Harness, keep=(103, 110)) -> None:
    """Fill every block with a one-block dummy, then free them in a shuffled order so
    later allocations get non-contiguous tables. `keep` stay resident with live data."""
    dummies = list(range(100, 100 + h.bm.num_blocks))
    for sid in dummies:
        h.bm.allocate(sid, BLOCK)
        slots = torch.tensor(h.bm.slot_mapping(sid, 0, BLOCK))
        for layer in range(L):
            h.cache.write(layer, h.rand(BLOCK, HKV), h.rand(BLOCK, HKV), slots)
    for i in torch.randperm(len(dummies), generator=h.gen).tolist():
        if dummies[i] not in keep:
            h.bm.free(dummies[i])


def is_contiguous(table: list[int]) -> bool:
    return all(b == a + 1 for a, b in zip(table, table[1:]))


# ---- kernel-level -----------------------------------------------------------------------
def test_interpreter_available():
    assert is_available()


def test_kernel_matches_reference_ragged_scrambled():
    """B=3 ragged contexts (5, 19, 40), block 16, GQA 4q/2kv, D=64, fp32, real tables."""
    h = Harness(num_blocks=8, seed=0)
    scramble_free_list(h)
    seqs, lens = [1, 2, 3], [5, 19, 40]  # 1 + 2 + 3 blocks: exactly the 6 free blocks
    meta = h.build_meta(seqs, lens, is_prefill=True)
    tables = {s: list(h.bm.get_block_table(s)) for s in seqs}
    assert [len(tables[s]) for s in seqs] == [1, 2, 3]
    assert not all(is_contiguous(t) for t in tables.values()), tables
    n_tok = sum(lens)
    k, v = h.rand(n_tok, HKV), h.rand(n_tok, HKV)
    h.cache.write(0, k, v, meta.slot_mapping)

    # Decode query = the LAST token of each sequence (the one just written), per
    # base.py semantics: context_lens counts it.
    q = h.rand(len(seqs), H)
    bt = meta.block_tables.clamp_min(0).contiguous()
    ctx = torch.tensor(meta.context_lens, dtype=torch.int32)
    out = paged_attention_decode(q, h.cache.k_cache[0], h.cache.v_cache[0], bt, ctx,
                                 D ** -0.5, num_splits=1)
    assert out.shape == (3, H, D) and out.dtype == torch.float32
    off = 0
    for i, n in enumerate(lens):
        ref = causal_softmax_attention(q[i:i + 1], k[off:off + n], v[off:off + n], 1)
        torch.testing.assert_close(out[i:i + 1], ref, atol=ATOL, rtol=0)
        off += n


def test_kernel_split_k_equals_single_pass():
    h = Harness(num_blocks=8, seed=4)
    scramble_free_list(h)
    seqs, lens = [1, 2, 3], [5, 19, 40]
    meta = h.build_meta(seqs, lens, is_prefill=True)
    k, v = h.rand(sum(lens), HKV), h.rand(sum(lens), HKV)
    h.cache.write(0, k, v, meta.slot_mapping)
    q = h.rand(3, H)
    bt = meta.block_tables.clamp_min(0).contiguous()
    ctx = torch.tensor(meta.context_lens, dtype=torch.int32)
    args = (q, h.cache.k_cache[0], h.cache.v_cache[0], bt, ctx, D ** -0.5)
    single = paged_attention_decode(*args, num_splits=1)
    for splits in (2, 3, 5):  # 5 > the 3 tiles-per-block; some splits see no tiles
        split = paged_attention_decode(*args, num_splits=splits)
        torch.testing.assert_close(split, single, atol=1e-5, rtol=0)
    # The heuristic never depends on tensor values (CUDA-graph safety): shapes only.
    assert default_num_splits(1, HKV, 48, torch.device("cpu")) >= 1


def test_kernel_fp16_cache_and_padded_groups():
    """fp16 cache computed in fp32; GQA groups=3 (non power of two -> padded to 4)."""
    Hq, Hk = 6, 2
    kc = torch.randn(4, BLOCK, Hk, D).half()
    vc = torch.randn(4, BLOCK, Hk, D).half()
    q = torch.randn(2, Hq, D).half()
    bt = torch.tensor([[3, 1, 0], [2, 0, 0]], dtype=torch.int32)
    ctx = torch.tensor([30, 16], dtype=torch.int32)
    out = paged_attention_decode(q, kc, vc, bt, ctx, D ** -0.5, num_splits=1)
    assert out.dtype == torch.float16
    for b in range(2):
        n = int(ctx[b])
        nb = -(-n // BLOCK)
        k = kc[bt[b, :nb].long()].reshape(-1, Hk, D)[:n]
        v = vc[bt[b, :nb].long()].reshape(-1, Hk, D)[:n]
        ref = causal_softmax_attention(q[b:b + 1].float(), k.float(), v.float(), 1)
        torch.testing.assert_close(out[b:b + 1].float(), ref, atol=2e-3, rtol=0)


def test_kernel_rejects_bad_shapes():
    kc = torch.randn(2, 8, 1, D)  # block_size 8 is not a multiple of 16
    with pytest.raises(AssertionError, match="block_size"):
        paged_attention_decode(torch.randn(1, 2, D), kc, kc, torch.zeros(1, 1, dtype=torch.int32),
                               torch.ones(1, dtype=torch.int32), 1.0)


# ---- backend-level ----------------------------------------------------------------------
def test_backend_prefill_then_decode_with_free_and_admit():
    """Prefill (delegated to paged_torch here), then scrambled-table decode, then a
    mid-run free + admit that lands the new sequence on stale blocks."""
    h = Harness(num_blocks=11)
    assert h.backend.prefill_backend_name == "PagedTorchAttentionBackend"
    scramble_free_list(h)
    seqs, lens = [1, 2, 3], [17, 33, 40]  # 2 + 3 + 3 blocks
    h.step(seqs, lens, is_prefill=True)
    tables = {s: list(h.bm.get_block_table(s)) for s in seqs}
    assert not all(is_contiguous(t) for t in tables.values()), tables
    running = list(seqs)
    for t in range(5):
        if t == 2:
            freed = set(h.bm.get_block_table(2))
            h.free(2)
            running.remove(2)
            h.step([4], [20], is_prefill=True)
            assert set(h.bm.get_block_table(4)) & freed, "seq 4 should reuse freed blocks"
            running.append(4)
            h.bm.free(103)
            h.bm.free(110)
        order = running[t % len(running):] + running[:t % len(running)]
        h.step(order, [1] * len(order), is_prefill=False)
    assert [h.bm.get_num_tokens(s) for s in running] == [22, 45, 23]


def test_backend_decode_crosses_block_boundary_and_matches_paged_torch():
    ht = Harness(num_blocks=6, seed=2)
    hr = Harness(num_blocks=6, seed=2, backend_cls=PagedTorchAttentionBackend)
    for h in (ht, hr):
        h.step([0], [14], is_prefill=True)
    for _ in range(4):  # 14 -> 18 crosses the 16 boundary
        mt, mr = ht.build_meta([0], [1], False), hr.build_meta([0], [1], False)
        q, k, v = ht.rand(1, H), ht.rand(1, HKV), ht.rand(1, HKV)
        for layer in range(L):
            a = ht.backend.forward(layer, q, k, v, mt)
            b = hr.backend.forward(layer, q, k, v, mr)
            torch.testing.assert_close(a, b, atol=ATOL, rtol=0)
    assert len(ht.bm.get_block_table(0)) == 2


def test_backend_uses_precomputed_meta_tensors():
    """The CUDA-graph path hands the backend context_lens_t / block_tables_nonneg; the
    Python lists must then be ignored (they hold stale bucket values during replay)."""
    h = Harness(num_blocks=6, seed=5)
    h.step([0], [20], is_prefill=True)
    meta = h.build_meta([0], [1], False)
    meta.context_lens_t = torch.tensor(meta.context_lens, dtype=torch.int32)
    meta.block_tables_nonneg = meta.block_tables.clamp_min(0).contiguous()
    meta.context_lens = [999]  # would index out of range if used
    q, k, v = h.rand(1, H), h.rand(1, HKV), h.rand(1, HKV)
    out = h.backend.forward(0, q, k, v, meta)
    h.hist_k[0, 0].append(k)
    h.hist_v[0, 0].append(v)
    ref = causal_softmax_attention(q, torch.cat(h.hist_k[0, 0]), torch.cat(h.hist_v[0, 0]), 1)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=0)


def test_backend_split_k_option_matches_default():
    ha = Harness(num_blocks=6, seed=7)
    hb = Harness(num_blocks=6, seed=7, num_splits=3)
    for h in (ha, hb):
        h.step([0, 1], [40, 9], is_prefill=True)
    ma, mb = ha.build_meta([0, 1], [1, 1], False), hb.build_meta([0, 1], [1, 1], False)
    q, k, v = ha.rand(2, H), ha.rand(2, HKV), ha.rand(2, HKV)
    a = ha.backend.forward(0, q, k, v, ma)
    b = hb.backend.forward(0, q, k, v, mb)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


def test_backend_rejects_bad_block_size_and_reset():
    with pytest.raises(RuntimeError, match="block_size % 16"):
        PagedTritonAttentionBackend(CFG, PagedKVCache(CFG, 2, 8, DEV, torch.float32))
    h = Harness(num_blocks=4)
    h.step([0], [5], is_prefill=True)
    h.backend.free_sequence(0)
    assert h.bm.has_sequence(0) and h.cache.k_cache[0].abs().sum() > 0
    h.backend.reset()
    assert all(t.abs().sum() == 0 for t in h.cache.k_cache + h.cache.v_cache)
