"""PagedFlashAttentionBackend vs the dense causal reference (GPU only).

Mirrors tests/test_paged_attn.py: a BlockManager drives allocation, tables are scrambled,
sequences are freed and admitted mid-run, and a cached-prefix prefill is exercised. The
backend runs fp16 on CUDA; the reference runs fp32 from the same fp16 inputs.

Block size is 256 because upstream flash-attn requires `page_block_size % 256 == 0`
(see pagedserve/attn/paged_flash.py), so prompts are made long enough to span blocks.

Run on the pod: `python -m pytest -m gpu -q tests/test_paged_flash_gpu.py`
"""

from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from pagedserve.attn.base import AttnMetadata, causal_softmax_attention
from pagedserve.attn.paged_torch import (PagedTorchAttentionBackend, build_block_tables_tensor,
                                         build_slot_mapping)
from pagedserve.config import ModelConfig
from pagedserve.kv.block_manager import BlockManager
from pagedserve.kv.cache import PagedKVCache

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("flash_attn")

from pagedserve.attn.paged_flash import PagedFlashAttentionBackend  # noqa: E402

CFG = ModelConfig.tiny(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                       hidden_size=64)  # H=4, Hkv=2, D=16
H, HKV, D, L = 4, 2, 16, 2
BLOCK = 256
DEV = "cuda"
ATOL, RTOL = 2e-2, 1e-2


class Harness:
    def __init__(self, num_blocks: int, seed: int = 0, backend_cls=PagedFlashAttentionBackend):
        self.bm = BlockManager(num_blocks, BLOCK)
        self.cache = PagedKVCache(CFG, num_blocks, BLOCK, device=DEV, dtype=torch.float16)
        self.backend = backend_cls(CFG, self.cache)
        self.gen = torch.Generator().manual_seed(seed)
        self.hist_k: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        self.hist_v: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)

    def rand(self, n: int, heads: int) -> torch.Tensor:
        return torch.randn(n, heads, D, generator=self.gen).to(DEV, torch.float16)

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
        positions = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, query_lens)]).to(DEV)
        return AttnMetadata(
            is_prefill=is_prefill, seq_ids=seq_ids, query_lens=query_lens,
            context_lens=context_lens, positions=positions,
            slot_mapping=build_slot_mapping(self.bm, seq_ids, starts, query_lens, DEV),
            block_tables=build_block_tables_tensor(
                [self.bm.get_block_table(s) for s in seq_ids], DEV),
            block_size=BLOCK, num_cached_tokens=starts if is_prefill else [])

    def step(self, seq_ids, query_lens, is_prefill) -> AttnMetadata:
        meta = self.build_meta(seq_ids, query_lens, is_prefill)
        n_tok = sum(query_lens)
        for layer in range(L):
            q, k, v = self.rand(n_tok, H), self.rand(n_tok, HKV), self.rand(n_tok, HKV)
            out = self.backend.forward(layer, q, k, v, meta)
            assert out.shape == (n_tok, H, D) and out.dtype == torch.float16
            off = 0
            for sid, n in zip(seq_ids, query_lens):
                self.hist_k[sid, layer].append(k[off:off + n])
                self.hist_v[sid, layer].append(v[off:off + n])
                k_all = torch.cat(self.hist_k[sid, layer])
                v_all = torch.cat(self.hist_v[sid, layer])
                assert k_all.shape[0] == meta.context_lens[seq_ids.index(sid)]
                ref = causal_softmax_attention(q[off:off + n].float(), k_all.float(),
                                               v_all.float(), n)
                torch.testing.assert_close(
                    out[off:off + n].float(), ref, atol=ATOL, rtol=RTOL,
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
    dummies = list(range(100, 100 + h.bm.num_blocks))
    for sid in dummies:
        h.bm.allocate(sid, BLOCK)
        slots = torch.tensor(h.bm.slot_mapping(sid, 0, BLOCK), device=DEV)
        for layer in range(L):
            h.cache.write(layer, h.rand(BLOCK, HKV), h.rand(BLOCK, HKV), slots)
    for i in torch.randperm(len(dummies), generator=h.gen).tolist():
        if dummies[i] not in keep:
            h.bm.free(dummies[i])


def is_contiguous(table: list[int]) -> bool:
    return all(b == a + 1 for a, b in zip(table, table[1:]))


def test_prefill_then_decode_with_free_and_admit():
    h = Harness(num_blocks=11)
    scramble_free_list(h)
    seqs, lens = [1, 2, 3], [300, 600, 900]  # 2 + 3 + 4 blocks of 256
    h.step(seqs, lens, is_prefill=True)
    tables = {s: list(h.bm.get_block_table(s)) for s in seqs}
    assert [len(tables[s]) for s in seqs] == [2, 3, 4]
    assert not all(is_contiguous(t) for t in tables.values()), tables
    running = list(seqs)
    for t in range(6):
        if t == 3:
            freed = set(h.bm.get_block_table(2))
            h.free(2)
            h.bm.free(103)
            h.bm.free(110)
            running.remove(2)
            h.step([4], [700], is_prefill=True)
            assert set(h.bm.get_block_table(4)) <= freed
            running.append(4)
        order = running[t % len(running):] + running[:t % len(running)]
        h.step(order, [1] * len(order), is_prefill=False)


def test_prefill_with_cached_prefix():
    h = Harness(num_blocks=8, seed=1)
    h.step([5], [256], is_prefill=True)
    meta = h.step([5, 6], [40, 17], is_prefill=True)
    assert meta.num_cached_tokens == [256, 0]
    assert meta.context_lens == [296, 17]
    h.step([6, 5], [1, 1], is_prefill=False)


def test_decode_crosses_block_boundary():
    h = Harness(num_blocks=4, seed=2)
    h.step([0], [250], is_prefill=True)
    for _ in range(10):  # 250 -> 260 crosses the 256 boundary
        h.step([0], [1], is_prefill=False)
    assert len(h.bm.get_block_table(0)) == 2


def test_rejects_bad_block_size_and_dtype():
    with pytest.raises(RuntimeError, match="multiple of 256"):
        PagedFlashAttentionBackend(CFG, PagedKVCache(CFG, 2, 16, DEV, torch.float16))
    with pytest.raises(RuntimeError, match="fp16/bf16"):
        PagedFlashAttentionBackend(CFG, PagedKVCache(CFG, 2, 256, DEV, torch.float32))


def test_paged_torch_and_flash_agree_batch32_decode():
    hf = Harness(num_blocks=80, seed=9)
    ht = Harness(num_blocks=80, seed=9, backend_cls=PagedTorchAttentionBackend)
    seqs = list(range(32))
    lens = [int(x) for x in torch.randint(5, 500, (32,), generator=torch.Generator().manual_seed(3))]
    for h in (hf, ht):
        h.step(seqs, lens, is_prefill=True)
    for _ in range(3):
        mf, mt = hf.build_meta(seqs, [1] * 32, False), ht.build_meta(seqs, [1] * 32, False)
        assert mf.context_lens == mt.context_lens
        q, k, v = hf.rand(32, H), hf.rand(32, HKV), hf.rand(32, HKV)
        for layer in range(L):
            a = hf.backend.forward(layer, q, k, v, mf)
            b = ht.backend.forward(layer, q, k, v, mt)
            torch.testing.assert_close(a.float(), b.float(), atol=ATOL, rtol=RTOL)


def test_mixed_step_decode_rows_plus_chunk():
    """A chunked-prefill style step: many sequences with query_len 1 and a couple mid-prefill
    with query_len > 1, all attending through the cache. Exercises MixedPlan: the decode
    rows go through one batched call, the chunk rows through a small padded call."""
    h = Harness(num_blocks=64, seed=4)
    seqs = list(range(6))
    h.step(seqs, [300, 40, 20, 260, 70, 10], is_prefill=True)   # everyone gets a context
    # decode rows for 0,1,2,5 (1 token each) + chunks of 33 and 257 for 3 and 4, mixed
    meta = h.step(seqs, [1, 1, 1, 33, 257, 1], is_prefill=True)
    assert meta.mixed_plan is not None
    assert meta.mixed_plan.dec_tokens.tolist() == [0, 1, 2, 293]
    assert meta.mixed_plan.pre_rows == [(3, 33), (36, 257)] and meta.mixed_plan.pre_max_q == 257
    h.step(seqs, [1] * 6, is_prefill=False)                       # plain decode after it
    # all-prefill chunks (no decode rows) and a single decode row alone also take the path
    h.step([0, 3], [5, 7], is_prefill=True)
    h.step([1], [1], is_prefill=True)
