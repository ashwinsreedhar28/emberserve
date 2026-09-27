"""Triton paged-decode kernel on a real GPU vs paged_torch and paged_flash (GPU only).

Kernel level: one decode step at the REAL geometry (H=14, Hkv=2 -> 7 groups, D=64, fp16)
over ragged contexts up to 2048 with scrambled block tables, at B in {1, 8, 32, 128} and
block sizes 16 and 256; checked against `PagedTorchAttentionBackend` (fp32 gather) and,
at block 256, against `PagedFlashAttentionBackend`. Both single-pass and split-K.

Engine level: 8 prompts, greedy, identical tokens across paged_torch / paged_flash /
paged_triton / paged_triton+cuda graphs on a tiny random model (mirrors
tests/test_cuda_graphs_gpu.py). paged_triton runs at block_size 16 — the point of the
kernel — while paged_flash stays at 256.

Run on the pod: `python -m pytest -m gpu -q -k triton`
"""

from __future__ import annotations

import pytest
import torch

from pagedserve.attn.base import AttnMetadata
from pagedserve.attn.paged_torch import (PagedTorchAttentionBackend, build_block_tables_tensor,
                                         build_slot_mapping)
from pagedserve.config import EngineConfig, ModelConfig
from pagedserve.engine import LLMEngine
from pagedserve.kv.block_manager import BlockManager
from pagedserve.kv.cache import PagedKVCache
from pagedserve.llm import LLM
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from pagedserve.sched.request import SamplingParams

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from pagedserve.attn import paged_flash  # noqa: E402
from pagedserve.attn.paged_triton import (  # noqa: E402
    PagedTritonAttentionBackend, paged_attention_decode)

DEV = "cuda"
H, HKV, D = 14, 2, 64  # Qwen2.5-0.5B attention geometry
CFG = ModelConfig.tiny(num_hidden_layers=1, num_attention_heads=H, num_key_value_heads=HKV,
                       hidden_size=H * D)
MAX_CTX = 2048
ATOL, RTOL = 2e-2, 1e-2


# ---- kernel level ----------------------------------------------------------------------
def _decode_case(batch: int, block: int, seed: int):
    """A decode step at `batch` with ragged contexts in [1, MAX_CTX] on scrambled tables.

    Returns (cache, meta, q). The cache is filled with random K/V for every block, so
    stale blocks hold real data and a wrong block-table read shows up as a mismatch.
    """
    gen = torch.Generator().manual_seed(seed)
    ctx = torch.randint(1, MAX_CTX + 1, (batch,), generator=gen).tolist()
    if batch >= 8:  # pin the extremes so boundaries are always covered
        ctx[0], ctx[1], ctx[2] = 1, MAX_CTX, block
    blocks_needed = sum(-(-c // block) for c in ctx)
    num_blocks = blocks_needed + 8
    bm = BlockManager(num_blocks, block)
    cache = PagedKVCache(CFG, num_blocks, block, device=DEV, dtype=torch.float16)
    for layer in range(CFG.num_hidden_layers):
        cache.k_cache[layer].copy_(torch.randn(cache.k_cache[layer].shape, generator=gen).half())
        cache.v_cache[layer].copy_(torch.randn(cache.v_cache[layer].shape, generator=gen).half())
    # Scramble the free list so tables are non-contiguous.
    for sid in range(num_blocks):
        bm.allocate(1000 + sid, block)
    for sid in torch.randperm(num_blocks, generator=gen).tolist():
        bm.free(1000 + sid)
    seqs = list(range(batch))
    for sid, c in zip(seqs, ctx):
        bm.allocate(sid, c)
    starts = [c - 1 for c in ctx]
    meta = AttnMetadata(
        is_prefill=False, seq_ids=seqs, query_lens=[1] * batch, context_lens=ctx,
        positions=torch.tensor(starts, device=DEV),
        slot_mapping=build_slot_mapping(bm, seqs, starts, 1, DEV),
        block_tables=build_block_tables_tensor([bm.get_block_table(s) for s in seqs], DEV),
        block_size=block)
    q = torch.randn(batch, H, D, generator=gen).to(DEV, torch.float16)
    k = torch.randn(batch, HKV, D, generator=gen).to(DEV, torch.float16)
    v = torch.randn(batch, HKV, D, generator=gen).to(DEV, torch.float16)
    return cache, meta, q, k, v


@pytest.mark.parametrize("block", [16, 256])
@pytest.mark.parametrize("batch", [1, 8, 32, 128])
def test_kernel_matches_paged_torch(batch, block):
    cache, meta, q, k, v = _decode_case(batch, block, seed=batch * 7 + block)
    torch_backend = PagedTorchAttentionBackend(CFG, cache)
    ref = torch_backend.forward(0, q, k, v, meta)  # writes k/v, fp32 math, fp16 out
    bt = paged_flash.block_tables_nonneg(meta)
    ctx = paged_flash.context_lens_tensor(meta, cache.device)
    for splits in (1, 4, None):
        out = paged_attention_decode(q, cache.k_cache[0], cache.v_cache[0], bt, ctx,
                                     D ** -0.5, num_splits=splits)
        torch.cuda.synchronize()
        assert out.shape == (batch, H, D) and out.dtype == torch.float16
        torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL,
                                   msg=lambda m, s=splits: f"num_splits={s}: {m}")


@pytest.mark.parametrize("batch", [1, 8, 32, 128])
def test_kernel_matches_paged_flash_block256(batch):
    pytest.importorskip("flash_attn")
    block = 256
    cache, meta, q, k, v = _decode_case(batch, block, seed=batch * 13)
    flash_backend = paged_flash.PagedFlashAttentionBackend(CFG, cache)
    ref = flash_backend.forward(0, q, k, v, meta)
    out = paged_attention_decode(q, cache.k_cache[0], cache.v_cache[0],
                                 paged_flash.block_tables_nonneg(meta),
                                 paged_flash.context_lens_tensor(meta, cache.device),
                                 D ** -0.5)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_backend_block16_matches_paged_torch_over_steps():
    """Backend-level: prefill (delegated) then 40 decode steps crossing block boundaries."""
    block = 16
    gen = torch.Generator().manual_seed(21)
    lens = [1, 15, 16, 17, 300, 777, 1024, 2000]
    num_blocks = sum(-(-(n + 40) // block) for n in lens) + 4
    caches = [PagedKVCache(CFG, num_blocks, block, device=DEV, dtype=torch.float16)
              for _ in range(2)]
    bm = BlockManager(num_blocks, block)
    tri = PagedTritonAttentionBackend(CFG, caches[0])
    ref = PagedTorchAttentionBackend(CFG, caches[1])
    seqs = list(range(len(lens)))
    for sid, n in zip(seqs, lens):
        bm.allocate(sid, n)

    def meta_for(qlens, starts, is_prefill):
        return AttnMetadata(
            is_prefill=is_prefill, seq_ids=seqs, query_lens=qlens,
            context_lens=[bm.get_num_tokens(s) for s in seqs],
            positions=torch.cat([torch.arange(s, s + n) for s, n in zip(starts, qlens)]).to(DEV),
            slot_mapping=build_slot_mapping(bm, seqs, starts, qlens, DEV),
            block_tables=build_block_tables_tensor([bm.get_block_table(s) for s in seqs], DEV),
            block_size=block, num_cached_tokens=starts if is_prefill else [])

    def rand(n, heads):
        return torch.randn(n, heads, D, generator=gen).to(DEV, torch.float16)

    meta = meta_for(lens, [0] * len(seqs), True)
    n_tok = sum(lens)
    q, k, v = rand(n_tok, H), rand(n_tok, HKV), rand(n_tok, HKV)
    a, b = tri.forward(0, q, k, v, meta), ref.forward(0, q, k, v, meta)
    torch.testing.assert_close(a.float(), b.float(), atol=ATOL, rtol=RTOL)
    for _ in range(40):
        starts = [bm.get_num_tokens(s) for s in seqs]
        for sid in seqs:
            bm.append_slots(sid, 1)
        meta = meta_for([1] * len(seqs), starts, False)
        q, k, v = rand(len(seqs), H), rand(len(seqs), HKV), rand(len(seqs), HKV)
        a, b = tri.forward(0, q, k, v, meta), ref.forward(0, q, k, v, meta)
        torch.testing.assert_close(a.float(), b.float(), atol=ATOL, rtol=RTOL)


# ---- engine level ----------------------------------------------------------------------
# ModelConfig.tiny() has D=16, which the kernel does not support: widen to D=64.
ENGINE_CFG = ModelConfig.tiny(num_attention_heads=4, num_key_value_heads=2, hidden_size=256)


def make_engine(backend: str, block: int, graphs: bool, num_blocks: int = 64,
                max_num_seqs: int = 64, max_model_len: int = 512) -> LLMEngine:
    model = Qwen2ForCausalLM(ENGINE_CFG)
    reset_parameters_deterministic(model, 0)
    model = model.to(DEV, torch.float16)
    ecfg = EngineConfig(device=DEV, dtype=torch.float16, block_size=block,
                        num_gpu_blocks=num_blocks, max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=4096, max_model_len=max_model_len,
                        attn_backend=backend, enable_cuda_graphs=graphs)
    return LLMEngine(model, ENGINE_CFG, ecfg, tokenizer=None)


def prompts(n: int, seed: int = 1) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, ENGINE_CFG.vocab_size,
                          (int(torch.randint(3, 40, (1,), generator=g)),), generator=g).tolist()
            for _ in range(n)]


def gen(eng: LLMEngine, ps, max_tokens=16):
    return [r.output_token_ids for r in
            LLM.from_engine(eng).generate(ps, SamplingParams.greedy(max_tokens, ignore_eos=True))]


def test_engine_greedy_identical_across_backends():
    ps = prompts(8)
    ref = gen(make_engine("paged_torch", 16, False), ps)
    tri = make_engine("paged_triton", 16, False)
    assert tri.graph_runner is None and tri.block_manager.num_blocks == 64
    assert gen(tri, ps) == ref
    tri_g = make_engine("paged_triton", 16, True, num_blocks=64)
    assert tri_g.graph_runner is not None and tri_g.scratch_block == 63
    assert gen(tri_g, ps) == ref
    if paged_flash.is_available():
        assert gen(make_engine("paged_flash", 256, False, num_blocks=16), ps) == ref
        assert tri.backend.prefill_backend_name == "PagedTorchAttentionBackend"  # block 16


@pytest.mark.parametrize("n", [5, 13])
def test_engine_graphs_batch_between_buckets(n):
    ps = prompts(n, seed=n)
    assert gen(make_engine("paged_triton", 16, True), ps) == \
        gen(make_engine("paged_triton", 16, False), ps)


@pytest.mark.skipif(not paged_flash.is_available(), reason="needs flash_attn")
def test_engine_triton_block256_delegates_prefill_to_flash():
    ps = prompts(8, seed=5)
    tri = make_engine("paged_triton", 256, True, num_blocks=16)
    assert tri.backend.prefill_backend_name == "PagedFlashAttentionBackend"
    assert gen(tri, ps) == gen(make_engine("paged_flash", 256, True, num_blocks=16), ps)
