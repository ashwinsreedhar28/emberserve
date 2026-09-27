"""Triton decode kernel for multi-head latent attention over the paged latent cache.

The absorbed form (see attn/mla_torch.py) turns MLA decode into MQA over the latent: every
query head attends to the same 576-wide cache row `[c (512) | k_pe (64)]`, with the value
being the first 512 of that row. flash-attn cannot take a 576-wide key, so this kernel is
the GPU path for DeepSeek-V2/V3 and Moonlight.

Grid `(B, num_splits)`: one program owns one sequence and ALL of its query heads (16 for
Moonlight, padded to a power of two), so each cache row is read from HBM once per sequence.
It walks the block table one tile (= one 16-token page) at a time:

    scores = q_c . c^T + q_pe . k_pe^T           two tl.dot, K = 512 and 64, fp32 accumulate
    online softmax over the tile (running m, l)
    acc   += P . c                              [H, 512] fp32

`num_splits > 1` is flash-decoding: contiguous tile ranges go to separate programs and
`paged_triton._reduce_kernel` merges the partial (m, l, acc). Prefill uses either
flash-attn's varlen kernel in the non-absorbed form (fresh prompts) or the torch reference
per sequence (chunks with cached context); see `MLATritonBackend`.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from pagedserve.attn.base import AttnMetadata
from pagedserve.attn.mla_torch import mla_attention_absorbed
from pagedserve.attn.paged_flash import block_tables_nonneg, context_lens_tensor
from pagedserve.attn.paged_triton import _SCRATCH, _next_pow2, default_num_splits
from pagedserve.attn.paged_triton import _kernels as _paged_kernels
from pagedserve.config import ModelConfig
from pagedserve.kv.cache import PagedLatentCache

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _mla_decode_kernel(
        q_ptr, lat_ptr, out_ptr, bt_ptr, ctx_ptr,
        m_part_ptr, l_part_ptr, acc_part_ptr,
        scale,
        stride_qb, stride_qh,
        stride_lb, stride_ls,
        stride_ob, stride_oh,
        stride_bt,
        tiles_per_split,
        H: tl.constexpr, H_PAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr,
        DL: tl.constexpr, DR: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        b = tl.program_id(0)
        split = tl.program_id(1)
        TILES_PER_BLOCK: tl.constexpr = BLOCK_SIZE // TILE

        ctx = tl.load(ctx_ptr + b)
        h = tl.arange(0, H_PAD)
        dl = tl.arange(0, DL)
        dr = tl.arange(0, DR)
        s = tl.arange(0, TILE)
        h_valid = h < H

        q_row = q_ptr + b * stride_qb + h[:, None] * stride_qh
        q_c = tl.load(q_row + dl[None, :], mask=h_valid[:, None], other=0.0)  # [H, DL]
        q_pe = tl.load(q_row + DL + dr[None, :], mask=h_valid[:, None], other=0.0)  # [H, DR]

        m_i = tl.full([H_PAD], float("-inf"), tl.float32)
        l_i = tl.zeros([H_PAD], tl.float32)
        acc = tl.zeros([H_PAD, DL], tl.float32)

        num_tiles = tl.cdiv(ctx, TILE)
        tile_start = split * tiles_per_split
        tile_end = tl.minimum(tile_start + tiles_per_split, num_tiles)

        for t in range(tile_start, tile_end):
            blk = t // TILES_PER_BLOCK
            phys = tl.load(bt_ptr + b * stride_bt + blk).to(tl.int64)
            pos = t * TILE + s
            in_blk = (t % TILES_PER_BLOCK) * TILE + s
            valid = pos < ctx
            row = lat_ptr + phys * stride_lb + in_blk[:, None] * stride_ls
            c = tl.load(row + dl[None, :], mask=valid[:, None], other=0.0)  # [T, DL]
            kpe = tl.load(row + DL + dr[None, :], mask=valid[:, None], other=0.0)  # [T, DR]
            scores = tl.dot(q_c, tl.trans(c.to(q_c.dtype))) + tl.dot(q_pe, tl.trans(kpe.to(q_pe.dtype)))
            scores = scores * scale  # [H, T] fp32
            scores = tl.where(valid[None, :], scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(c.dtype), c)  # [H, DL]
            m_i = m_new

        if SPLIT_K:
            row_id = (split * tl.num_programs(0) + b) * H + h  # partial layout [S, B, H]
            tl.store(m_part_ptr + row_id, m_i, mask=h_valid)
            tl.store(l_part_ptr + row_id, l_i, mask=h_valid)
            tl.store(acc_part_ptr + row_id[:, None] * DL + dl[None, :], acc, mask=h_valid[:, None])
        else:
            out = acc / l_i[:, None]
            o_off = b * stride_ob + h[:, None] * stride_oh + dl[None, :]
            tl.store(out_ptr + o_off, out.to(out_ptr.dtype.element_ty), mask=h_valid[:, None])

    _KERNEL = _mla_decode_kernel
    return _KERNEL


def mla_decode(q_abs: Tensor, latent_cache: Tensor, block_tables: Tensor, context_lens: Tensor,
               scale: float, kv_lora_rank: int, *, num_splits: int | None = None,
               out: Tensor | None = None, num_warps: int = 8) -> Tensor:
    """`q_abs [B, H, DL + DR]` (absorbed query: [q_c | q_pe]); `latent_cache [num_blocks,
    block_size, DL + DR]`; `block_tables [B, max_blocks] int32` (non-negative padding);
    `context_lens [B] int32`. Returns `out_c [B, H, DL]` = softmax(scores) . c in q's dtype."""
    B, H, D = q_abs.shape
    num_blocks, block_size, Dl = latent_cache.shape
    DL = kv_lora_rank
    DR = D - DL
    assert Dl == D, (latent_cache.shape, q_abs.shape)
    assert DL & (DL - 1) == 0 and DR & (DR - 1) == 0 and DR >= 16, (DL, DR)
    assert q_abs.dtype == latent_cache.dtype and q_abs.dtype in SUPPORTED_DTYPES, (q_abs.dtype, latent_cache.dtype)
    assert q_abs.stride(2) == 1 and latent_cache.stride(2) == 1
    assert block_tables.dtype == torch.int32 and context_lens.dtype == torch.int32
    assert block_tables.shape[0] == B and context_lens.shape == (B,)
    tile = min(block_size, 16)
    assert block_size % tile == 0
    h_pad = max(16, _next_pow2(H))
    max_blocks = block_tables.shape[1]
    max_context = max_blocks * block_size
    num_tiles_max = max(1, -(-max_context // tile))
    if num_splits is None:
        num_splits = default_num_splits(B, 1, max_context, q_abs.device)
    num_splits = max(1, min(int(num_splits), num_tiles_max))
    tiles_per_split = -(-num_tiles_max // num_splits)
    if out is None:
        out = torch.empty((B, H, DL), dtype=q_abs.dtype, device=q_abs.device)
    assert out.shape == (B, H, DL) and out.stride(2) == 1
    if num_splits == 1:
        m_p = l_p = acc_p = out
    else:
        m_p, l_p, acc_p = _SCRATCH.get(num_splits, B, H, DL, q_abs.device)
    _kernel()[(B, num_splits)](
        q_abs, latent_cache, out, block_tables, context_lens,
        m_p, l_p, acc_p,
        float(scale),
        q_abs.stride(0), q_abs.stride(1),
        latent_cache.stride(0), latent_cache.stride(1),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        tiles_per_split,
        H=H, H_PAD=h_pad, BLOCK_SIZE=block_size, TILE=tile, DL=DL, DR=DR,
        SPLIT_K=num_splits > 1, num_warps=num_warps,
    )
    if num_splits > 1:
        _paged_kernels()["reduce"][(B, H)](
            m_p, l_p, acc_p, out,
            out.stride(0), out.stride(1),
            num_splits,
            NUM_SPLITS_PAD=_next_pow2(num_splits), D=DL,
            num_warps=1,
        )
    return out


class MLATritonBackend:
    """Latent attention on CUDA: the Triton kernel for decode rows, flash-attn varlen (in
    the non-absorbed form, v zero-padded to the qk head dim) for fresh-prompt prefill, and
    the torch absorbed reference for rows that attend through the cache with query_len > 1
    (chunked-prefill chunks, cached-prefix prompts)."""

    name = "mla_triton"

    def __init__(self, config: ModelConfig, cache: PagedLatentCache) -> None:
        assert config.mla is not None
        if cache.device.type != "cuda" and os.environ.get("TRITON_INTERPRET", "0") != "1":
            raise RuntimeError("mla_triton needs a CUDA cache (or TRITON_INTERPRET=1)")
        self.config = config
        self.cache = cache
        self.mla = config.mla
        self.kv_lora_rank = config.mla.kv_lora_rank
        self._varlen = None
        if cache.device.type == "cuda":
            try:
                from flash_attn import flash_attn_varlen_func

                self._varlen = flash_attn_varlen_func
            except ImportError:
                self._varlen = None

    # ---- entry point -------------------------------------------------------------------
    def forward(self, layer_idx: int, q_nope: Tensor, q_pe: Tensor, latent: Tensor,
                w_uk: Tensor, w_uv: Tensor, scale: float, meta: AttnMetadata,
                kv_b_weight: Tensor | None = None) -> Tensor:
        assert meta.slot_mapping is not None and meta.block_tables is not None
        self.cache.write(layer_idx, latent, meta.slot_mapping)
        if all(n == 1 for n in meta.query_lens):
            return self._decode(layer_idx, q_nope, q_pe, w_uk, w_uv, scale, meta)
        fresh = not meta.num_cached_tokens or all(c == 0 for c in meta.num_cached_tokens)
        if fresh and self._varlen is not None and kv_b_weight is not None:
            return self._prefill_varlen(q_nope, q_pe, latent, kv_b_weight, scale, meta)
        return self._prefill_reference(layer_idx, q_nope, q_pe, w_uk, w_uv, scale, meta)

    # ---- decode: absorb W_UK into q, kernel over the latent, W_UV after -----------------
    def _absorb(self, q_nope: Tensor, q_pe: Tensor, w_uk: Tensor) -> Tensor:
        # q_c[n, h, :] = w_uk[h] @ q_nope[n, h, :]  ->  bmm over heads: [H, N, Dn] x [H, Dn, Dl]
        q_c = torch.bmm(q_nope.transpose(0, 1), w_uk).transpose(0, 1)  # [N, H, Dl]
        return torch.cat([q_c, q_pe], dim=-1).contiguous()  # [N, H, Dl + Dr]

    def _decode(self, layer_idx: int, q_nope: Tensor, q_pe: Tensor, w_uk: Tensor, w_uv: Tensor,
                scale: float, meta: AttnMetadata) -> Tensor:
        q_abs = self._absorb(q_nope, q_pe, w_uk)
        out_c = mla_decode(q_abs, self.cache.latent[layer_idx], block_tables_nonneg(meta),
                           context_lens_tensor(meta, self.cache.device), scale, self.kv_lora_rank)
        # out[n, h, :] = w_uv[h] @ out_c[n, h, :]  ->  [H, N, Dl] x [H, Dl, Dv]
        return torch.bmm(out_c.transpose(0, 1), w_uv.transpose(1, 2)).transpose(0, 1)

    # ---- fresh-prompt prefill: materialize k/v per head, flash varlen -------------------
    def _prefill_varlen(self, q_nope: Tensor, q_pe: Tensor, latent: Tensor, kv_b_weight: Tensor,
                        scale: float, meta: AttnMetadata) -> Tensor:
        n, h, dn = q_nope.shape
        dr, dv = self.mla.qk_rope_head_dim, self.mla.v_head_dim
        c = latent[:, :self.kv_lora_rank]
        k_pe = latent[:, self.kv_lora_rank:]
        kv = torch.nn.functional.linear(c, kv_b_weight).view(n, h, dn + dv)
        k = torch.cat([kv[..., :dn], k_pe[:, None, :].expand(n, h, dr)], dim=-1)  # [N, H, Dn + Dr]
        q = torch.cat([q_nope, q_pe], dim=-1)
        v = kv[..., dn:]
        v_pad = torch.nn.functional.pad(v, (0, dn + dr - dv)) if dv != dn + dr else v
        cu = meta.cu_seqlens_q
        max_q = max(meta.query_lens)
        out = self._varlen(q, k, v_pad, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max_q,
                           max_seqlen_k=max_q, softmax_scale=scale, causal=True)
        return out[..., :dv]

    # ---- rows attending through the cache with query_len > 1: torch reference -----------
    def _prefill_reference(self, layer_idx: int, q_nope: Tensor, q_pe: Tensor, w_uk: Tensor,
                           w_uv: Tensor, scale: float, meta: AttnMetadata) -> Tensor:
        outs = []
        start = 0
        for i, q_len in enumerate(meta.query_lens):
            ctx = self.cache.gather(layer_idx, meta.block_tables[i], meta.context_lens[i])
            outs.append(mla_attention_absorbed(q_nope[start:start + q_len], q_pe[start:start + q_len],
                                               ctx, w_uk, w_uv, scale, self.kv_lora_rank))
            start += q_len
        return torch.cat(outs, dim=0)

    def free_sequence(self, seq_id: int) -> None:
        pass

    def reset(self) -> None:
        pass
