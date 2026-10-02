"""Fused mixture-of-experts forward in Triton (the grouped GEMM behind DeepseekMoE on CUDA).

The per-expert Python loop in `moe.py` is the reference: correct, but on Moonlight it cost a
host sync plus ~6 kernel launches per active expert per layer (a 64 ms decode step at
batch 1 against a ~4 ms weight-read floor). The fused path never asks the host how many
tokens each expert got:

  1. `moe_align`: sort the (token, choice) assignments by expert on the GPU and lay them
     out so every BLOCK_M-row block belongs to one expert (each expert's run is padded up
     to a multiple of BLOCK_M). The layout size is bounded by `N*k + E*(BLOCK_M-1)`, so the
     grid is fixed from shapes; blocks past the real end carry expert id E and exit.
  2. `_grouped_gemm_kernel` over that layout: program (block, n-tile) gathers its block's
     activation rows, multiplies by ITS expert's weight tile, and scatters the rows back.
     Run once for gate/up (rows gathered by token), `silu_and_mul`, then once for down
     (rows are the intermediate, one per assignment) with the routing weight folded in.
  3. Sum the k assignments per token.

Same structure as vLLM's `fused_moe` kernel. Block alignment is one Triton kernel
(`_moe_align_kernel`, program per expert: count, prefix-sum, rank, scatter) for the sizes a
decode step produces; the torch-op version stays as the reference and the large-N path.
`_topk_gate_kernel` does the router's post-linear work (sigmoid, bias, top-k, gather,
normalize, scale) in one launch for the ungrouped case.

Why the launch count matters: Moonlight's 26 MoE layers ran ~40 small kernels each for
routing + alignment; with everything else already fused that was most of the batch-1 step.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from emberserve.model import ops

_KERNEL = None
BLOCK_M = 16


def fused_moe_enabled(x: Tensor) -> bool:
    return (os.environ.get("EMBERSERVE_FUSED_MOE", "1") != "0"
            and (x.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1"))


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _grouped_gemm_kernel(
        a_ptr, w_ptr, out_ptr, sorted_ids_ptr, block_expert_ptr, topk_w_ptr,
        n_out, k_dim, num_valid, num_experts,
        stride_am, stride_we, stride_wn, stride_wk, stride_om,
        A_DIV: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(block_expert_ptr + pid_m)
        if e >= num_experts:
            return
        offs_m = pid_m * BM + tl.arange(0, BM)
        ids = tl.load(sorted_ids_ptr + offs_m)  # assignment ids; >= num_valid is padding
        valid = ids < num_valid
        a_rows = tl.where(valid, ids // A_DIV, 0)
        offs_n = pid_n * BN + tl.arange(0, BN)
        n_valid = offs_n < n_out
        offs_k = tl.arange(0, BK)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(0, k_dim, BK):
            kk = k0 + offs_k
            k_valid = kk < k_dim
            a = tl.load(a_ptr + a_rows[:, None] * stride_am + kk[None, :],
                        mask=valid[:, None] & k_valid[None, :], other=0.0)  # [BM, BK]
            w = tl.load(w_ptr + e.to(tl.int64) * stride_we + offs_n[None, :] * stride_wn
                        + kk[:, None] * stride_wk,
                        mask=k_valid[:, None] & n_valid[None, :], other=0.0)  # [BK, BN]
            acc += tl.dot(a, w)
        if MUL_ROUTED_WEIGHT:
            rw = tl.load(topk_w_ptr + ids, mask=valid, other=0.0)
            acc = acc * rw[:, None]
        tl.store(out_ptr + ids[:, None] * stride_om + offs_n[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=valid[:, None] & n_valid[None, :])

    _KERNEL = _grouped_gemm_kernel
    return _KERNEL


_ALIGN_KERNEL = None
ALIGN_KERNEL_MAX_ASSIGNMENTS = 4096  # past this the torch version (sort-based) is used


def _align_kernel():
    global _ALIGN_KERNEL
    if _ALIGN_KERNEL is not None:
        return _ALIGN_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _moe_align_kernel(
        ids_ptr, sorted_ptr, block_expert_ptr, num_valid, max_padded, max_blocks,
        E: tl.constexpr, E_PAD: tl.constexpr, BLOCK_M: tl.constexpr, CHUNK: tl.constexpr,
    ):
        """Program e: count every expert's assignments (all programs redundantly, so no
        second launch is needed for the prefix sum), then place ITS assignments at
        `pad_start[e] + rank` and stamp its blocks with e. Program E-1 also fills the tail
        past the last real block with the sentinels (expert E, assignment num_valid)."""
        e = tl.program_id(0)
        offs_e = tl.arange(0, E_PAD)
        offs_c = tl.arange(0, CHUNK)
        counts = tl.zeros([E_PAD], dtype=tl.int32)
        for start in range(0, num_valid, CHUNK):
            i = start + offs_c
            ids = tl.load(ids_ptr + i, mask=i < num_valid, other=-1)
            counts += tl.sum((ids[None, :] == offs_e[:, None]).to(tl.int32), axis=1)
        padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        ends = tl.cumsum(padded, axis=0)
        starts = ends - padded
        mine = offs_e == e
        my_count = tl.sum(tl.where(mine, counts, 0), axis=0)
        my_start = tl.sum(tl.where(mine, starts, 0), axis=0)
        my_padded = tl.sum(tl.where(mine, padded, 0), axis=0)
        carry = tl.zeros([], dtype=tl.int32)
        for start in range(0, num_valid, CHUNK):
            i = start + offs_c
            ids = tl.load(ids_ptr + i, mask=i < num_valid, other=-1)
            hit = ids == e
            hit_i = hit.to(tl.int32)
            rank = tl.cumsum(hit_i, axis=0) - hit_i + carry
            tl.store(sorted_ptr + my_start + rank, i.to(tl.int32), mask=hit)
            carry += tl.sum(hit_i, axis=0)
        # padding rows of this expert's run (fewer than BLOCK_M of them)
        t = tl.arange(0, BLOCK_M)
        tl.store(sorted_ptr + my_start + my_count + t, tl.full([BLOCK_M], 0, tl.int32) + num_valid,
                 mask=t < my_padded - my_count)
        # this expert's blocks
        b0 = my_start // BLOCK_M
        nb = my_padded // BLOCK_M
        e_vec = tl.full([CHUNK], 0, tl.int32) + e
        for s in range(0, nb, CHUNK):
            tl.store(block_expert_ptr + b0 + s + offs_c, e_vec, mask=s + offs_c < nb)
        if e == E - 1:
            total = tl.max(ends, axis=0)
            total_blocks = total // BLOCK_M
            sentinel_e = tl.full([CHUNK], 0, tl.int32) + E
            for s in range(total_blocks, max_blocks, CHUNK):
                tl.store(block_expert_ptr + s + offs_c, sentinel_e, mask=s + offs_c < max_blocks)
            sentinel_v = tl.full([CHUNK], 0, tl.int32) + num_valid
            for s in range(total, max_padded, CHUNK):
                tl.store(sorted_ptr + s + offs_c, sentinel_v, mask=s + offs_c < max_padded)

    _ALIGN_KERNEL = _moe_align_kernel
    return _ALIGN_KERNEL


_GATE_KERNEL = None


def _gate_kernel():
    global _GATE_KERNEL
    if _GATE_KERNEL is not None:
        return _GATE_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _topk_gate_kernel(
        logits_ptr, bias_ptr, idx_ptr, w_ptr, n, scaling, stride_l,
        E: tl.constexpr, E_PAD: tl.constexpr, K: tl.constexpr, NORM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Rows of sigmoid(logits): choose the top-K by `score + bias` (ties -> lowest
        expert id), weights are the UNBIASED scores, normalized to sum 1 if NORM, times
        `scaling`. Two passes of K arg-max steps: the first for the normalizer."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        cols = tl.arange(0, E_PAD)
        in_range = (rows[:, None] < n) & (cols[None, :] < E)
        logits = tl.load(logits_ptr + rows[:, None] * stride_l + cols[None, :], mask=in_range,
                         other=0.0)
        scores = tl.sigmoid(logits.to(tl.float32))
        bias = tl.load(bias_ptr + cols, mask=cols < E, other=0.0).to(tl.float32)
        choice0 = tl.where(in_range, scores + bias[None, :], float("-inf"))
        choice = choice0
        wsum = tl.zeros([BLOCK_N], dtype=tl.float32)
        for _ in tl.static_range(K):
            best = tl.max(choice, axis=1)
            sel = tl.min(tl.where(choice == best[:, None], cols[None, :], E_PAD), axis=1)
            picked = cols[None, :] == sel[:, None]
            wsum += tl.sum(tl.where(picked, scores, 0.0), axis=1)
            choice = tl.where(picked, float("-inf"), choice)
        if NORM:
            denom = wsum + 1e-20
        else:
            denom = tl.zeros([BLOCK_N], dtype=tl.float32) + 1.0
        choice = choice0
        for j in tl.static_range(K):
            best = tl.max(choice, axis=1)
            sel = tl.min(tl.where(choice == best[:, None], cols[None, :], E_PAD), axis=1)
            picked = cols[None, :] == sel[:, None]
            w = tl.sum(tl.where(picked, scores, 0.0), axis=1) / denom * scaling
            tl.store(idx_ptr + rows * K + j, sel.to(tl.int64), mask=rows < n)
            tl.store(w_ptr + rows * K + j, w, mask=rows < n)
            choice = tl.where(picked, float("-inf"), choice)

    _GATE_KERNEL = _topk_gate_kernel
    return _GATE_KERNEL


def moe_align(topk_ids: Tensor, num_experts: int, block_m: int = BLOCK_M,
              use_kernel: bool | None = None) -> tuple[Tensor, Tensor, int]:
    """Block-aligned expert layout, all on the device (no host sync).

    Returns `(sorted_ids [max_padded] int32, block_expert [max_blocks] int32, num_valid)`:
    `sorted_ids[p]` is the assignment id (`token * k + choice`) at padded position p, or
    `num_valid` for padding; `block_expert[b]` is the expert of block b, or `num_experts`
    past the end. Within an expert's run the assignments keep ascending id order (both
    versions), so the two are bit-identical.

    One Triton launch (`_moe_align_kernel`) for up to `ALIGN_KERNEL_MAX_ASSIGNMENTS`
    assignments — every decode step — else ~18 torch ops (sort, scatter_add, cumsums,
    searchsorted); `use_kernel` forces either for tests."""
    n, k = topk_ids.shape
    num_valid = n * k
    dev = topk_ids.device
    flat = topk_ids.reshape(-1)
    if use_kernel is None:
        use_kernel = (num_valid <= ALIGN_KERNEL_MAX_ASSIGNMENTS
                      and (flat.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1"))
    if use_kernel:
        return _moe_align_kernel_launch(flat.to(torch.int32), num_experts, block_m)
    order = torch.argsort(flat, stable=True)
    sorted_e = flat[order]
    # scatter_add, not bincount: bincount on CUDA syncs to size its output (input.max()).
    counts = torch.zeros(num_experts, dtype=torch.long, device=dev).scatter_add_(
        0, flat.long(), torch.ones_like(flat, dtype=torch.long))  # [E]
    padded = (counts + block_m - 1) // block_m * block_m
    pad_starts = torch.cumsum(padded, 0) - padded
    unpadded_starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(num_valid, device=dev) - unpadded_starts[sorted_e]
    pos = pad_starts[sorted_e] + rank
    max_padded = (num_valid + num_experts * (block_m - 1) + block_m - 1) // block_m * block_m
    sorted_ids = torch.full((max_padded,), num_valid, dtype=torch.int32, device=dev)
    sorted_ids[pos] = order.to(torch.int32)
    max_blocks = max_padded // block_m
    block_ends = torch.cumsum(padded, 0) // block_m  # cumulative block count per expert
    block_expert = torch.searchsorted(block_ends, torch.arange(max_blocks, device=dev),
                                      right=True).to(torch.int32)
    return sorted_ids, block_expert, num_valid


def _moe_align_kernel_launch(flat: Tensor, num_experts: int, block_m: int) -> tuple[Tensor, Tensor, int]:
    import triton

    num_valid = flat.numel()
    max_padded = (num_valid + num_experts * (block_m - 1) + block_m - 1) // block_m * block_m
    max_blocks = max_padded // block_m
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=flat.device)
    block_expert = torch.empty(max_blocks, dtype=torch.int32, device=flat.device)
    e_pad = triton.next_power_of_2(num_experts)
    chunk = max(16, min(256, 8192 // e_pad))
    _align_kernel()[(num_experts,)](
        flat, sorted_ids, block_expert, num_valid, max_padded, max_blocks,
        E=num_experts, E_PAD=e_pad, BLOCK_M=block_m, CHUNK=chunk, num_warps=4)
    return sorted_ids, block_expert, num_valid


def topk_gate(logits: Tensor, bias: Tensor, k: int, norm_topk_prob: bool,
              routed_scaling_factor: float) -> tuple[Tensor, Tensor]:
    """The ungrouped (`n_group == 1`) router after the linear: `logits [N, E]` fp32 ->
    `(idx [N, k] int64, w [N, k] fp32)`, one launch. See `_topk_gate_kernel`."""
    import triton

    n, e = logits.shape
    idx = torch.empty((n, k), dtype=torch.int64, device=logits.device)
    w = torch.empty((n, k), dtype=torch.float32, device=logits.device)
    if n == 0:
        return idx, w
    e_pad = triton.next_power_of_2(e)
    block_n = max(1, min(16, 4096 // e_pad))
    _gate_kernel()[(triton.cdiv(n, block_n),)](
        logits, bias, idx, w, n, float(routed_scaling_factor), logits.stride(0),
        E=e, E_PAD=e_pad, K=k, NORM=norm_topk_prob, BLOCK_N=block_n, num_warps=4)
    return idx, w


def gemm_config(num_tokens: int) -> dict:
    """Tile config for the grouped GEMM: `(block_n, block_k, num_warps, num_stages)`.

    Decode-sized steps (few tokens) are pure weight streaming; prefill-sized steps have
    real reuse across the 16 rows of a block, so the two may want different tiles.
    `EMBERSERVE_MOE_CONFIG=bn,bk,warps,stages` overrides for A/B runs
    (`scripts/bench_moe.py` sweeps them at Moonlight's geometry).
    """
    forced = os.environ.get("EMBERSERVE_MOE_CONFIG")
    if forced:
        bn, bk, warps, stages = (int(v) for v in forced.split(","))
        return dict(block_n=bn, block_k=bk, num_warps=warps, num_stages=stages)
    return dict(block_n=64, block_k=64, num_warps=4, num_stages=3)  # until bench_moe says otherwise


def _grouped_gemm(a: Tensor, w: Tensor, sorted_ids: Tensor, block_expert: Tensor, num_valid: int,
                  a_div: int, topk_w: Tensor | None, out_rows: int, block_m: int = BLOCK_M,
                  block_n: int = 64, block_k: int = 64, num_warps: int = 4,
                  num_stages: int = 3) -> Tensor:
    """`out[assignment, :] = a[assignment // a_div, :] @ w[expert]^T (* topk_w[assignment])`.
    `w: [E, N_out, K]`."""
    import triton

    num_experts, n_out, k_dim = w.shape
    assert a.shape[1] == k_dim and a.stride(1) == 1 and w.stride(2) == 1
    out = torch.empty((out_rows, n_out), dtype=a.dtype, device=a.device)
    block_n = min(block_n, max(16, triton.next_power_of_2(n_out)))
    block_k = min(block_k, max(16, triton.next_power_of_2(k_dim)))
    grid = (block_expert.numel(), triton.cdiv(n_out, block_n))
    _kernel()[grid](
        a, w, out, sorted_ids, block_expert,
        topk_w if topk_w is not None else out,
        n_out, k_dim, num_valid, num_experts,
        a.stride(0), w.stride(0), w.stride(1), w.stride(2), out.stride(0),
        A_DIV=a_div, MUL_ROUTED_WEIGHT=topk_w is not None,
        BM=block_m, BN=block_n, BK=block_k, num_warps=num_warps, num_stages=num_stages,
    )
    return out


def fused_moe_forward(x: Tensor, topk_ids: Tensor, topk_w: Tensor, w_gate_up: Tensor,
                      w_down: Tensor) -> Tensor:
    """`x [N, H]`, `topk_ids [N, k]`, `topk_w [N, k]` fp32, `w_gate_up [E, 2I, H]`,
    `w_down [E, H, I]` -> `[N, H]` (routed experts only; shared experts are added by the caller)."""
    n, k = topk_ids.shape
    num_experts = w_gate_up.shape[0]
    cfg = gemm_config(n)
    sorted_ids, block_expert, num_valid = moe_align(topk_ids, num_experts)
    inter = _grouped_gemm(x, w_gate_up, sorted_ids, block_expert, num_valid, a_div=k,
                          topk_w=None, out_rows=num_valid, **cfg)  # [N*k, 2I]
    act = ops.silu_and_mul(inter)  # [N*k, I]
    down = _grouped_gemm(act, w_down, sorted_ids, block_expert, num_valid, a_div=1,
                         topk_w=topk_w.reshape(-1).float().contiguous(), out_rows=num_valid,
                         **cfg)  # [N*k, H]
    return down.view(n, k, -1).sum(dim=1)
