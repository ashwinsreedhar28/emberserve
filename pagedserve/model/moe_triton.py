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

Same structure as vLLM's `fused_moe` kernel; block alignment is done with torch ops here
instead of a custom CUDA kernel.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from pagedserve.model import ops

_KERNEL = None
BLOCK_M = 16


def fused_moe_enabled(x: Tensor) -> bool:
    return (os.environ.get("PAGEDSERVE_FUSED_MOE", "1") != "0"
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


def moe_align(topk_ids: Tensor, num_experts: int, block_m: int = BLOCK_M) -> tuple[Tensor, Tensor, int]:
    """Block-aligned expert layout, all on the device (no host sync).

    Returns `(sorted_ids [max_padded] int32, block_expert [max_blocks] int32, num_valid)`:
    `sorted_ids[p]` is the assignment id (`token * k + choice`) at padded position p, or
    `num_valid` for padding; `block_expert[b]` is the expert of block b, or `num_experts`
    past the end."""
    n, k = topk_ids.shape
    num_valid = n * k
    dev = topk_ids.device
    flat = topk_ids.reshape(-1)
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


def _grouped_gemm(a: Tensor, w: Tensor, sorted_ids: Tensor, block_expert: Tensor, num_valid: int,
                  a_div: int, topk_w: Tensor | None, out_rows: int, block_m: int = BLOCK_M,
                  block_n: int = 64, block_k: int = 64) -> Tensor:
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
        BM=block_m, BN=block_n, BK=block_k, num_warps=4,
    )
    return out


def fused_moe_forward(x: Tensor, topk_ids: Tensor, topk_w: Tensor, w_gate_up: Tensor,
                      w_down: Tensor) -> Tensor:
    """`x [N, H]`, `topk_ids [N, k]`, `topk_w [N, k]` fp32, `w_gate_up [E, 2I, H]`,
    `w_down [E, H, I]` -> `[N, H]` (routed experts only; shared experts are added by the caller)."""
    n, k = topk_ids.shape
    num_experts = w_gate_up.shape[0]
    sorted_ids, block_expert, num_valid = moe_align(topk_ids, num_experts)
    inter = _grouped_gemm(x, w_gate_up, sorted_ids, block_expert, num_valid, a_div=k,
                          topk_w=None, out_rows=num_valid)  # [N*k, 2I]
    act = ops.silu_and_mul(inter)  # [N*k, I]
    down = _grouped_gemm(act, w_down, sorted_ids, block_expert, num_valid, a_div=1,
                         topk_w=topk_w.reshape(-1).float().contiguous(), out_rows=num_valid)  # [N*k, H]
    return down.view(n, k, -1).sum(dim=1)
