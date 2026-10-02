"""Triton kernels behind `emberserve.model.ops` (imported lazily; see `ops._kernels`).

Each wrapper takes the same arguments as its PyTorch reference in `ops.py` and matches it
in rounding structure. Kernels run on CUDA, or on the CPU under `TRITON_INTERPRET=1` for
the tests in tests/test_fused_ops.py.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr, n_cols, eps,
                    stride_x, stride_res, stride_out, stride_res_out,
                    HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        r = tl.load(res_ptr + row * stride_res + cols, mask=mask, other=0.0)
        # Round the sum to the activation dtype first: this IS the new residual stream.
        x = (x.to(tl.float32) + r.to(tl.float32)).to(x.dtype)
        tl.store(res_out_ptr + row * stride_res_out + cols, x, mask=mask)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(var + eps)
    y = (xf * inv).to(x.dtype)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    out = (y.to(tl.float32) * w.to(tl.float32)).to(x.dtype)
    tl.store(out_ptr + row * stride_out + cols, out, mask=mask)

@triton.jit
def _rope_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_q_heads, num_k_heads,
                 stride_q_tok, stride_q_head, stride_k_tok, stride_k_head,
                 HALF: tl.constexpr, HEADS_PAD: tl.constexpr):
    """One program per token: rotates every q head and every k head in place.
    Rotate-half layout: out[:half] = x1*c - x2*s, out[half:] = x2*c + x1*s, with the
    table row `cos[pos]` of width 2*HALF holding the same HALF values twice."""
    tok = tl.program_id(0)
    pos = tl.load(pos_ptr + tok)
    d = tl.arange(0, HALF)
    c = tl.load(cos_ptr + pos * (2 * HALF) + d)  # float32 tables
    s = tl.load(sin_ptr + pos * (2 * HALF) + d)
    heads = tl.arange(0, HEADS_PAD)

    qmask = heads[:, None] < num_q_heads
    q_base = q_ptr + tok * stride_q_tok + heads[:, None] * stride_q_head + d[None, :]
    x1 = tl.load(q_base, mask=qmask, other=0.0)
    x2 = tl.load(q_base + HALF, mask=qmask, other=0.0)
    x1f = x1.to(tl.float32)
    x2f = x2.to(tl.float32)
    o1 = x1f * c[None, :] + (-x2f) * s[None, :]
    o2 = x2f * c[None, :] + x1f * s[None, :]
    tl.store(q_base, o1.to(x1.dtype), mask=qmask)
    tl.store(q_base + HALF, o2.to(x1.dtype), mask=qmask)

    kmask = heads[:, None] < num_k_heads
    k_base = k_ptr + tok * stride_k_tok + heads[:, None] * stride_k_head + d[None, :]
    y1 = tl.load(k_base, mask=kmask, other=0.0)
    y2 = tl.load(k_base + HALF, mask=kmask, other=0.0)
    y1f = y1.to(tl.float32)
    y2f = y2.to(tl.float32)
    p1 = y1f * c[None, :] + (-y2f) * s[None, :]
    p2 = y2f * c[None, :] + y1f * s[None, :]
    tl.store(k_base, p1.to(y1.dtype), mask=kmask)
    tl.store(k_base + HALF, p2.to(y1.dtype), mask=kmask)

@triton.jit
def _silu_and_mul_kernel(x_ptr, out_ptr, inter, stride_x, stride_out, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.program_id(1) * BLOCK
    cols = start + tl.arange(0, BLOCK)
    mask = cols < inter
    g = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    u = tl.load(x_ptr + row * stride_x + inter + cols, mask=mask, other=0.0)
    gf = g.to(tl.float32)
    act = (gf / (1.0 + tl.exp(-gf))).to(g.dtype)  # silu in fp32, rounded once
    out = (act.to(tl.float32) * u.to(tl.float32)).to(g.dtype)
    tl.store(out_ptr + row * stride_out + cols, out, mask=mask)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(x2)
    n = x2.shape[-1]
    _rmsnorm_kernel[(x2.shape[0],)](
        x2, x2, weight, out, out, n, eps, x2.stride(0), 0, out.stride(0), 0,
        HAS_RESIDUAL=False, BLOCK=_next_pow2(n))
    return out.view_as(x)


def fused_add_rmsnorm_triton(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                             eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    x2 = x.reshape(-1, x.shape[-1])
    r2 = residual.reshape(-1, x.shape[-1])
    out = torch.empty_like(x2)
    new_res = torch.empty_like(r2)
    n = x2.shape[-1]
    _rmsnorm_kernel[(x2.shape[0],)](
        x2, r2, weight, out, new_res, n, eps, x2.stride(0), r2.stride(0), out.stride(0),
        new_res.stride(0), HAS_RESIDUAL=True, BLOCK=_next_pow2(n))
    return out.view_as(x), new_res.view_as(residual)


def rope_triton(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """In place on `q` and `k` (both `[N, heads, D]`, contiguous in the head/dim axes)."""
    n, h, d = q.shape
    hk = k.shape[1]
    assert q.stride(2) == 1 and k.stride(2) == 1 and d % 2 == 0
    if n == 0:
        return q, k
    _rope_kernel[(n,)](
        q, k, cos, sin, positions, h, hk, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        HALF=d // 2, HEADS_PAD=_next_pow2(max(h, hk)))
    return q, k


def silu_and_mul_triton(x: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    inter = x2.shape[-1] // 2
    out = torch.empty((x2.shape[0], inter), dtype=x.dtype, device=x.device)
    block = min(1024, _next_pow2(inter))
    _silu_and_mul_kernel[(x2.shape[0], triton.cdiv(inter, block))](
        x2, out, inter, x2.stride(0), out.stride(0), BLOCK=block)
    return out.view(*x.shape[:-1], inter)


