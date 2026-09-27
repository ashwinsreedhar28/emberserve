"""Weight-only int8 quantization: per-output-channel symmetric int8 weights, dequantized
on the fly inside a Triton GEMM.

Why: a decode step at batch 1 is the weight read (7B fp16 on an A100: 15 GB at ~1.5 TB/s,
10 ms), so halving the bytes of every projection halves the floor; the activations stay
fp16/bf16 and the arithmetic is unchanged. Per-channel scales (one fp32 per output row)
keep the rounding error small enough that greedy outputs mostly match the fp16 model; it
is a quality trade, not an exact transform, and `check_golden --quantization int8`
reports how many tokens move.

    W [N, K] fp16  ->  q = round(W / s) in int8, s[n] = max_k |W[n, k]| / 127

The GEMM (`int8_gemm`, Triton) streams int8 weight tiles, converts them to the activation
dtype, multiplies on the tensor cores and applies `s` to the fp32 accumulator, so the
result is `x @ (q * s)^T` computed as `(x @ q^T) * s` (exact rearrangement). The torch
reference does the same in plain ops (CPU tests); `PAGEDSERVE_INT8_KERNEL=0` forces it.
`quantize_model` swaps every `nn.Linear` of the decoder (and the lm_head) for
`Int8Linear`, one at a time so the fp16 copy is freed before the next one is converted.
MoE experts (stacked 3-D weights) stay fp16/bf16 for now.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_KERNEL = None


def quantize_int8_weight(w: Tensor) -> tuple[Tensor, Tensor]:
    """`w [N, K]` -> (`q [N, K]` int8, `scale [N]` fp32), symmetric per output row."""
    wf = w.detach().float()
    scale = wf.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(wf / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


def int8_gemm_torch(x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None = None) -> Tensor:
    """Reference: `(x @ q^T) * scale (+ bias)` with an fp32 accumulator, in x's dtype."""
    out = torch.matmul(x.float(), q.float().t()) * scale[None, :]
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def kernel_enabled(x: Tensor) -> bool:
    return (os.environ.get("PAGEDSERVE_INT8_KERNEL", "1") != "0"
            and (x.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1"))


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _int8_gemm_kernel(
        a_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr, M, N, K,
        stride_am, stride_wn, stride_om,
        HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        m_valid = offs_m < M
        n_valid = offs_n < N
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + offs_k
            k_valid = kk < K
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + kk[None, :],
                        mask=m_valid[:, None] & k_valid[None, :], other=0.0)  # [BM, BK]
            w = tl.load(w_ptr + offs_n[:, None] * stride_wn + kk[None, :],
                        mask=n_valid[:, None] & k_valid[None, :], other=0)  # [BN, BK] int8
            acc += tl.dot(a, tl.trans(w.to(a.dtype)))
        scale = tl.load(scale_ptr + offs_n, mask=n_valid, other=0.0)
        out = acc * scale[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_valid, other=0.0).to(tl.float32)
            out = out + bias[None, :]
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
                 out.to(out_ptr.dtype.element_ty), mask=m_valid[:, None] & n_valid[None, :])

    _KERNEL = _int8_gemm_kernel
    return _KERNEL


def int8_gemm(x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None = None) -> Tensor:
    """`x [M, K]` (fp16/bf16, unit last stride) @ dequant(`q [N, K]` int8, `scale [N]`)^T."""
    import triton

    m, k = x.shape
    n = q.shape[0]
    assert q.shape[1] == k and x.stride(1) == 1 and q.stride(1) == 1
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    bm = 16 if m <= 16 else (64 if m <= 64 else 128)
    bn = 64 if m <= 16 else 128
    bk = 128 if m <= 16 else 64
    bk = min(bk, max(16, triton.next_power_of_2(k)))
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _kernel()[grid](
        x, q, scale, bias if bias is not None else scale, out, m, n, k,
        x.stride(0), q.stride(0), out.stride(0),
        HAS_BIAS=bias is not None, BM=bm, BN=bn, BK=bk,
        num_warps=4 if bm <= 64 else 8, num_stages=3,
    )
    return out


class Int8Linear(nn.Module):
    """Drop-in for `nn.Linear` holding int8 weights + fp32 per-row scales (+ the fp16 bias)."""

    def __init__(self, weight_q: Tensor, scale: Tensor, bias: Tensor | None,
                 act_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.weight_q = nn.Parameter(weight_q, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.in_features = weight_q.shape[1]
        self.out_features = weight_q.shape[0]
        self.act_dtype = act_dtype

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "Int8Linear":
        q, s = quantize_int8_weight(lin.weight)
        bias = lin.bias.detach().clone() if lin.bias is not None else None
        return cls(q, s, bias, act_dtype=lin.weight.dtype)

    @property
    def weight(self) -> Tensor:
        """The dequantized weight (materialized; for inspection and the tests)."""
        return (self.weight_q.float() * self.scale[:, None]).to(self.act_dtype)

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        if kernel_enabled(x2) and x2.dtype in (torch.float16, torch.bfloat16):
            out = int8_gemm(x2.contiguous(), self.weight_q, self.scale, self.bias)
        else:
            out = int8_gemm_torch(x2, self.weight_q, self.scale, self.bias)
        return out.view(*lead, self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8 per-channel"


DEFAULT_SKIP = ("embed_tokens", "kv_b_proj")  # kv_b_proj: MLA reads its weight directly (2 MB)


def quantize_model(model: nn.Module, method: str = "int8", skip: tuple[str, ...] = DEFAULT_SKIP) -> int:
    """Swap every 2-D `nn.Linear` (decoder projections, lm_head) for `Int8Linear`, in
    place, one module at a time. Returns the number of layers converted. `nn.Embedding`,
    the MoE expert stacks and the names in `skip` are left as they are."""
    if method != "int8":
        raise ValueError(f"unknown quantization {method!r} (supported: int8)")
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, Int8Linear.from_linear(child))
                del child
                count += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return count


def linear(module: nn.Module, x: Tensor) -> Tensor:
    """`F.linear` on an `nn.Linear` or the int8 path on an `Int8Linear` (for code that
    calls into a projection's weight directly)."""
    if isinstance(module, Int8Linear):
        return module(x)
    return F.linear(x, module.weight, module.bias)
