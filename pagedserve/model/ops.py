"""Fused elementwise ops for the decoder: RMSNorm (+ residual add), RoPE, SiLU-mul.

Each op has a plain-PyTorch implementation (the reference; what CPU and MPS run) and a
Triton kernel used on CUDA. The kernels exist because of kernel *count*, not FLOPs: a
batch-1 decode step of Qwen2.5-0.5B inside a CUDA graph measured 3.9 ms on an A100 with
~42 kernels per layer (an unfused RMSNorm alone is ~8: cast, square, mean, add, rsqrt,
mul, cast, mul), and every kernel costs 3-4 us of GPU time even with launch overhead gone.

Numerics follow HF exactly in structure so the golden gate stays meaningful:
  rmsnorm:   weight * (x.float() * rsqrt(mean(x.float()^2) + eps)).to(x.dtype)
  residual:  the fp16 sum `residual + x` is rounded first, then normalized
  rope:      float32 rotate-half math, cast back to the input dtype
  silu_mul:  silu in float32, cast, then a single half-precision multiply

`PAGEDSERVE_FUSED_OPS=0` forces the PyTorch path on CUDA (A/B runs, bisecting numerics).
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import torch
import torch.nn.functional as F

_triton_ops: ModuleType | None | bool = None  # lazily imported; False once import failed


def _kernels() -> ModuleType | None:
    """`pagedserve.model.ops_triton`, imported on first CUDA use. Importing triton is
    deferred on purpose: triton reads `TRITON_INTERPRET` once, at import, and the CPU
    interpreter tests set it at the top of their own module."""
    global _triton_ops
    if _triton_ops is None:
        try:
            _triton_ops = importlib.import_module("pagedserve.model.ops_triton")
        except ImportError:  # pragma: no cover - CPU-only installs
            _triton_ops = False
    return _triton_ops or None


def fused_enabled(x: torch.Tensor) -> bool:
    """Triton path: CUDA tensor, triton importable, not disabled by env."""
    return (x.is_cuda and os.environ.get("PAGEDSERVE_FUSED_OPS", "1") != "0"
            and os.environ.get("TRITON_INTERPRET", "0") != "1" and _kernels() is not None)


# ---- reference (PyTorch) --------------------------------------------------------------

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return weight * xf.to(x.dtype)


def fused_add_rmsnorm_torch(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                            eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """`residual = residual + x` (in the activation dtype), then `rmsnorm(residual)`.
    Returns (normed, new_residual)."""
    residual = residual + x
    return rmsnorm_torch(residual, weight, eps), residual


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_torch(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`q: [N, H, D]`, `k: [N, Hkv, D]`, tables `cos/sin: [max_pos, D]` float32."""
    c = cos[positions][:, None, :]
    s = sin[positions][:, None, :]

    def rot(x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * c + _rotate_half(xf) * s).to(x.dtype)

    return rot(q), rot(k)


def silu_and_mul_torch(x: torch.Tensor) -> torch.Tensor:
    """`x: [N, 2I]` -> `silu(x[:, :I]) * x[:, I:]`."""
    inter = x.shape[-1] // 2
    return F.silu(x[..., :inter]) * x[..., inter:]


# ---- dispatch ---------------------------------------------------------------------------

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if fused_enabled(x):
        return _kernels().rmsnorm_triton(x, weight, eps)
    return rmsnorm_torch(x, weight, eps)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                      eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if fused_enabled(x):
        return _kernels().fused_add_rmsnorm_triton(x, residual, weight, eps)
    return fused_add_rmsnorm_torch(x, residual, weight, eps)


def rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
         positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if fused_enabled(q):
        return _kernels().rope_triton(q, k, cos, sin, positions)
    return rope_torch(q, k, cos, sin, positions)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    if fused_enabled(x):
        return _kernels().silu_and_mul_triton(x)
    return silu_and_mul_torch(x)
