"""Fused Triton ops (RMSNorm + residual, RoPE, SiLU-mul) vs their PyTorch references, on
the CPU through Triton's interpreter. Same module-level rules as tests/test_paged_triton.py:
`TRITON_INTERPRET=1` is set before triton is imported and the module is skipped on CUDA
machines (the GPU tests cover the kernels there)."""

from __future__ import annotations

import os

import pytest

try:
    import torch as _torch_probe
    _HAS_CUDA = _torch_probe.cuda.is_available()
except Exception:  # noqa: BLE001
    _HAS_CUDA = False
if _HAS_CUDA and os.environ.get("PAGEDSERVE_FORCE_INTERPRETER") != "1":
    pytest.skip("CUDA present: interpreter tests skipped so TRITON_INTERPRET does not leak "
                "into the GPU tests", allow_module_level=True)
os.environ["TRITON_INTERPRET"] = "1"  # noqa: E402  (must precede the triton import)

import torch  # noqa: E402

pytest.importorskip("triton")

from pagedserve.model import ops  # noqa: E402
from pagedserve.model import ops_triton as kt  # noqa: E402

torch.set_num_threads(2)
DTYPES = [torch.float32, torch.float16]


def _rand(*shape, dtype, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden", [64, 896])  # 896 = Qwen2.5-0.5B; not a power of two
def test_rmsnorm(dtype, hidden):
    x = _rand(5, hidden, dtype=dtype, seed=1, scale=3.0)
    w = _rand(hidden, dtype=dtype, seed=2, scale=0.5) + 1.0
    torch.testing.assert_close(kt.rmsnorm_triton(x, w, 1e-6), ops.rmsnorm_torch(x, w, 1e-6),
                               atol=0, rtol=0) if dtype == torch.float16 else \
        torch.testing.assert_close(kt.rmsnorm_triton(x, w, 1e-6), ops.rmsnorm_torch(x, w, 1e-6),
                                   atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_rmsnorm(dtype):
    x = _rand(7, 896, dtype=dtype, seed=3)
    r = _rand(7, 896, dtype=dtype, seed=4, scale=4.0)
    w = _rand(896, dtype=dtype, seed=5, scale=0.3) + 1.0
    got, got_res = kt.fused_add_rmsnorm_triton(x, r, w, 1e-6)
    want, want_res = ops.fused_add_rmsnorm_torch(x, r, w, 1e-6)
    torch.testing.assert_close(got_res, want_res, atol=0, rtol=0)  # the rounded fp16 sum
    tol = dict(atol=0, rtol=0) if dtype == torch.float16 else dict(atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(got, want, **tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", [(14, 2), (4, 4), (3, 1)])
def test_rope_in_place_matches_reference(dtype, heads):
    h, hk, d = heads[0], heads[1], 64
    n = 6
    positions = torch.tensor([0, 1, 5, 100, 4095, 17])
    inv = 1.0 / (1e6 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    freqs = torch.outer(torch.arange(4096, dtype=torch.float32), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    # q/k as strided views of one fused [n, (h+2hk)*d] buffer, like the model's qkv split
    qkv = _rand(n, (h + 2 * hk) * d, dtype=dtype, seed=6)
    q = qkv[:, :h * d].view(n, h, d)
    k = qkv[:, h * d:(h + hk) * d].view(n, hk, d)
    v_before = qkv[:, (h + hk) * d:].clone()
    want_q, want_k = ops.rope_torch(q.clone(), k.clone(), cos, sin, positions)
    got_q, got_k = kt.rope_triton(q, k, cos, sin, positions)
    tol = dict(atol=1e-3, rtol=0) if dtype == torch.float16 else dict(atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got_q, want_q, **tol)
    torch.testing.assert_close(got_k, want_k, **tol)
    assert got_q.data_ptr() == q.data_ptr(), "rope must be in place"
    torch.testing.assert_close(qkv[:, (h + hk) * d:], v_before, atol=0, rtol=0), "v untouched"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inter", [128, 4864])  # 4864 = Qwen2.5-0.5B intermediate size
def test_silu_and_mul(dtype, inter):
    x = _rand(3, 2 * inter, dtype=dtype, seed=7, scale=2.0)
    got = kt.silu_and_mul_triton(x)
    want = ops.silu_and_mul_torch(x)
    tol = dict(atol=1e-3, rtol=1e-2) if dtype == torch.float16 else dict(atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(got, want, **tol)
    assert got.shape == (3, inter)


def test_dispatch_uses_torch_off_cuda():
    x = torch.randn(2, 64)
    w = torch.ones(64)
    assert not ops.fused_enabled(x)
    torch.testing.assert_close(ops.rmsnorm(x, w, 1e-6), ops.rmsnorm_torch(x, w, 1e-6))
