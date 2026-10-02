"""The int8 dequant GEMM on the Triton CPU interpreter vs the torch reference. Same
module rules as tests/test_paged_triton.py."""

from __future__ import annotations

import os

import pytest

try:
    import torch as _torch_probe
    _HAS_CUDA = _torch_probe.cuda.is_available()
except Exception:  # noqa: BLE001
    _HAS_CUDA = False
if _HAS_CUDA and os.environ.get("EMBERSERVE_FORCE_INTERPRETER") != "1":
    pytest.skip("CUDA present: interpreter tests skipped", allow_module_level=True)
import importlib.util  # noqa: E402

if importlib.util.find_spec("triton") is None:  # before the env var: it would leak into
    pytest.skip("triton not installed", allow_module_level=True)  # other modules' tests
os.environ["TRITON_INTERPRET"] = "1"  # noqa: E402

import torch  # noqa: E402

pytest.importorskip("triton")

from emberserve.model.quant import Int8Linear, int8_gemm, int8_gemm_torch, quantize_int8_weight  # noqa: E402

torch.set_num_threads(2)


@pytest.mark.parametrize("m", [1, 5, 40])
@pytest.mark.parametrize("n,k", [(96, 64), (40, 200)])
@pytest.mark.parametrize("bias", [False, True])
def test_kernel_matches_reference(m, n, k, bias):
    g = torch.Generator().manual_seed(m * 7 + n)
    w = torch.randn(n, k, generator=g) * 0.1
    q, s = quantize_int8_weight(w)
    x = (torch.randn(m, k, generator=g)).half()
    b = (torch.randn(n, generator=g)).half() if bias else None
    got = int8_gemm(x, q, s, b)
    want = int8_gemm_torch(x, q, s, b)
    torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)


def test_split_k_partials_reduce_to_the_reference():
    """Force several K pieces (including one past the end of K) and check the reduce."""
    import triton

    from emberserve.model import quant

    g = torch.Generator().manual_seed(11)
    m, n, k = 3, 40, 300
    q, s = quantize_int8_weight(torch.randn(n, k, generator=g) * 0.1)
    x = torch.randn(m, k, generator=g).half()
    b = torch.randn(n, generator=g).half()
    want = int8_gemm_torch(x, q, s, b).float()
    assert quant.split_k(1, 3584, 18944) == 4 and quant.split_k(1, 4608, 3584) == 4  # 224, 288 programs
    assert quant.split_k(4096, 4608, 3584) == 1 and quant.split_k(1, 96, 64) == 1
    for splits in (2, 4, 8):  # BK=64 -> 5 K tiles: 8 splits leaves 3 empty programs
        out = torch.empty((m, n), dtype=torch.float16)
        ws = torch.empty((splits, m, n), dtype=torch.float32)
        grid = (1, triton.cdiv(n, 64), splits)
        quant._kernel()[grid](x, q, s, b, out, ws, m, n, k, x.stride(0), q.stride(0), out.stride(0),
                              M_BUCKET=4, SPLIT_K=splits, HAS_BIAS=True, BM=16, BN=64, BK=64,
                              num_warps=4, num_stages=2)
        quant._reduce()[(1, triton.cdiv(n, 128))](ws, s, b, out, m, n, out.stride(0), SPLIT_K=splits,
                                                   HAS_BIAS=True, BM=16, BN=128, num_warps=4)
        torch.testing.assert_close(out.float(), want, atol=2e-2, rtol=2e-2)


def test_module_uses_the_kernel_under_the_interpreter():
    lin = torch.nn.Linear(64, 48).half()
    q = Int8Linear.from_linear(lin)
    x = torch.randn(3, 64).half()
    torch.testing.assert_close(q(x).float(), int8_gemm_torch(x, q.weight_q, q.scale, q.bias).float(),
                               atol=2e-2, rtol=2e-2)
