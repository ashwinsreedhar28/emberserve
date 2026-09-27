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
if _HAS_CUDA and os.environ.get("PAGEDSERVE_FORCE_INTERPRETER") != "1":
    pytest.skip("CUDA present: interpreter tests skipped", allow_module_level=True)
os.environ["TRITON_INTERPRET"] = "1"  # noqa: E402

import torch  # noqa: E402

pytest.importorskip("triton")

from pagedserve.model.quant import Int8Linear, int8_gemm, int8_gemm_torch, quantize_int8_weight  # noqa: E402

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


def test_module_uses_the_kernel_under_the_interpreter():
    lin = torch.nn.Linear(64, 48).half()
    q = Int8Linear.from_linear(lin)
    x = torch.randn(3, 64).half()
    torch.testing.assert_close(q(x).float(), int8_gemm_torch(x, q.weight_q, q.scale, q.bias).float(),
                               atol=2e-2, rtol=2e-2)
