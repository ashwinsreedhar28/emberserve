"""Fused MoE grouped GEMM on CUDA vs the per-expert loop, at Moonlight's expert geometry."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from pagedserve.config import MoEConfig  # noqa: E402
from pagedserve.model.moe import DeepseekMoE  # noqa: E402
from pagedserve.model.moe_triton import fused_moe_forward  # noqa: E402

DEV = "cuda"


def _moe(dtype, seed=0) -> DeepseekMoE:
    c = MoEConfig(hidden_size=2048, moe_intermediate_size=1408, n_routed_experts=64,
                  num_experts_per_tok=6, n_shared_experts=2, routed_scaling_factor=2.446)
    m = DeepseekMoE(c)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (0.3 if "bias" in name else 0.02))
    return m.to(DEV, dtype).eval()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n", [1, 8, 200])
def test_fused_matches_loop(dtype, n):
    moe = _moe(dtype)
    x = (torch.randn(n, 2048, generator=torch.Generator().manual_seed(n)) * 0.5).to(DEV, dtype)
    idx, w = moe.gate(x)
    with torch.no_grad():
        got = fused_moe_forward(x, idx, w, moe.experts_gate_up, moe.experts_down).float()
        want = moe.forward_loop(x, idx, w).float()
    tol = dict(atol=2e-2, rtol=2e-2) if dtype == torch.float16 else dict(atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(got, want, **tol)


def test_fused_is_the_module_path_on_cuda():
    moe = _moe(torch.bfloat16, seed=1)
    x = torch.randn(5, 2048, device=DEV, dtype=torch.bfloat16)
    idx, w = moe.gate(x)
    with torch.no_grad():
        via_module = moe(x).float()
        manual = (fused_moe_forward(x, idx, w, moe.experts_gate_up, moe.experts_down)
                  + moe.shared_experts(x)).float()
    torch.testing.assert_close(via_module, manual, atol=0, rtol=0)
