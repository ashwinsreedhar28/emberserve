"""Fused MoE grouped GEMM on CUDA vs the per-expert loop, at Moonlight's expert geometry."""

from __future__ import annotations

import dataclasses

import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from pagedserve.config import MoEConfig  # noqa: E402
from pagedserve.model.moe import DeepseekMoE  # noqa: E402
from pagedserve.model.moe_triton import fused_moe_forward, moe_align, topk_gate  # noqa: E402

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
        # The shared experts add the routed output in their down GEMM's epilogue (addmm:
        # one rounding), so the manual path must do the same to be bit-identical; adding
        # the two bf16 results afterwards rounds twice (a 1-ULP difference on ~30% of
        # elements on an RTX 4090).
        routed = fused_moe_forward(x, idx, w, moe.experts_gate_up, moe.experts_down)
        manual = moe.shared_experts(x, add_to=routed).float()
    torch.testing.assert_close(via_module, manual, atol=0, rtol=0)


@pytest.mark.parametrize("n", [1, 8, 200, 682, 1500])
def test_align_kernel_matches_torch_layout(n):
    """Single-launch alignment == the sort-based torch layout, bit for bit (E=64, k=6; 682*6
    is the last size the kernel takes by default, 1500*6 is above it and forced)."""
    ids = torch.randint(0, 64, (n, 6), generator=torch.Generator().manual_seed(n)).to(DEV)
    a = moe_align(ids, num_experts=64, block_m=16, use_kernel=True)
    b = moe_align(ids, num_experts=64, block_m=16, use_kernel=False)
    assert a[2] == b[2]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("n", [1, 8, 200, 4096])
@pytest.mark.parametrize("norm", [True, False])
def test_topk_gate_matches_torch(n, norm):
    moe = _moe(torch.bfloat16, seed=3)
    moe.config = dataclasses.replace(moe.config, norm_topk_prob=norm)
    moe.gate.config = moe.config
    x = (torch.randn(n, 2048, generator=torch.Generator().manual_seed(n)) * 0.5).to(DEV, torch.bfloat16)
    logits = torch.nn.functional.linear(x.float(), moe.gate.weight.float())
    wi, ww = moe.gate.select(logits)
    gi, gw = topk_gate(logits, moe.gate.e_score_correction_bias, 6, norm, 2.446)
    oi, ow = wi.argsort(1), gi.argsort(1)
    assert torch.equal(wi.gather(1, oi), gi.gather(1, ow))
    torch.testing.assert_close(gw.gather(1, ow), ww.gather(1, oi), atol=1e-5, rtol=1e-5)
    # module path on CUDA is the kernel
    mi, mw = moe.gate(x)
    assert torch.equal(mi.gather(1, mi.argsort(1)), gi.gather(1, ow))
