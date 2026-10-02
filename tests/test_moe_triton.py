"""Fused MoE (grouped GEMM) on the CPU interpreter vs the per-expert loop reference.
Same module rules as tests/test_paged_triton.py."""

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

from emberserve.model.moe import DeepseekMoE, moe_forward_reference  # noqa: E402
from emberserve.model.moe_triton import fused_moe_forward, moe_align  # noqa: E402
from tests.test_moe import H, cfg, seed_module  # noqa: E402

torch.set_num_threads(2)


def test_align_layout():
    ids = torch.tensor([[1, 3], [3, 0], [1, 1], [3, 2]])  # N=4, k=2, E=4
    sorted_ids, block_expert, num_valid = moe_align(ids, num_experts=4, block_m=4)
    assert num_valid == 8
    # expert 0: 1 assignment -> 1 block; 1: 3 -> 1 block; 2: 1 -> 1; 3: 3 -> 1  => 4 blocks used
    assert block_expert.tolist()[:4] == [0, 1, 2, 3]
    assert all(e == 4 for e in block_expert.tolist()[4:])
    flat = ids.reshape(-1)
    for b, e in enumerate(block_expert.tolist()[:4]):
        rows = sorted_ids[b * 4:(b + 1) * 4].tolist()
        real = [r for r in rows if r < num_valid]
        assert all(flat[r] == e for r in real), (b, e, rows)
        assert len(real) == int((flat == e).sum())
    assert sorted(r for r in sorted_ids.tolist() if r < num_valid) == list(range(8))


@pytest.mark.parametrize("n", [1, 7, 40])
@pytest.mark.parametrize("groups", [(1, 1), (2, 1)])
def test_fused_matches_loop(n, groups):
    moe = seed_module(DeepseekMoE(cfg(n_group=groups[0], topk_group=groups[1], n_shared_experts=0)), 30 + n)
    x = torch.randn(n, H, generator=torch.Generator().manual_seed(n))
    idx, w = moe.gate(x)
    got = fused_moe_forward(x, idx, w, moe.experts_gate_up, moe.experts_down)
    want = moe.forward_loop(x, idx, w)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    # and through the module (fused path selected under the interpreter), shared experts on
    moe2 = seed_module(DeepseekMoE(cfg(n_shared_experts=1)), 60 + n)
    torch.testing.assert_close(moe2(x), moe_forward_reference(moe2, x), atol=1e-4, rtol=1e-4)


def test_many_experts_few_tokens():
    """Most experts empty: the padded layout is mostly sentinel rows and exit-early blocks."""
    moe = seed_module(DeepseekMoE(cfg(n_routed_experts=32, num_experts_per_tok=6, n_shared_experts=0)), 9)
    x = torch.randn(2, H, generator=torch.Generator().manual_seed(9))
    idx, w = moe.gate(x)
    got = fused_moe_forward(x, idx, w, moe.experts_gate_up, moe.experts_down)
    torch.testing.assert_close(got, moe.forward_loop(x, idx, w), atol=1e-4, rtol=1e-4)


def _gate_sorted(idx, w):
    """topk order is unspecified (torch uses sorted=False): sort each row by expert id."""
    order = idx.argsort(dim=1)
    return idx.gather(1, order), w.gather(1, order)


def _assert_gate_equal(got, want):
    gi, gw = _gate_sorted(*got)
    wi, ww = _gate_sorted(*want)
    assert gi.tolist() == wi.tolist()
    torch.testing.assert_close(gw, ww, atol=1e-5, rtol=1e-5)  # sigmoid differs by an ulp


@pytest.mark.parametrize("n", [1, 5, 33])
@pytest.mark.parametrize("norm", [True, False])
def test_topk_gate_matches_torch(n, norm):
    from emberserve.model.moe_triton import topk_gate

    moe = seed_module(DeepseekMoE(cfg(n_shared_experts=0, norm_topk_prob=norm)), 70 + n)
    x = torch.randn(n, H, generator=torch.Generator().manual_seed(n))
    logits = torch.nn.functional.linear(x, moe.gate.weight)
    want = moe.gate.select(logits)
    got = topk_gate(logits, moe.gate.e_score_correction_bias, moe.config.num_experts_per_tok,
                    norm, moe.config.routed_scaling_factor)
    assert got[0].dtype == torch.int64 and got[1].dtype == torch.float32
    _assert_gate_equal(got, want)
    # the module picks the kernel under the interpreter
    _assert_gate_equal(moe.gate(x), want)


@pytest.mark.parametrize("n", [1, 7, 40, 200])
def test_align_kernel_matches_torch(n):
    """The single-launch alignment kernel produces exactly the torch layout."""
    e, k = 16, 4
    ids = torch.randint(0, e, (n, k), generator=torch.Generator().manual_seed(n))
    a = moe_align(ids, num_experts=e, block_m=16, use_kernel=True)
    b = moe_align(ids, num_experts=e, block_m=16, use_kernel=False)
    assert a[2] == b[2]
    assert a[0].tolist() == b[0].tolist()
    assert a[1].tolist() == b[1].tolist()


def test_align_kernel_many_experts_few_tokens():
    ids = torch.tensor([[63, 0], [63, 5]])
    a = moe_align(ids, num_experts=64, block_m=16, use_kernel=True)
    b = moe_align(ids, num_experts=64, block_m=16, use_kernel=False)
    assert a[0].tolist() == b[0].tolist() and a[1].tolist() == b[1].tolist()
