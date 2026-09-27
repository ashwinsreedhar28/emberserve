"""Fused Triton ops on CUDA vs the PyTorch references, then the whole model with the fused
path on and off (`PAGEDSERVE_FUSED_OPS`)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from pagedserve.attn.naive import NaiveAttentionBackend  # noqa: E402
from pagedserve.model import ops  # noqa: E402
from pagedserve.model import ops_triton as kt  # noqa: E402
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic  # noqa: E402
from tests.test_model import make_prefill_meta  # noqa: E402
from pagedserve.config import ModelConfig  # noqa: E402

DEV = "cuda"
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _rand(*shape, dtype, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(dtype).to(DEV)


def _tol(dtype):
    return {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float16: dict(atol=1e-3, rtol=1e-3),
            torch.bfloat16: dict(atol=1e-2, rtol=1e-2)}[dtype]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", [1, 37, 300])
def test_rmsnorm_and_residual(dtype, rows):
    x = _rand(rows, 896, dtype=dtype, seed=1, scale=3.0)
    r = _rand(rows, 896, dtype=dtype, seed=2, scale=4.0)
    w = _rand(896, dtype=dtype, seed=3, scale=0.3) + 1.0
    torch.testing.assert_close(kt.rmsnorm_triton(x, w, 1e-6), ops.rmsnorm_torch(x, w, 1e-6),
                               **_tol(dtype))
    got, got_res = kt.fused_add_rmsnorm_triton(x, r, w, 1e-6)
    want, want_res = ops.fused_add_rmsnorm_torch(x, r, w, 1e-6)
    torch.testing.assert_close(got_res, want_res, atol=0, rtol=0)
    torch.testing.assert_close(got, want, **_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_rope(dtype):
    h, hk, d, n = 14, 2, 64, 50
    positions = torch.randint(0, 4096, (n,), generator=torch.Generator().manual_seed(9)).to(DEV)
    inv = 1.0 / (1e6 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    emb = torch.cat([torch.outer(torch.arange(4096, dtype=torch.float32), inv)] * 2, dim=-1)
    cos, sin = emb.cos().to(DEV), emb.sin().to(DEV)
    qkv = _rand(n, (h + 2 * hk) * d, dtype=dtype, seed=6)
    q = qkv[:, :h * d].view(n, h, d)
    k = qkv[:, h * d:(h + hk) * d].view(n, hk, d)
    want_q, want_k = ops.rope_torch(q.clone(), k.clone(), cos, sin, positions)
    got_q, got_k = kt.rope_triton(q, k, cos, sin, positions)
    torch.testing.assert_close(got_q, want_q, **_tol(dtype))
    torch.testing.assert_close(got_k, want_k, **_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_and_mul(dtype):
    x = _rand(33, 2 * 4864, dtype=dtype, seed=7, scale=2.0)
    torch.testing.assert_close(kt.silu_and_mul_triton(x), ops.silu_and_mul_torch(x), **_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_model_forward_fused_vs_torch(dtype, monkeypatch):
    """Whole tiny model, prefill of two sequences, fused ops on vs off: same logits."""
    cfg = ModelConfig.tiny(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           hidden_size=256, intermediate_size=512)
    model = Qwen2ForCausalLM(cfg)
    reset_parameters_deterministic(model, 0)
    model = model.to(DEV, dtype).eval()
    ids = torch.randint(0, cfg.vocab_size, (30,), generator=torch.Generator().manual_seed(1)).to(DEV)
    meta = make_prefill_meta([0, 1], [12, 18])
    meta.positions = meta.positions.to(DEV)

    def run() -> torch.Tensor:
        backend = NaiveAttentionBackend(cfg, device=DEV, dtype=dtype)
        with torch.inference_mode():
            return model.forward_logits_all(ids, backend, meta).float().cpu()

    monkeypatch.setenv("PAGEDSERVE_FUSED_OPS", "0")
    ref = run()
    monkeypatch.setenv("PAGEDSERVE_FUSED_OPS", "1")
    assert ops.fused_enabled(torch.zeros(1, device=DEV))
    fused = run()
    tol = dict(atol=1e-4, rtol=1e-4) if dtype == torch.float32 else dict(atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(fused, ref, **tol)
