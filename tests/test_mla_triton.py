"""The Triton MLA decode kernel on the CPU interpreter vs `mla_attention_absorbed`.
Same module rules as tests/test_paged_triton.py (TRITON_INTERPRET set before triton is
imported; skipped on CUDA machines, where tests/test_mla_triton_gpu.py covers it)."""

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

from pagedserve.attn.mla_torch import mla_attention_absorbed  # noqa: E402
from pagedserve.attn.mla_triton import mla_decode  # noqa: E402

torch.set_num_threads(2)
H, DL, DR, DN, DV = 4, 32, 16, 16, 16  # tiny latent geometry (DL, DR powers of two)
BLOCK = 16


def _setup(seed: int, batch: int, ctx_lens: list[int]):
    g = torch.Generator().manual_seed(seed)
    max_blocks = max((c + BLOCK - 1) // BLOCK for c in ctx_lens)
    num_blocks = batch * max_blocks + 3
    latent = torch.randn(num_blocks, BLOCK, DL + DR, generator=g)
    # scrambled block tables
    perm = torch.randperm(num_blocks, generator=g).tolist()
    tables = torch.zeros(batch, max_blocks, dtype=torch.int32)
    for b in range(batch):
        for j in range((ctx_lens[b] + BLOCK - 1) // BLOCK):
            tables[b, j] = perm.pop()
    q_abs = torch.randn(batch, H, DL + DR, generator=g)
    w_uk = torch.randn(H, DN, DL, generator=g) * 0.2
    w_uv = torch.randn(H, DV, DL, generator=g) * 0.2
    return latent, tables, torch.tensor(ctx_lens, dtype=torch.int32), q_abs, w_uk, w_uv


def _reference(latent, tables, ctx_lens, q_abs, w_uv):
    """Per sequence: gather its rows, run the absorbed reference with an identity W_UK so
    the kernel's raw `P . c` output is compared directly."""
    outs = []
    eye = torch.eye(DL).expand(H, DL, DL)
    for b in range(q_abs.shape[0]):
        n = int(ctx_lens[b])
        rows = latent[tables[b, :(n + BLOCK - 1) // BLOCK].long()].reshape(-1, DL + DR)[:n]
        q_c, q_pe = q_abs[b, :, :DL], q_abs[b, :, DL:]
        out_c = mla_attention_absorbed(q_c[None], q_pe[None], rows, eye, eye, 0.3, DL)[0]  # [H, DL]
        outs.append(out_c)
    return torch.stack(outs)


@pytest.mark.parametrize("ctx_lens", [[1], [16], [17, 5, 40], [33, 64, 3, 50]])
def test_kernel_matches_reference(ctx_lens):
    latent, tables, ctx, q_abs, _, _ = _setup(1, len(ctx_lens), ctx_lens)
    got = mla_decode(q_abs, latent, tables, ctx, 0.3, DL, num_splits=1)
    want = _reference(latent, tables, ctx, q_abs, None)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("splits", [2, 3])
def test_split_k_matches(splits):
    ctx_lens = [70, 9, 48]
    latent, tables, ctx, q_abs, _, _ = _setup(2, 3, ctx_lens)
    got = mla_decode(q_abs, latent, tables, ctx, 0.3, DL, num_splits=splits)
    want = _reference(latent, tables, ctx, q_abs, None)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
