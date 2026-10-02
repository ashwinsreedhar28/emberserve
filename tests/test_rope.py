"""RoPE tests: identity at position 0, norm preservation, and agreement with an
independent complex-number reference of the rotate_half (non-interleaved) convention."""

from __future__ import annotations

import torch

from emberserve.model.rope import RotaryEmbedding, apply_rotary, rotate_half

torch.set_num_threads(2)

D = 16
MAX_POS = 128
BASE = 10_000.0


def _complex_reference(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate `x: [N, H, D]` by treating dims (i, i + D/2) as (re, im) and multiplying
    by exp(i * pos * inv_freq_i). Written independently of `apply_rotary`."""
    n, h, d = x.shape
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float64) / d))  # [D/2]
    angles = positions.double()[:, None] * inv_freq[None, :]  # [N, D/2]
    rot = torch.polar(torch.ones_like(angles), angles)  # e^{i*angle}, [N, D/2]
    z = torch.complex(x[..., :half].double(), x[..., half:].double())  # [N, H, D/2]
    zr = z * rot[:, None, :]
    return torch.cat((zr.real, zr.imag), dim=-1).to(x.dtype)


def test_rotate_half() -> None:
    x = torch.arange(6.0).view(1, 1, 6)
    torch.testing.assert_close(rotate_half(x), torch.tensor([[[-3.0, -4.0, -5.0, 0.0, 1.0, 2.0]]]))


def test_position_zero_is_identity() -> None:
    rope = RotaryEmbedding(D, MAX_POS, BASE)
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(5, 4, D, generator=gen)
    k = torch.randn(5, 2, D, generator=gen)
    q_out, k_out = rope(q, k, torch.zeros(5, dtype=torch.int64))
    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)


def test_preserves_per_head_norm() -> None:
    rope = RotaryEmbedding(D, MAX_POS, BASE)
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(32, 4, D, generator=gen)
    k = torch.randn(32, 2, D, generator=gen)
    positions = torch.randint(0, MAX_POS, (32,), generator=gen)
    q_out, k_out = rope(q, k, positions)
    torch.testing.assert_close(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=0)
    torch.testing.assert_close(k_out.norm(dim=-1), k.norm(dim=-1), atol=1e-5, rtol=0)


def test_matches_complex_reference() -> None:
    rope = RotaryEmbedding(D, MAX_POS, BASE)
    gen = torch.Generator().manual_seed(2)
    q = torch.randn(40, 4, D, generator=gen)
    k = torch.randn(40, 2, D, generator=gen)
    positions = torch.randint(0, MAX_POS, (40,), generator=gen)
    q_out, k_out = rope(q, k, positions)
    torch.testing.assert_close(q_out, _complex_reference(q, positions, BASE), atol=1e-5, rtol=0)
    torch.testing.assert_close(k_out, _complex_reference(k, positions, BASE), atol=1e-5, rtol=0)


def test_apply_rotary_matches_module_and_keeps_dtype() -> None:
    rope = RotaryEmbedding(D, MAX_POS, BASE)
    gen = torch.Generator().manual_seed(3)
    positions = torch.randint(0, MAX_POS, (10,), generator=gen)
    x = torch.randn(10, 3, D, generator=gen)
    cos, sin = rope.tables(positions)
    assert cos.shape == sin.shape == (10, D) and cos.dtype == torch.float32
    q_out, _ = rope(x, x[:, :1], positions)
    torch.testing.assert_close(apply_rotary(x, cos, sin), q_out)
    half_out, _ = rope(x.half(), x.half()[:, :1], positions)
    assert half_out.dtype == torch.float16
    torch.testing.assert_close(half_out.float(), q_out, atol=2e-2, rtol=0)


def test_rotation_is_relative() -> None:
    """q.k after RoPE depends only on the position difference (m - n)."""
    rope = RotaryEmbedding(D, MAX_POS, BASE)
    gen = torch.Generator().manual_seed(4)
    q = torch.randn(1, 1, D, generator=gen)
    k = torch.randn(1, 1, D, generator=gen)

    def score(m: int, n: int) -> torch.Tensor:
        qm, _ = rope(q, q, torch.tensor([m]))
        kn, _ = rope(k, k, torch.tensor([n]))
        return (qm * kn).sum()

    torch.testing.assert_close(score(10, 3), score(27, 20), atol=1e-5, rtol=0)
    torch.testing.assert_close(score(5, 5), score(60, 60), atol=1e-5, rtol=0)
