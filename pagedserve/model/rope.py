"""Rotary position embedding (RoPE) in the HF Qwen2 / Llama convention.

Non-interleaved "rotate_half" layout: dimension i is paired with dimension i + D/2.
Tables are built lazily per (device) and cached; the rotation itself runs in float32
and is cast back to the input dtype, which is what makes it match HF at fp32 tolerances.
"""

from __future__ import annotations

import torch
from torch import nn

from pagedserve.model import ops


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1) over the two halves of the last dim."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate `x: [num_tokens, H, D]` by per-token `cos`/`sin` of shape `[num_tokens, D]`.

    Math is done in float32; the result is returned in `x.dtype`.
    """
    xf = x.float()
    c = cos[:, None, :].float()
    s = sin[:, None, :].float()
    return (xf * c + rotate_half(xf) * s).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables `[max_position_embeddings, head_dim]` on first use."""

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def inv_freq(self, device: torch.device | None = None) -> torch.Tensor:
        """`1 / base^(2i/D)` for i in [0, D/2), float32."""
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim
        return 1.0 / (self.base ** exponent)

    def _build_tables(self, device: torch.device) -> None:
        positions = torch.arange(self.max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, self.inv_freq(device))  # [max_pos, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, D]
        self._cos = emb.cos()
        self._sin = emb.sin()

    def tables(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin rows `[num_tokens, D]` (float32) for the given int64 positions."""
        if self._cos is None or self._cos.device != positions.device:
            self._build_tables(positions.device)
        assert self._cos is not None and self._sin is not None
        return self._cos[positions], self._sin[positions]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate `q: [N, H, D]` and `k: [N, Hkv, D]` at `positions: [N]`."""
        if self._cos is None or self._cos.device != positions.device:
            self._build_tables(positions.device)
        assert self._cos is not None and self._sin is not None
        return ops.rope(q, k, self._cos, self._sin, positions)
