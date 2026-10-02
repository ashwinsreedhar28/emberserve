"""Reference attention backend: one growing K/V tensor per (sequence, layer).

No paging, no block tables, `torch.cat` on every append. It exists to define correct
behaviour for the paged backends and to run the model on CPU in tests.
"""

from __future__ import annotations

import torch

from emberserve.attn.base import AttentionBackend, AttnMetadata, causal_softmax_attention
from emberserve.config import ModelConfig


class NaiveAttentionBackend(AttentionBackend):
    """Per-sequence KV history kept in a dict keyed by `(seq_id, layer_idx)`."""

    def __init__(self, config: ModelConfig, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        # (seq_id, layer_idx) -> (K [T, Hkv, D], V [T, Hkv, D])
        self._cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                meta: AttnMetadata) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        start = 0
        for i, seq_id in enumerate(meta.seq_ids):
            qlen = meta.query_lens[i]
            end = start + qlen
            key = (seq_id, layer_idx)
            k_new, v_new = k[start:end], v[start:end]
            cached = 0 if not meta.num_cached_tokens else meta.num_cached_tokens[i]
            fresh = key not in self._cache or (meta.is_prefill and cached == 0)
            if fresh:
                k_all, v_all = k_new, v_new
            else:
                k_old, v_old = self._cache[key]
                k_all = torch.cat((k_old, k_new), dim=0)
                v_all = torch.cat((v_old, v_new), dim=0)
            self._cache[key] = (k_all, v_all)
            assert k_all.shape[0] == meta.context_lens[i], (
                f"seq {seq_id} layer {layer_idx}: cache has {k_all.shape[0]} tokens, "
                f"metadata says context_len={meta.context_lens[i]}")
            outputs.append(causal_softmax_attention(q[start:end], k_all, v_all, qlen))
            start = end
        return torch.cat(outputs, dim=0)

    def free_sequence(self, seq_id: int) -> None:
        for key in [key for key in self._cache if key[0] == seq_id]:
            del self._cache[key]

    def reset(self) -> None:
        self._cache.clear()

    def num_cached_tokens(self, seq_id: int, layer_idx: int = 0) -> int:
        """Tokens currently stored for `seq_id` at `layer_idx` (0 if none)."""
        entry = self._cache.get((seq_id, layer_idx))
        return 0 if entry is None else int(entry[0].shape[0])
