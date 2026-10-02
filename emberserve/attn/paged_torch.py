"""Pure-PyTorch PagedAttention backend.

Reference implementation of the paged KV path: this step's k/v are scattered into the
PagedKVCache through `meta.slot_mapping`, then attention reads them back through the
per-sequence block tables. Prefill runs one `causal_softmax_attention` per sequence over
its gathered context; decode is a single batched gather + masked softmax. The GPU
flash-attn backend must reproduce these numbers.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from emberserve.attn.base import AttentionBackend, AttnMetadata, causal_softmax_attention
from emberserve.config import ModelConfig
from emberserve.kv.block_manager import BlockManager
from emberserve.kv.cache import PagedKVCache


def build_block_tables_tensor(tables: list[list[int]], device: str | torch.device) -> Tensor:
    """Stack per-sequence block tables into [B, max_blocks] int32, right-padded with -1."""
    max_blocks = max((len(t) for t in tables), default=0)
    out = torch.full((len(tables), max_blocks), -1, dtype=torch.int32)
    for i, table in enumerate(tables):
        if table:
            out[i, : len(table)] = torch.tensor(table, dtype=torch.int32)
    return out.to(device)


def build_slot_mapping(block_manager: BlockManager, seq_ids: list[int],
                       start_positions: list[int], num_tokens: list[int] | int,
                       device: str | torch.device) -> Tensor:
    """Physical slots for each sequence's tokens in packed order: [N] int64.

    `num_tokens` is per sequence, or one int applied to every sequence (decode: 1).
    """
    if isinstance(num_tokens, int):
        num_tokens = [num_tokens] * len(seq_ids)
    slots: list[int] = []
    for sid, start, n in zip(seq_ids, start_positions, num_tokens, strict=True):
        slots.extend(block_manager.slot_mapping(sid, start, n))
    return torch.tensor(slots, dtype=torch.int64, device=device)


class PagedTorchAttentionBackend(AttentionBackend):
    """PagedAttention over a PagedKVCache, computed with plain tensor ops."""

    def __init__(self, config: ModelConfig, cache: PagedKVCache) -> None:
        assert cache.num_kv_heads == config.num_key_value_heads
        assert cache.head_dim == config.head_dim
        self.config = config
        self.cache = cache
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, layer_idx: int, q: Tensor, k: Tensor, v: Tensor,
                meta: AttnMetadata) -> Tensor:
        assert meta.slot_mapping is not None and meta.block_tables is not None \
            and meta.block_size is not None, "paged backend needs slot_mapping/block_tables/block_size"
        assert meta.block_size == self.cache.block_size, (meta.block_size, self.cache.block_size)
        assert q.shape == (meta.num_tokens, self.num_heads, self.head_dim), q.shape
        assert meta.block_tables.shape[0] == meta.num_seqs, meta.block_tables.shape
        self.cache.write(layer_idx, k, v, meta.slot_mapping)
        if meta.is_prefill:
            return self._prefill(layer_idx, q, meta)
        return self._decode(layer_idx, q, meta)

    def _prefill(self, layer_idx: int, q: Tensor, meta: AttnMetadata) -> Tensor:
        """Per sequence: gather its full context and run the causal reference.

        Queries are the last query_lens[i] of context_lens[i] keys, so a cached prefix
        (num_cached_tokens > 0) needs no special handling here.
        """
        cu = meta.cu_seqlens_q.tolist()
        outs: list[Tensor] = []
        for i in range(meta.num_seqs):
            q_i = q[cu[i]:cu[i + 1]]
            k_ctx, v_ctx = self.cache.gather(layer_idx, meta.block_tables[i], meta.context_lens[i])
            outs.append(causal_softmax_attention(q_i, k_ctx, v_ctx, meta.query_lens[i]))
        return torch.cat(outs, dim=0)

    def _decode(self, layer_idx: int, q: Tensor, meta: AttnMetadata) -> Tensor:
        """One query per sequence, fully batched over the padded gathered context."""
        assert all(n == 1 for n in meta.query_lens), "decode step expects query_lens == 1"
        k, v, valid = self.cache.gather_batch(layer_idx, meta.block_tables, meta.context_lens)
        groups = self.num_heads // self.num_kv_heads
        kf = k.float().repeat_interleave(groups, dim=2)  # [B, ctx, H, D]
        vf = v.float().repeat_interleave(groups, dim=2)
        scores = torch.einsum("bhd,bkhd->bhk", q.float(), kf) * self.scale  # [B, H, ctx]
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhk,bkhd->bhd", probs, vf)  # [B, H, D]
        return out.to(q.dtype)

    def free_sequence(self, seq_id: int) -> None:
        """No-op: the BlockManager owns block lifetime; slots are simply overwritten."""

    def reset(self) -> None:
        self.cache.reset()
