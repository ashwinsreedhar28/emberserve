"""PagedKVCache: physical block storage for every layer's K and V.

Each layer owns two tensors of shape [num_blocks, block_size, num_kv_heads, head_dim].
A token lives at physical slot `block_id * block_size + offset`. The BlockManager decides
which slot a token gets; this class only stores and gathers.
"""

from __future__ import annotations

import torch
from torch import Tensor

from emberserve.config import ModelConfig


class PagedKVCache:
    """Per-layer paged K/V storage with scatter-write and block-table gather."""

    def __init__(self, config: ModelConfig, num_blocks: int, block_size: int,
                 device: str | torch.device, dtype: torch.dtype) -> None:
        assert num_blocks > 0 and block_size > 0, (num_blocks, block_size)
        self.config = config
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        shape = (num_blocks, block_size, self.num_kv_heads, self.head_dim)
        self.k_cache: list[Tensor] = [torch.zeros(shape, device=self.device, dtype=dtype)
                                      for _ in range(self.num_layers)]
        self.v_cache: list[Tensor] = [torch.zeros(shape, device=self.device, dtype=dtype)
                                      for _ in range(self.num_layers)]

    # ---- geometry --------------------------------------------------------------
    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    @staticmethod
    def num_blocks_for_bytes(config: ModelConfig, block_size: int, bytes_available: int,
                             dtype: torch.dtype) -> int:
        """How many blocks (K+V, all layers) fit in `bytes_available`."""
        return int(bytes_available) // (config.kv_bytes_per_token(dtype) * block_size)

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k_cache + self.v_cache)

    # ---- write -------------------------------------------------------------------
    def write(self, layer_idx: int, k: Tensor, v: Tensor, slot_mapping: Tensor) -> None:
        """Scatter this step's k/v ([N, Hkv, D]) into physical slots ([N] int64)."""
        slots = slot_mapping.to(device=self.device, dtype=torch.long)
        expected = (slots.numel(), self.num_kv_heads, self.head_dim)
        assert tuple(k.shape) == tuple(v.shape) == expected, (k.shape, v.shape, expected)
        flat = (self.num_slots, self.num_kv_heads, self.head_dim)
        self.k_cache[layer_idx].view(flat).index_copy_(0, slots, k.to(self.dtype))
        self.v_cache[layer_idx].view(flat).index_copy_(0, slots, v.to(self.dtype))

    # ---- gather ------------------------------------------------------------------
    def gather(self, layer_idx: int, block_table: list[int] | Tensor,
               context_len: int) -> tuple[Tensor, Tensor]:
        """K/V of one sequence in logical order: ([ctx, Hkv, D], [ctx, Hkv, D]).

        Only the first `blocks_needed(context_len)` table entries are read, so a row of a
        -1-padded batch table is accepted.
        """
        n = self.blocks_needed(context_len)
        table = torch.as_tensor(block_table, dtype=torch.long, device=self.device)[:n]
        assert table.numel() == n and bool((table >= 0).all()), "block table too short"

        def one(store: list[Tensor]) -> Tensor:
            blocks = store[layer_idx][table]  # [n, block_size, Hkv, D]
            return blocks.reshape(-1, self.num_kv_heads, self.head_dim)[:context_len]

        return one(self.k_cache), one(self.v_cache)

    def gather_batch(self, layer_idx: int, block_tables: Tensor,
                     context_lens: list[int] | Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Padded K/V for a batch, plus a validity mask.

        block_tables: [B, max_blocks] int32, padded with -1.
        Returns (k [B, max_ctx, Hkv, D], v [B, max_ctx, Hkv, D], valid [B, max_ctx] bool).
        Positions beyond each sequence's context_len hold stale data and are False in
        `valid`; the caller must mask them.
        """
        if isinstance(context_lens, Tensor):
            ctx = context_lens.to(device=self.device, dtype=torch.long)
            max_ctx = int(ctx.max())
        else:
            max_ctx = max(context_lens)
            ctx = torch.tensor(context_lens, dtype=torch.long, device=self.device)
        n = self.blocks_needed(max_ctx)
        assert block_tables.shape[1] >= n, (block_tables.shape, max_ctx)
        idx = block_tables[:, :n].to(device=self.device, dtype=torch.long).clamp_min(0)
        batch = idx.shape[0]

        def one(store: list[Tensor]) -> Tensor:
            blocks = store[layer_idx][idx]  # [B, n, block_size, Hkv, D]
            flat = blocks.reshape(batch, n * self.block_size, self.num_kv_heads, self.head_dim)
            return flat[:, :max_ctx]

        valid = torch.arange(max_ctx, device=self.device)[None, :] < ctx[:, None]
        return one(self.k_cache), one(self.v_cache), valid

    # ---- lifecycle ---------------------------------------------------------------
    def reset(self) -> None:
        for t in self.k_cache + self.v_cache:
            t.zero_()

    def __repr__(self) -> str:
        return (f"PagedKVCache(layers={self.num_layers}, num_blocks={self.num_blocks}, "
                f"block_size={self.block_size}, kv_heads={self.num_kv_heads}, "
                f"head_dim={self.head_dim}, dtype={self.dtype}, device={self.device}, "
                f"bytes={self.memory_bytes()})")


class PagedLatentCache:
    """Paged storage for multi-head latent attention: one row per token per layer holding
    the compressed KV latent and the shared rope key (`kv_lora_rank + qk_rope_head_dim`,
    576 for DeepSeek-V2-Lite and Moonlight) instead of K and V per head. Same block /
    slot geometry as `PagedKVCache`, so the BlockManager and block tables are unchanged."""

    def __init__(self, config: ModelConfig, num_blocks: int, block_size: int,
                 device: str | torch.device, dtype: torch.dtype) -> None:
        assert config.mla is not None, "PagedLatentCache needs an MLA model config"
        assert num_blocks > 0 and block_size > 0, (num_blocks, block_size)
        self.config = config
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.latent_dim = config.mla.latent_dim
        shape = (num_blocks, block_size, self.latent_dim)
        self.latent: list[Tensor] = [torch.zeros(shape, device=self.device, dtype=dtype)
                                     for _ in range(self.num_layers)]

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.latent)

    def write(self, layer_idx: int, latent: Tensor, slot_mapping: Tensor) -> None:
        """Scatter this step's latent rows ([N, latent_dim]) into physical slots ([N])."""
        slots = slot_mapping.to(device=self.device, dtype=torch.long)
        assert tuple(latent.shape) == (slots.numel(), self.latent_dim), latent.shape
        self.latent[layer_idx].view(self.num_slots, self.latent_dim).index_copy_(
            0, slots, latent.to(self.dtype))

    def gather(self, layer_idx: int, block_table: list[int] | Tensor, context_len: int) -> Tensor:
        """One sequence's latent rows in logical order: [ctx, latent_dim]."""
        n = self.blocks_needed(context_len)
        table = torch.as_tensor(block_table, dtype=torch.long, device=self.device)[:n]
        assert table.numel() == n and bool((table >= 0).all()), "block table too short"
        blocks = self.latent[layer_idx][table]  # [n, block_size, L]
        return blocks.reshape(-1, self.latent_dim)[:context_len]
