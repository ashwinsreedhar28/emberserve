"""Attention backend interface.

The model is written against this interface and never touches the cache directly.
All backends use the PACKED token layout: every tensor that is "per token" has shape
[num_tokens, ...] where num_tokens is the sum of query lengths across the sequences in
the batch (vLLM-style). There is no batch/padding dimension.

  prefill step : query_lens = [len(prompt_i)]   -> num_tokens = sum(prompt lens)
  decode step  : query_lens = [1, 1, ..., 1]     -> num_tokens = batch size

`context_lens[i]` is the total number of tokens in sequence i's KV cache AFTER this
step, i.e. (already cached tokens) + query_lens[i]. For a fresh prefill it equals
query_lens[i]; for decode it is the full history length including the new token.
Every backend must attend causally over exactly context_lens[i] keys for sequence i.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class AttnMetadata:
    is_prefill: bool
    seq_ids: list[int]  # engine-level sequence ids, one per sequence in the batch
    query_lens: list[int]  # tokens each sequence contributes to this step
    context_lens: list[int]  # KV length per sequence after this step
    positions: torch.Tensor  # [num_tokens] int64 absolute positions (for RoPE)
    # ---- paged backends only -----------------------------------------------
    # Physical slot for each token in this step: block_id * block_size + offset.
    slot_mapping: torch.Tensor | None = None  # [num_tokens] int64
    # Block table per sequence, padded with -1 to max_blocks: [batch, max_blocks] int32.
    block_tables: torch.Tensor | None = None
    block_size: int | None = None
    # ---- prefix caching (later) ---------------------------------------------
    # Number of already-cached tokens per sequence at the start of a prefill step
    # (0 without prefix caching). Prefill queries attend to these AND the new tokens.
    num_cached_tokens: list[int] = field(default_factory=list)
    # ---- convenience --------------------------------------------------------
    cu_seqlens_q: torch.Tensor | None = None  # [batch+1] int32 cumulative query lens
    # ---- GPU / CUDA-graph backends (optional; never required by CPU backends) ------
    # Device copy of `context_lens`: [batch] int32. `paged_flash` reads flash-attn's
    # `cache_seqlens` from this tensor when it is present (and caches one here when it
    # is not), so a captured decode step never does a host->device copy per layer.
    context_lens_t: torch.Tensor | None = None
    # `block_tables` with the -1 padding replaced by a valid block id (0):
    # [batch, max_blocks] int32. flash-attn may dereference padding entries of the last
    # KV tile (their values are masked out), so they must point at real memory.
    block_tables_nonneg: torch.Tensor | None = None

    @property
    def num_seqs(self) -> int:
        return len(self.seq_ids)

    @property
    def num_tokens(self) -> int:
        return int(sum(self.query_lens))

    def __post_init__(self) -> None:
        assert len(self.query_lens) == len(self.context_lens) == len(self.seq_ids)
        for q, c in zip(self.query_lens, self.context_lens):
            assert 1 <= q <= c, (q, c)
        if self.cu_seqlens_q is None:
            cu = [0]
            for q in self.query_lens:
                cu.append(cu[-1] + q)
            self.cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=self.positions.device)


class AttentionBackend(ABC):
    """Owns the KV storage for all layers and computes attention for one layer.

    Contract for `forward`:
      q: [num_tokens, num_heads, head_dim]        (already RoPE'd)
      k: [num_tokens, num_kv_heads, head_dim]     (already RoPE'd)
      v: [num_tokens, num_kv_heads, head_dim]
      returns: [num_tokens, num_heads, head_dim]
    The backend must (1) write k/v for this step into its cache at the positions the
    metadata describes, then (2) attend each query to all cached keys of its own
    sequence up to and including itself (causal), with GQA handled inside the backend
    (repeat kv heads num_heads // num_kv_heads times). Scale is 1/sqrt(head_dim).
    """

    @abstractmethod
    def forward(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                meta: AttnMetadata) -> torch.Tensor: ...

    @abstractmethod
    def free_sequence(self, seq_id: int) -> None:
        """Release anything held for a finished/aborted sequence."""

    def reset(self) -> None:  # optional
        pass


def causal_softmax_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             num_query_tokens: int) -> torch.Tensor:
    """Reference single-sequence attention used by tests and the naive backend.

    q: [Tq, H, D]   k, v: [Tk, Hkv, D] with Tk >= Tq; the queries are the LAST Tq
    positions of the Tk keys (causal). GQA expansion done here. Returns [Tq, H, D].
    Computed in float32 for numerical stability regardless of input dtype.
    """
    Tq, H, D = q.shape
    Tk, Hkv, _ = k.shape
    assert Tq == num_query_tokens and Tk >= Tq
    groups = H // Hkv
    qf = q.float().transpose(0, 1)  # [H, Tq, D]
    kf = k.float().repeat_interleave(groups, dim=1).transpose(0, 1)  # [H, Tk, D]
    vf = v.float().repeat_interleave(groups, dim=1).transpose(0, 1)  # [H, Tk, D]
    scores = torch.matmul(qf, kf.transpose(-1, -2)) / (D ** 0.5)  # [H, Tq, Tk]
    # query i (0-based within the last Tq) sits at absolute key index (Tk - Tq + i)
    qpos = torch.arange(Tk - Tq, Tk, device=q.device)[:, None]
    kpos = torch.arange(Tk, device=q.device)[None, :]
    mask = kpos > qpos  # True where key is in the future
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)  # [H, Tq, D]
    return out.transpose(0, 1).to(q.dtype)
