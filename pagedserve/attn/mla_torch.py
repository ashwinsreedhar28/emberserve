"""Multi-head latent attention over a paged latent cache, in plain PyTorch (the reference).

DeepSeek-V2/V3 attention, per head h, for query token i over context tokens t:

    score[i, t] = (q_nope[i,h] . k_nope[t,h] + q_pe[i,h] . k_pe[t]) * scale
    k_nope[t,h] = W_UK[h] c[t]          v[t,h] = W_UV[h] c[t]
    out[i,h]    = sum_t softmax(score)[i,t] * v[t,h]

`c[t]` is the 512-dim compressed latent and `k_pe[t]` the 64-dim rope key shared by all
heads: that pair is the whole cache row. Nothing per head is ever stored, which is why MLA
caches ~7x less than the equivalent GQA layout.

This backend runs the *absorbed* form: fold W_UK into the query (`q_c = W_UK[h]^T q_nope`,
a 512-dim query per head) so scores are dot products against the raw cache rows, and fold
W_UV out after the softmax (`out = W_UV[h] (P c)`). Attention is then MQA over the latent:
16 query heads, one 576-dim "key head", one 512-dim "value head". The same math serves
prefill (queries = the last query_len positions of the context) and decode. Everything is
computed in float32 per sequence; the Triton and flash paths are the GPU optimization.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pagedserve.attn.base import AttnMetadata
from pagedserve.config import ModelConfig
from pagedserve.kv.cache import PagedLatentCache


def mla_attention_absorbed(q_nope: Tensor, q_pe: Tensor, latent: Tensor, w_uk: Tensor,
                           w_uv: Tensor, scale: float, kv_lora_rank: int) -> Tensor:
    """One sequence. `q_nope [Q, H, Dn]`, `q_pe [Q, H, Dr]` are the LAST Q positions of a
    context whose cache rows are `latent [T, Dl + Dr]`; `w_uk [H, Dn, Dl]`, `w_uv [H, Dv, Dl]`.
    Returns `[Q, H, Dv]` in q_nope's dtype; math in float32."""
    q_len, t_len = q_nope.shape[0], latent.shape[0]
    assert q_len <= t_len, (q_len, t_len)
    c = latent[:, :kv_lora_rank].float()  # [T, Dl]
    k_pe = latent[:, kv_lora_rank:].float()  # [T, Dr]
    q_c = torch.einsum("qhn,hnl->qhl", q_nope.float(), w_uk.float())  # [Q, H, Dl]
    scores = (torch.einsum("qhl,tl->hqt", q_c, c)
              + torch.einsum("qhr,tr->hqt", q_pe.float(), k_pe)) * scale  # [H, Q, T]
    # causal: query j sits at absolute position T - Q + j and may see t <= that position
    pos = torch.arange(t_len - q_len, t_len, device=scores.device)[:, None]
    mask = torch.arange(t_len, device=scores.device)[None, :] > pos  # [Q, T]
    scores = scores.masked_fill(mask[None], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out_c = torch.einsum("hqt,tl->qhl", p, c)  # [Q, H, Dl]
    return torch.einsum("qhl,hvl->qhv", out_c, w_uv.float()).to(q_nope.dtype)


class MLATorchBackend:
    """Owns a `PagedLatentCache`; `forward` writes this step's latent rows then attends."""

    name = "mla_torch"

    def __init__(self, config: ModelConfig, cache: PagedLatentCache) -> None:
        assert config.mla is not None
        self.config = config
        self.cache = cache
        self.kv_lora_rank = config.mla.kv_lora_rank

    def forward(self, layer_idx: int, q_nope: Tensor, q_pe: Tensor, latent: Tensor,
                w_uk: Tensor, w_uv: Tensor, scale: float, meta: AttnMetadata,
                kv_b_weight: Tensor | None = None) -> Tensor:
        assert meta.slot_mapping is not None and meta.block_tables is not None
        self.cache.write(layer_idx, latent, meta.slot_mapping)
        outs = []
        start = 0
        for i, q_len in enumerate(meta.query_lens):
            ctx = self.cache.gather(layer_idx, meta.block_tables[i], meta.context_lens[i])
            outs.append(mla_attention_absorbed(q_nope[start:start + q_len], q_pe[start:start + q_len],
                                               ctx, w_uk, w_uv, scale, self.kv_lora_rank))
            start += q_len
        return torch.cat(outs, dim=0)

    def free_sequence(self, seq_id: int) -> None:  # blocks are the BlockManager's business
        pass

    def reset(self) -> None:
        pass
