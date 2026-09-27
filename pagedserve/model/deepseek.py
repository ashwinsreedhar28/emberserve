"""DeepSeek-V2/V3 decoder (Moonshot Moonlight, DeepSeek-V2-Lite): multi-head latent
attention + (dense or mixture-of-experts) SwiGLU, written against `MLATorchBackend` and
friends. Module names mirror HF `DeepseekV3ForCausalLM` so checkpoints load by name.

Attention (HF `DeepseekV3Attention`):
    q      = q_proj(x)                              or  q_b_proj(q_a_layernorm(q_a_proj(x)))
    q_nope, q_pe = split(q.view(H, qk_nope + qk_rope))
    ckv    = kv_a_proj_with_mqa(x)  ->  c (kv_lora_rank), k_pe (qk_rope, one per token, all heads)
    c      = kv_a_layernorm(c)
    k_pe, q_pe rotated with RoPE in DeepSeek's interleaved-pair convention
    cache row = [c | k_pe]; k_nope and v are W_UK c / W_UV c from kv_b_proj, never stored
    out    = attention(q, k, v) with scale 1/sqrt(qk_nope + qk_rope);  o_proj
"""

from __future__ import annotations

import torch
from torch import nn

from pagedserve.attn.base import AttnMetadata
from pagedserve.config import ModelConfig
from pagedserve.model.moe import DeepseekMoE
from pagedserve.model.qwen2 import Qwen2MLP, RMSNorm
from pagedserve.model.rope import RotaryEmbedding


def interleave_to_halves(x: torch.Tensor) -> torch.Tensor:
    """DeepSeek pairs dimension 2i with 2i+1 for RoPE; the rotate-half kernel pairs i with
    i + D/2. Permute so the same pairs meet: [..., D] -> [..., D/2, 2] -> [..., 2, D/2]."""
    *lead, d = x.shape
    return x.view(*lead, d // 2, 2).transpose(-1, -2).reshape(*lead, d)


class MLAAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        assert config.mla is not None
        m = config.mla
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.qk_nope = m.qk_nope_head_dim
        self.qk_rope = m.qk_rope_head_dim
        self.v_head_dim = m.v_head_dim
        self.kv_lora_rank = m.kv_lora_rank
        self.q_head_dim = m.qk_head_dim
        hidden = config.hidden_size
        if m.q_lora_rank is None:
            self.q_proj = nn.Linear(hidden, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(hidden, m.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(m.q_lora_rank, config.rms_norm_eps)
            self.q_b_proj = nn.Linear(m.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden, m.kv_lora_rank + m.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(m.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(m.kv_lora_rank, self.num_heads * (m.qk_nope_head_dim + m.v_head_dim),
                                   bias=False)
        self.o_proj = nn.Linear(self.num_heads * m.v_head_dim, hidden, bias=False)
        self.softmax_scale = self.q_head_dim ** -0.5
        self.rotary_emb = rotary_emb

    def up_projections(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`(w_uk [H, Dn, Dl], w_uv [H, Dv, Dl])`: kv_b_proj split per head into the
        k_nope and v up-projections the absorbed attention folds into q and out."""
        w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope + self.v_head_dim, self.kv_lora_rank)
        return w[:, :self.qk_nope, :], w[:, self.qk_nope:, :]

    def forward(self, hidden: torch.Tensor, backend, meta: AttnMetadata) -> torch.Tensor:
        n = hidden.shape[0]
        if hasattr(self, "q_proj"):
            q = self.q_proj(hidden)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden)))
        q = q.view(n, self.num_heads, self.q_head_dim)
        q_nope, q_pe = q.split([self.qk_nope, self.qk_rope], dim=-1)
        ckv = self.kv_a_proj_with_mqa(hidden)
        c, k_pe = ckv.split([self.kv_lora_rank, self.qk_rope], dim=-1)
        c = self.kv_a_layernorm(c)
        q_pe, k_pe = self.rotary_emb(interleave_to_halves(q_pe).contiguous(),
                                     interleave_to_halves(k_pe).unsqueeze(1).contiguous(),
                                     meta.positions)
        latent = torch.cat([c, k_pe.squeeze(1)], dim=-1)  # [n, Dl + Dr]
        w_uk, w_uv = self.up_projections()
        out = backend.forward(self.layer_idx, q_nope.contiguous(), q_pe, latent, w_uk, w_uv,
                              self.softmax_scale, meta, kv_b_weight=self.kv_b_proj.weight)  # [n, H, Dv]
        return self.o_proj(out.reshape(n, self.num_heads * self.v_head_dim))


class DeepseekDecoderLayer(nn.Module):
    """Pre-norm block in the (hidden, residual) form of `Qwen2DecoderLayer`; the MLP is
    dense for the first `first_k_dense_replace` layers and MoE after that."""

    def __init__(self, config: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = MLAAttention(config, layer_idx, rotary_emb)
        if config.is_moe_layer(layer_idx):
            assert config.moe is not None
            self.mlp: nn.Module = DeepseekMoE(config.moe)
        else:
            self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None, backend,
                meta: AttnMetadata) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_with_residual(hidden, residual)
        hidden = self.self_attn(hidden, backend, meta)
        hidden, residual = self.post_attention_layernorm.forward_with_residual(hidden, residual)
        return self.mlp(hidden), residual


class DeepseekModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.mla is not None
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.mla.qk_rope_head_dim, config.max_position_embeddings,
                                          config.rope_theta, rope_scaling=config.rope_scaling)
        self.layers = nn.ModuleList(
            DeepseekDecoderLayer(config, i, self.rotary_emb) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, backend, meta: AttnMetadata) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, backend, meta)
        normed, _ = self.norm.forward_with_residual(hidden, residual)
        return normed


class DeepseekForCausalLM(nn.Module):
    """Same interface as `Qwen2ForCausalLM`."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = DeepseekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, backend, meta: AttnMetadata) -> torch.Tensor:
        return self.model(input_ids, backend, meta)

    def compute_logits(self, hidden: torch.Tensor, meta: AttnMetadata | None = None) -> torch.Tensor:
        if meta is not None:
            last = (meta.cu_seqlens_q[1:] - 1).to(torch.long)
            hidden = hidden.index_select(0, last)
        return self.lm_head(hidden)

    def forward_logits_all(self, input_ids: torch.Tensor, backend, meta: AttnMetadata) -> torch.Tensor:
        return self.compute_logits(self.forward(input_ids, backend, meta))


__all__ = ["DeepseekForCausalLM", "DeepseekDecoderLayer", "MLAAttention", "interleave_to_halves"]
