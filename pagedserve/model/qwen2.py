"""Qwen2 / Llama / Mistral decoder written from scratch against the `AttentionBackend` interface.

The three families share this block exactly (pre-norm RMSNorm, rotate-half RoPE, GQA,
SwiGLU); `ModelConfig` carries the differences (attention bias, RoPE scaling, eos ids).

Module attribute names mirror HF's `Qwen2ForCausalLM` so safetensors weight names map
one-to-one (see `model/weights.py`). Every activation uses the packed layout
`[num_tokens, hidden]`; per-sequence boundaries and RoPE positions come from
`AttnMetadata`. The attention scale (1/sqrt(head_dim)) is applied inside the backend.
"""

from __future__ import annotations

import torch
from torch import nn

from pagedserve.attn.base import AttentionBackend, AttnMetadata
from pagedserve.config import ModelConfig
from pagedserve.model import ops
from pagedserve.model.rope import RotaryEmbedding


class RMSNorm(nn.Module):
    """HF `Qwen2RMSNorm`: normalize in float32, cast back, then scale by `weight`.
    `forward_with_residual` is the fused pre-norm step: add the block's output to the
    residual stream, return the normed value and the new stream (one kernel on CUDA)."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.rmsnorm(x, self.weight, self.variance_epsilon)

    def forward_with_residual(self, x: torch.Tensor,
                              residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return ops.fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)


class Qwen2MLP(nn.Module):
    """SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`, no biases. `gate_proj` and
    `up_proj` are one fused `gate_up_proj` matmul (rows [0, I) are gate, [I, 2I) are up);
    the checkpoint's two tensors are copied into it by `model/weights.py`."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden, inter = config.hidden_size, config.intermediate_size
        self.intermediate_size = inter
        self.gate_up_proj = nn.Linear(hidden, 2 * inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(ops.silu_and_mul(self.gate_up_proj(x)))


class Qwen2Attention(nn.Module):
    """QKV projections + RoPE; the attention itself is delegated to the backend."""

    def __init__(self, config: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # q_proj / k_proj / v_proj fused into one matmul: output columns are [q | k | v].
        self.qkv_proj = nn.Linear(hidden, self.q_size + 2 * self.kv_size,
                                  bias=config.attention_bias)
        self.o_proj = nn.Linear(self.q_size, hidden, bias=False)
        # Shared with every other layer; it holds no parameters, only cached cos/sin tables.
        self.rotary_emb = rotary_emb

    def pre_attention(self, hidden: torch.Tensor,
                      positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Everything before the attention kernel: `(q, k, v)`, rope applied. Row-wise, so
        it can run on padded rows (piecewise CUDA graphs)."""
        n = hidden.shape[0]
        qkv = self.qkv_proj(hidden)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(n, self.num_heads, self.head_dim)
        k = k.view(n, self.num_kv_heads, self.head_dim)
        v = v.view(n, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(q, k, positions)
        return q, k, v

    def attend(self, pre: tuple[torch.Tensor, ...], backend: AttentionBackend,
               meta: AttnMetadata) -> torch.Tensor:
        """The attention kernel over the real rows: `[n, H, D]`."""
        q, k, v = pre
        return backend.forward(self.layer_idx, q, k, v, meta)

    def attn_out_shape(self, n: int) -> tuple[int, int, int]:
        return (n, self.num_heads, self.head_dim)

    def post_attention(self, out: torch.Tensor) -> torch.Tensor:
        return self.o_proj(out.reshape(out.shape[0], self.q_size))

    def forward(self, hidden: torch.Tensor, backend: AttentionBackend,
                meta: AttnMetadata) -> torch.Tensor:
        pre = self.pre_attention(hidden, meta.positions)
        return self.post_attention(self.attend(pre, backend, meta))


class Qwen2DecoderLayer(nn.Module):
    """Pre-norm transformer block: `x + attn(norm(x))`, then `x + mlp(norm(x))`.

    Written in the (hidden, residual) form so each norm can fuse the residual add that
    precedes it: the layer receives the previous block's output `hidden` and the residual
    stream, and returns its own output and the updated stream (the final add happens in
    the next layer's input norm, or in the model's final norm)."""

    def __init__(self, config: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx, rotary_emb)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    # The layer is written as three pieces so `attn/piecewise_graphs.py` can capture `pre`
    # and `post` (row-wise, shape-stable) into CUDA graphs and run `attend` eagerly between
    # them; `forward` is the three in a row.
    def pre(self, hidden: torch.Tensor, residual: torch.Tensor | None,
            positions: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Input norm (+ residual add) and the attention projections: `(pre, residual)`."""
        if residual is None:  # first layer: the embedding is the residual stream
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_with_residual(hidden, residual)
        return self.self_attn.pre_attention(hidden, positions), residual

    def attend(self, pre: tuple[torch.Tensor, ...], backend: AttentionBackend,
               meta: AttnMetadata) -> torch.Tensor:
        return self.self_attn.attend(pre, backend, meta)

    def post(self, attn_out: torch.Tensor,
             residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Output projection, post-attention norm (+ residual add), MLP: `(hidden, residual)`."""
        hidden = self.self_attn.post_attention(attn_out)
        hidden, residual = self.post_attention_layernorm.forward_with_residual(hidden, residual)
        return self.mlp(hidden), residual

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None,
                backend: AttentionBackend, meta: AttnMetadata) -> tuple[torch.Tensor, torch.Tensor]:
        pre, residual = self.pre(hidden, residual, meta.positions)
        return self.post(self.attend(pre, backend, meta), residual)


class Qwen2Model(nn.Module):
    """Embedding -> decoder layers -> final norm. Returns hidden states, not logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
            rope_scaling=config.rope_scaling)
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(config, i, self.rotary_emb) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, backend: AttentionBackend,
                meta: AttnMetadata) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, backend, meta)
        normed, _ = self.norm.forward_with_residual(hidden, residual)
        return normed


class Qwen2ForCausalLM(nn.Module):
    """Qwen2 with an LM head. Parameters are created in the default dtype on CPU;
    the caller moves them with `.to(device, dtype)` before or after loading weights."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, backend: AttentionBackend,
                meta: AttnMetadata) -> torch.Tensor:
        """`input_ids: [N] int64` -> final-normed hidden states `[N, hidden]`."""
        return self.model(input_ids, backend, meta)

    def compute_logits(self, hidden: torch.Tensor,
                       meta: AttnMetadata | None = None) -> torch.Tensor:
        """Project to the vocabulary. With `meta`, only the last token of each sequence
        is projected -> `[num_seqs, vocab]`; without it every token -> `[N, vocab]`."""
        if meta is not None:
            last = (meta.cu_seqlens_q[1:] - 1).to(torch.long)
            hidden = hidden.index_select(0, last)
        return self.lm_head(hidden)

    def forward_logits_all(self, input_ids: torch.Tensor, backend: AttentionBackend,
                           meta: AttnMetadata) -> torch.Tensor:
        """Logits for every token in the step, `[N, vocab]` (golden-logit comparisons)."""
        return self.compute_logits(self.forward(input_ids, backend, meta))


# Same module for every supported family; the name records where it started.
LlamaForCausalLM = Qwen2ForCausalLM


def reset_parameters_deterministic(model: nn.Module, seed: int) -> None:
    """Re-initialize every parameter from a fixed generator so tests get reproducible
    random models: linear/embedding weights ~ N(0, 0.02), biases ~ N(0, 0.01), norms = 1.
    Tied parameters are initialized once."""
    gen = torch.Generator().manual_seed(seed)
    seen: set[int] = set()
    with torch.no_grad():
        for module in model.modules():
            for pname, param in module.named_parameters(recurse=False):
                if id(param) in seen:
                    continue
                seen.add(id(param))
                if isinstance(module, RMSNorm):
                    param.fill_(1.0)
                    continue
                std = 0.01 if pname == "bias" else 0.02
                param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32) * std)
