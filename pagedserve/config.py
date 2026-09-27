"""Static configuration: model architecture and engine/runtime knobs.

ModelConfig mirrors the fields of an HF `config.json` for the Qwen2 family.
The defaults are Qwen/Qwen2.5-0.5B-Instruct. Tests use `ModelConfig.tiny()`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    # Qwen2 attention projections carry a bias; o_proj and the MLP do not.
    attention_bias: bool = True
    eos_token_id: int = 151645  # <|im_end|> for the Instruct model
    bos_token_id: int | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        """Query heads per KV head (GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    def kv_bytes_per_token(self, dtype: torch.dtype) -> int:
        """Bytes of K+V stored per token across all layers."""
        esize = torch.tensor([], dtype=dtype).element_size()
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * esize

    @classmethod
    def from_hf_dir(cls, model_dir: str | os.PathLike) -> "ModelConfig":
        """Read an HF snapshot directory's config.json."""
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        assert cfg.get("model_type") == "qwen2", f"unsupported model_type {cfg.get('model_type')}"
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            max_position_embeddings=cfg.get("max_position_embeddings", 32768),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=cfg.get("rope_theta", 1_000_000.0),
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            attention_bias=True,
            eos_token_id=cfg["eos_token_id"] if isinstance(cfg.get("eos_token_id"), int) else 151645,
            bos_token_id=cfg.get("bos_token_id"),
        )

    @classmethod
    def tiny(cls, **overrides) -> "ModelConfig":
        """A 2-layer toy config for unit tests. Same code path as the real model."""
        base = dict(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
            tie_word_embeddings=True,
            attention_bias=True,
            eos_token_id=1,
            bos_token_id=None,
        )
        base.update(overrides)
        return cls(**base)


@dataclass
class EngineConfig:
    model_dir: str | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    # Paged KV cache geometry.
    block_size: int = 16
    num_gpu_blocks: int | None = None  # None -> derive from gpu_memory_utilization
    gpu_memory_utilization: float = 0.90
    # Scheduler limits.
    max_num_seqs: int = 256
    # Token budget per prefill step. With `enable_chunked_prefill` it is the cap on EVERY
    # step (decode tokens + prefill chunk tokens); typical values are 512-2048.
    max_num_batched_tokens: int = 8192
    max_model_len: int = 4096
    # Attention backend: "naive" (per-seq growing cache), "paged_torch" (gather), "paged_flash" (GPU).
    attn_backend: str = "paged_torch"
    # Feature flags (later phases).
    enable_prefix_caching: bool = False
    enable_cuda_graphs: bool = False
    # Chunked prefill (Sarathi-Serve / vLLM): every step carries one token per decoding
    # request plus as many prompt tokens as fit in the remaining `max_num_batched_tokens`,
    # so a long prompt is split across steps instead of stalling every decode for one
    # long prefill step. Prompts longer than the budget are accepted when this is on.
    enable_chunked_prefill: bool = False
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @staticmethod
    def dtype_from_str(s: str) -> torch.dtype:
        return {"float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[s]
