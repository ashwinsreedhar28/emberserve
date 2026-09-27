"""Static configuration: model architecture and engine/runtime knobs.

ModelConfig mirrors the fields of an HF `config.json` for the Qwen2 and Llama families
(`model_type` qwen2 / llama / mistral: the same decoder block, differing in attention
bias, RoPE scaling and the set of end-of-sequence ids). The defaults are
Qwen/Qwen2.5-0.5B-Instruct. Tests use `ModelConfig.tiny()`.
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
    # Qwen2 attention projections carry a bias; o_proj and the MLP do not. Llama/Mistral: none.
    attention_bias: bool = True
    eos_token_id: int = 151645  # <|im_end|> for the Instruct model
    bos_token_id: int | None = None
    model_type: str = "qwen2"
    # Every id that ends generation (Llama 3 lists several: <|end_of_text|>, <|eot_id|>...).
    eos_token_ids: tuple[int, ...] = ()
    # HF `rope_scaling` (e.g. Llama 3's {"rope_type": "llama3", "factor": 32, ...}); None = plain.
    rope_scaling: dict | None = None

    SUPPORTED_MODEL_TYPES = ("qwen2", "llama", "mistral")

    def __post_init__(self) -> None:
        if not self.eos_token_ids:  # normalize so configs compare equal however they were built
            object.__setattr__(self, "eos_token_ids", (self.eos_token_id,))

    @property
    def all_eos_token_ids(self) -> frozenset[int]:
        return frozenset(self.eos_token_ids) | {self.eos_token_id}

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
        model_type = cfg.get("model_type")
        if model_type not in cls.SUPPORTED_MODEL_TYPES:
            raise ValueError(f"unsupported model_type {model_type!r}; supported: "
                             f"{cls.SUPPORTED_MODEL_TYPES}")
        if cfg.get("sliding_window") not in (None, 0) and cfg.get("use_sliding_window", True):
            raise ValueError("sliding-window attention is not supported (set for this checkpoint)")
        if cfg.get("mlp_bias", False):
            raise ValueError("mlp_bias=True is not supported")
        if cfg.get("head_dim") not in (None, cfg["hidden_size"] // cfg["num_attention_heads"]):
            raise ValueError("head_dim != hidden_size / num_attention_heads is not supported")
        eos = cfg.get("eos_token_id")
        if isinstance(eos, list):
            eos_ids = tuple(int(e) for e in eos)
        elif isinstance(eos, int):
            eos_ids = (eos,)
        else:
            eos_ids = (151645,) if model_type == "qwen2" else ()
        if not eos_ids:
            raise ValueError("config.json has no eos_token_id")
        rope_scaling = cfg.get("rope_scaling")
        if rope_scaling is not None:
            kind = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if kind not in ("llama3", "default"):
                raise ValueError(f"unsupported rope_scaling type {kind!r}")
            if kind == "default":
                rope_scaling = None
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            max_position_embeddings=cfg.get("max_position_embeddings", 32768),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=cfg.get("rope_theta", 1_000_000.0 if model_type == "qwen2" else 10_000.0),
            tie_word_embeddings=cfg.get("tie_word_embeddings", model_type == "qwen2"),
            attention_bias=cfg.get("attention_bias", model_type == "qwen2"),
            eos_token_id=eos_ids[0],
            bos_token_id=cfg.get("bos_token_id"),
            model_type=model_type,
            eos_token_ids=eos_ids,
            rope_scaling=rope_scaling,
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
