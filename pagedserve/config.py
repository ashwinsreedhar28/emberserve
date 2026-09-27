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
class MoEConfig:
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int = 0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"

    def __post_init__(self) -> None:
        if self.scoring_func != "sigmoid":
            raise ValueError(f"unsupported scoring_func {self.scoring_func!r} (sigmoid only)")
        if self.topk_method != "noaux_tc":
            raise ValueError(f"unsupported topk_method {self.topk_method!r} (noaux_tc only)")
        if self.n_routed_experts % self.n_group:
            raise ValueError("n_routed_experts must be divisible by n_group")
        if not 1 <= self.topk_group <= self.n_group:
            raise ValueError("topk_group must be in [1, n_group]")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be in [1, n_routed_experts]")

    @classmethod
    def from_hf(cls, cfg: dict) -> "MoEConfig":
        """From an HF `config.json` dict (DeepSeek-V2/V3 field names)."""
        return cls(
            hidden_size=cfg["hidden_size"],
            moe_intermediate_size=cfg["moe_intermediate_size"],
            n_routed_experts=cfg["n_routed_experts"],
            num_experts_per_tok=cfg["num_experts_per_tok"],
            n_shared_experts=cfg.get("n_shared_experts", 0) or 0,
            n_group=cfg.get("n_group", 1) or 1,
            topk_group=cfg.get("topk_group", 1) or 1,
            norm_topk_prob=cfg.get("norm_topk_prob", True),
            routed_scaling_factor=float(cfg.get("routed_scaling_factor", 1.0)),
            scoring_func=cfg.get("scoring_func", "sigmoid"),
            topk_method=cfg.get("topk_method", "noaux_tc"),
        )


@dataclass(frozen=True)
class MLAConfig:
    """Multi-head latent attention geometry (DeepSeek-V2/V3, Moonshot Moonlight)."""

    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_dim(self) -> int:
        """What the cache stores per token: the compressed KV latent plus the shared rope key."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @classmethod
    def from_hf(cls, cfg: dict) -> "MLAConfig":
        return cls(q_lora_rank=cfg.get("q_lora_rank"), kv_lora_rank=cfg["kv_lora_rank"],
                   qk_nope_head_dim=cfg["qk_nope_head_dim"], qk_rope_head_dim=cfg["qk_rope_head_dim"],
                   v_head_dim=cfg["v_head_dim"])


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
    # DeepSeek-V2/V3 family: latent attention and mixture-of-experts layers.
    mla: MLAConfig | None = None
    moe: MoEConfig | None = None
    first_k_dense_replace: int = 0  # the first k layers use the dense MLP
    moe_layer_freq: int = 1  # every k-th layer (past the dense ones) is MoE

    SUPPORTED_MODEL_TYPES = ("qwen2", "llama", "mistral", "deepseek_v2", "deepseek_v3")

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (self.moe is not None and layer_idx >= self.first_k_dense_replace
                and layer_idx % self.moe_layer_freq == 0)

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
        """Bytes of cache stored per token across all layers: K+V per KV head, or, with
        MLA, one latent row (`kv_lora_rank + qk_rope_head_dim`) per layer."""
        esize = torch.tensor([], dtype=dtype).element_size()
        if self.mla is not None:
            return self.num_hidden_layers * self.mla.latent_dim * esize
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
        is_deepseek = model_type in ("deepseek_v2", "deepseek_v3")
        if not is_deepseek and cfg.get("head_dim") not in (
                None, cfg["hidden_size"] // cfg["num_attention_heads"]):
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
        mla = moe = None
        first_k_dense = 0
        moe_freq = 1
        if is_deepseek:
            mla = MLAConfig.from_hf(cfg)
            if cfg.get("n_routed_experts"):
                moe = MoEConfig.from_hf(cfg)
                first_k_dense = int(cfg.get("first_k_dense_replace", 0))
                moe_freq = int(cfg.get("moe_layer_freq", 1) or 1)
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
            mla=mla,
            moe=moe,
            first_k_dense_replace=first_k_dense,
            moe_layer_freq=moe_freq,
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
    # Async scheduling (vLLM v1): the engine launches step N+1 (schedule, build inputs,
    # forward, sample) before it has read step N's sampled tokens back from the device.
    # Decode rows whose token is still on the device take it from the previous step's
    # sampled tensor with a device-side gather, so the CPU work of one step overlaps the
    # GPU work of the previous one and the device never waits for Python between steps.
    # A request that finishes on EOS computes one extra (discarded) token; length limits
    # are anticipated so they waste nothing. Outputs of a step come back from the NEXT
    # `step()` call.
    async_scheduling: bool = False
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @staticmethod
    def dtype_from_str(s: str) -> torch.dtype:
        return {"float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[s]
