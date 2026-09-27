"""Load HF Qwen2 safetensors checkpoints into `Qwen2ForCausalLM`.

Our module tree mirrors HF's, so the name mapping is the identity apart from a few
non-parameter tensors some snapshots carry (e.g. `model.rotary_emb.inv_freq`), which are
skipped. Loading is strict: unexpected checkpoint keys and never-loaded parameters both
raise, with the tied `lm_head.weight` as the only tolerated absence.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors import safe_open

from pagedserve.config import ModelConfig
from pagedserve.model.qwen2 import Qwen2ForCausalLM

_SKIP_SUFFIXES = ("rotary_emb.inv_freq",)
# HF tensor name suffix -> (fused local suffix, shard id). The fused parameter's rows are
# [q | k | v] and [gate | up]; `_shard_rows` computes the slice from the model config.
_FUSED = {
    "self_attn.q_proj": ("self_attn.qkv_proj", "q"),
    "self_attn.k_proj": ("self_attn.qkv_proj", "k"),
    "self_attn.v_proj": ("self_attn.qkv_proj", "v"),
    "mlp.gate_proj": ("mlp.gate_up_proj", "gate"),
    "mlp.up_proj": ("mlp.gate_up_proj", "up"),
}
_SHARDS = {"qkv_proj": ("q", "k", "v"), "gate_up_proj": ("gate", "up")}


def hf_to_local(name: str) -> tuple[str, str | None] | None:
    """Map an HF checkpoint key to `(state_dict key, shard)`; shard is None for a whole
    tensor and one of q/k/v/gate/up for a slice of a fused projection. None = skip."""
    if name.endswith(_SKIP_SUFFIXES):
        return None
    head, _, leaf = name.rpartition(".")  # leaf: weight | bias
    for hf_suffix, (local_suffix, shard) in _FUSED.items():
        if head.endswith(hf_suffix):
            return f"{head[:-len(hf_suffix)]}{local_suffix}.{leaf}", shard
    return name, None


def hf_to_local_name(name: str) -> str | None:
    """`state_dict` key an HF checkpoint key lands in (fused projections included)."""
    m = hf_to_local(name)
    return None if m is None else m[0]


def _shard_rows(config: ModelConfig, shard: str) -> slice:
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size
    return {"q": slice(0, q), "k": slice(q, q + kv), "v": slice(q + kv, q + 2 * kv),
            "gate": slice(0, inter), "up": slice(inter, 2 * inter)}[shard]


def _expected_shards(local: str) -> tuple[str | None, ...]:
    for fused, shards in _SHARDS.items():
        if f".{fused}." in local:
            return shards
    return (None,)


def _hf_name(local: str, shard: str | None) -> str:
    """Inverse of `hf_to_local` for error messages."""
    if shard is None:
        return local
    head, _, leaf = local.rpartition(".")
    for hf_suffix, (local_suffix, sh) in _FUSED.items():
        if sh == shard and head.endswith(local_suffix):
            return f"{head[:-len(local_suffix)]}{hf_suffix}.{leaf}"
    return local  # pragma: no cover


def hf_state_dict(model: Qwen2ForCausalLM) -> dict[str, torch.Tensor]:
    """The model's parameters under HF names, fused projections split back apart (for
    writing checkpoints the HF loader, or this loader, can read)."""
    out: dict[str, torch.Tensor] = {}
    for local, tensor in model.state_dict().items():
        head, _, leaf = local.rpartition(".")
        pieces = [(hf_suffix, shard) for hf_suffix, (local_suffix, shard) in _FUSED.items()
                  if head.endswith(local_suffix)]
        if not pieces:
            out[local] = tensor
            continue
        local_suffix = _FUSED[pieces[0][0]][0]
        prefix = head[:-len(local_suffix)]
        for hf_suffix, shard in pieces:
            out[f"{prefix}{hf_suffix}.{leaf}"] = tensor[_shard_rows(model.config, shard)]
    return out


def load_hf_weights(model: Qwen2ForCausalLM, model_dir: str | os.PathLike,
                    dtype: torch.dtype | None = None,
                    device: torch.device | str | None = None) -> None:
    """Copy every tensor from `model_dir/*.safetensors` into `model` in place.

    Tensors are cast to `dtype` (default: the target parameter's dtype) and moved to
    `device` (default: the target parameter's device) during the copy. q/k/v and gate/up
    tensors are copied into their row range of the fused parameter.
    """
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {model_dir}")

    state = model.state_dict()
    loaded: set[tuple[str, str | None]] = set()
    with torch.no_grad():
        for path in files:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for hf_name in f.keys():
                    m = hf_to_local(hf_name)
                    if m is None:
                        continue
                    local, shard = m
                    if local not in state:
                        raise KeyError(f"unexpected checkpoint key {hf_name!r} in {path.name}")
                    if (local, shard) in loaded:
                        raise KeyError(f"duplicate checkpoint key {hf_name!r} in {path.name}")
                    target = state[local]
                    if shard is not None:
                        target = target[_shard_rows(model.config, shard)]
                    src = f.get_tensor(hf_name)
                    if src.shape != target.shape:
                        raise ValueError(
                            f"shape mismatch for {hf_name!r}: checkpoint {tuple(src.shape)} "
                            f"vs model {tuple(target.shape)}")
                    target.copy_(src.to(device=device or target.device,
                                        dtype=dtype or target.dtype))
                    loaded.add((local, shard))

    tied = model.config.tie_word_embeddings
    missing = [_hf_name(k, sh) for k in state for sh in _expected_shards(k)
               if not (tied and k == "lm_head.weight") and (k, sh) not in loaded]
    if missing:
        raise KeyError(f"parameters never loaded from {model_dir}: {missing}")


def load_model(model_dir: str | os.PathLike, device: torch.device | str = "cpu",
               dtype: torch.dtype = torch.float32) -> Qwen2ForCausalLM:
    """Build a `Qwen2ForCausalLM` from an HF snapshot directory, weights loaded, in eval mode."""
    config = ModelConfig.from_hf_dir(model_dir)
    model = Qwen2ForCausalLM(config).to(device=device, dtype=dtype)
    load_hf_weights(model, model_dir, dtype=dtype, device=device)
    return model.eval()
