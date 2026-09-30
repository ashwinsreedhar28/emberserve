"""Load HF Qwen2 safetensors checkpoints into `Qwen2ForCausalLM`.

Our module tree mirrors HF's, so the name mapping is the identity apart from a few
non-parameter tensors some snapshots carry (e.g. `model.rotary_emb.inv_freq`), which are
skipped. Loading is strict: unexpected checkpoint keys and never-loaded parameters both
raise, with the tied `lm_head.weight` as the only tolerated absence.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from pagedserve.config import ModelConfig
from pagedserve.model.qwen2 import Qwen2ForCausalLM

_SKIP_SUFFIXES = ("rotary_emb.inv_freq",)
# HF tensor name suffix -> (fused local suffix, shard id). The fused parameter's rows are
# [q | k | v] and [gate | up]; `_shard_view` computes the slice from the config / shape.
_FUSED = {
    "self_attn.q_proj": ("self_attn.qkv_proj", "q"),
    "self_attn.k_proj": ("self_attn.qkv_proj", "k"),
    "self_attn.v_proj": ("self_attn.qkv_proj", "v"),
    "mlp.gate_proj": ("mlp.gate_up_proj", "gate"),
    "mlp.up_proj": ("mlp.gate_up_proj", "up"),
    "shared_experts.gate_proj": ("shared_experts.gate_up_proj", "gate"),
    "shared_experts.up_proj": ("shared_experts.gate_up_proj", "up"),
}
# DeepSeek latent attention: q_proj (or q_a_proj) and kv_a_proj_with_mqa both read the hidden
# state, so they are one fused `qkv_a_proj` with rows [q | c | k_pe]; `_shard_view` splits it
# by the config's latent geometry.
_FUSED_MLA = {
    "self_attn.q_proj": ("self_attn.qkv_a_proj", "mla_q"),
    "self_attn.q_a_proj": ("self_attn.qkv_a_proj", "mla_q"),
    "self_attn.kv_a_proj_with_mqa": ("self_attn.qkv_a_proj", "mla_kv"),
}
# MoE routed experts: `...mlp.experts.<e>.{gate,up,down}_proj.weight` land in the stacked
# `...mlp.experts_gate_up [E, 2I, H]` / `...mlp.experts_down [E, H, I]` (model/moe.py).
_EXPERT_RE = re.compile(r"^(?P<prefix>.*\.mlp)\.experts\.(?P<e>\d+)\.(?P<part>gate|up|down)_proj\.weight$")

# A shard is None (whole tensor), one of "q"/"k"/"v"/"gate"/"up" (a row range of a fused
# 2-D parameter), or ("expert", e, part) (a slice of a stacked 3-D expert parameter).
Shard = str | tuple[str, int, str] | None


def hf_to_local(name: str, model_type: str = "qwen2") -> tuple[str, Shard] | None:
    """Map an HF checkpoint key to `(state_dict key, shard)`. None = skip the tensor.
    DeepSeek's latent attention keeps its own projection names (`q_proj` there is not a
    third of a fused qkv), so the attention fusion applies to the qwen2/llama families only."""
    if name.endswith(_SKIP_SUFFIXES):
        return None
    m = _EXPERT_RE.match(name)
    if m:
        e, part = int(m["e"]), m["part"]
        stacked = "experts_down" if part == "down" else "experts_gate_up"
        return f"{m['prefix']}.{stacked}", ("expert", e, part)
    head, _, leaf = name.rpartition(".")  # leaf: weight | bias
    deepseek = model_type.startswith("deepseek")
    if deepseek:
        for hf_suffix, (local_suffix, shard) in _FUSED_MLA.items():
            if head.endswith(hf_suffix):
                return f"{head[:-len(hf_suffix)]}{local_suffix}.{leaf}", shard
    for hf_suffix, (local_suffix, shard) in _FUSED.items():
        if deepseek and hf_suffix.startswith("self_attn."):
            continue
        if head.endswith(hf_suffix):
            return f"{head[:-len(hf_suffix)]}{local_suffix}.{leaf}", shard
    return name, None


def hf_to_local_name(name: str) -> str | None:
    """`state_dict` key an HF checkpoint key lands in (fused projections included)."""
    m = hf_to_local(name)
    return None if m is None else m[0]


def _shard_view(config: ModelConfig, param: torch.Tensor, shard: Shard) -> torch.Tensor:
    """The slice of `param` that `shard` names (a view; `copy_` into it loads the shard)."""
    if shard is None:
        return param
    if isinstance(shard, tuple):
        _, e, part = shard
        sub = param[e]
        if part == "down":
            return sub
        half = sub.shape[0] // 2
        return sub[:half] if part == "gate" else sub[half:]
    if shard in ("gate", "up"):  # fused gate_up: rows split in half, whatever the width
        half = param.shape[0] // 2
        return param[:half] if shard == "gate" else param[half:]
    if shard in ("mla_q", "mla_kv"):
        assert config.mla is not None
        kv_rows = config.mla.kv_lora_rank + config.mla.qk_rope_head_dim
        q_rows = param.shape[0] - kv_rows
        return param[:q_rows] if shard == "mla_q" else param[q_rows:]
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    rows = {"q": slice(0, q), "k": slice(q, q + kv), "v": slice(q + kv, q + 2 * kv)}[shard]
    return param[rows]


def _expected_shards(local: str, param: torch.Tensor) -> list[Shard]:
    """Every shard a parameter needs before it counts as loaded."""
    head = local.rpartition(".")[0]
    if local.endswith(".experts_gate_up"):
        return [("expert", e, p) for e in range(param.shape[0]) for p in ("gate", "up")]
    if local.endswith(".experts_down"):
        return [("expert", e, "down") for e in range(param.shape[0])]
    if head.endswith("qkv_proj"):
        return ["q", "k", "v"]
    if head.endswith("qkv_a_proj"):
        return ["mla_q", "mla_kv"]
    if head.endswith("gate_up_proj"):
        return ["gate", "up"]
    return [None]


def _hf_name(local: str, shard: Shard, config: ModelConfig | None = None) -> str:
    """Inverse of `hf_to_local` for error messages and `hf_state_dict`. `config` decides
    whether an MLA q shard was `q_proj` or `q_a_proj` (low-rank q)."""
    if shard is None:
        return local
    if isinstance(shard, tuple):
        _, e, part = shard
        prefix = local[:local.rindex(".mlp.") + len(".mlp")]
        return f"{prefix}.experts.{e}.{part}_proj.weight"
    head, _, leaf = local.rpartition(".")
    if shard in ("mla_q", "mla_kv"):
        base = head[:-len("self_attn.qkv_a_proj")]
        if shard == "mla_kv":
            return f"{base}self_attn.kv_a_proj_with_mqa.{leaf}"
        low_rank = config is not None and config.mla is not None and config.mla.q_lora_rank is not None
        return f"{base}self_attn.{'q_a_proj' if low_rank else 'q_proj'}.{leaf}"
    for hf_suffix, (local_suffix, sh) in _FUSED.items():
        if sh == shard and head.endswith(local_suffix):
            return f"{head[:-len(local_suffix)]}{hf_suffix}.{leaf}"
    return local  # pragma: no cover


def hf_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The model's parameters under HF names, fused projections and stacked experts split
    back apart (for writing checkpoints the HF loader, or this loader, can read)."""
    config = model.config
    folded = bool(getattr(model, "rope_folded", False))
    if folded:  # export the HF rope layout, then put the folded one back
        model.fold_rope_permutation(False)
    try:
        out: dict[str, torch.Tensor] = {}
        for local, tensor in model.state_dict().items():
            for shard in _expected_shards(local, tensor):
                t = _shard_view(config, tensor, shard)
                out[_hf_name(local, shard, config)] = t.clone() if folded else t
    finally:
        if folded:
            model.fold_rope_permutation(True)
    return out


def load_hf_weights(model: nn.Module, model_dir: str | os.PathLike,
                    dtype: torch.dtype | None = None,
                    device: torch.device | str | None = None) -> None:
    """Copy every tensor from `model_dir/*.safetensors` into `model` in place.

    Tensors are cast to `dtype` (default: the target parameter's dtype) and moved to
    `device` (default: the target parameter's device) during the copy. q/k/v, gate/up and
    per-expert tensors are copied into their slice of the fused / stacked parameter.
    `model.config` must carry the head geometry (a `ModelConfig`) and, when tied, whether
    `lm_head.weight` may be absent. Under tensor parallelism (`dist.get_tp()`), each
    checkpoint tensor is cut to this rank's slice first (`dist.shard_tensor`).
    """
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {model_dir}")
    from pagedserve import dist as tpdist

    if tpdist.get_tp().size == 1 and os.environ.get("PAGEDSERVE_LOADER", "stream") != "safetensors":
        from pagedserve.model.fastload import stream_weights

        dev = device if device is not None else next(model.parameters()).device
        model.load_stats = stream_weights(model, model_dir, dev)  # read by the boot phases
        return

    def tensors():
        for path in files:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for hf_name in f.keys():
                    yield hf_name, f.get_tensor(hf_name), path.name

    _load_tensors(model, tensors(), dtype, device, str(model_dir))


def load_hf_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor],
                       dtype: torch.dtype | None = None,
                       device: torch.device | str | None = None, tp=None) -> None:
    """`load_hf_weights` from an in-memory HF-layout state dict (e.g. `hf_state_dict()` of a
    full model, loaded into its tensor-parallel shards in the tests). `tp`: a `TPState`
    to shard for, default the process group's."""
    _load_tensors(model, ((k, v, "<state_dict>") for k, v in state_dict.items()),
                  dtype, device, "<state_dict>", tp)


def _load_tensors(model: nn.Module, tensors, dtype, device, source: str, tp=None) -> None:
    from pagedserve import dist as tpdist

    tp = tp or tpdist.get_tp()
    config = model.config
    state = model.state_dict()
    loaded: set[tuple[str, Shard]] = set()
    with torch.no_grad():
        for hf_name, src, where in tensors:
            m = hf_to_local(hf_name, getattr(config, "model_type", "qwen2"))
            if m is None:
                continue
            local, shard = m
            if local not in state:
                raise KeyError(f"unexpected checkpoint key {hf_name!r} in {where}")
            if (local, shard) in loaded:
                raise KeyError(f"duplicate checkpoint key {hf_name!r} in {where}")
            target = _shard_view(config, state[local], shard)
            src = tpdist.shard_tensor(hf_name, src, tp.rank, tp.size)
            if src.shape != target.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name!r}: checkpoint {tuple(src.shape)} "
                    f"vs model {tuple(target.shape)}")
            target.copy_(src.to(device=device or target.device, dtype=dtype or target.dtype))
            loaded.add((local, shard))

        tied = getattr(config, "tie_word_embeddings", False)
        if tied and ("lm_head.weight", None) not in loaded and tp.size > 1 \
                and "lm_head.weight" in state:
            # A shard cannot alias a slice of the (replicated) embedding: copy its rows.
            emb = state["model.embed_tokens.weight"]
            head = state["lm_head.weight"]
            rows = tpdist.shard_tensor("lm_head.weight", emb, tp.rank, tp.size) \
                if head.shape[0] != emb.shape[0] else emb
            head.copy_(rows)
            loaded.add(("lm_head.weight", None))

    missing = [_hf_name(k, sh, config) for k, t in state.items() for sh in _expected_shards(k, t)
               if not (tied and k == "lm_head.weight") and (k, sh) not in loaded]
    if missing:
        raise KeyError(f"parameters never loaded from {source}: {missing}")


def build_model(config: ModelConfig) -> nn.Module:
    """The module for a config's family: DeepSeek (MLA + MoE) or the Qwen2/Llama block."""
    if config.mla is not None:
        from pagedserve.model.deepseek import DeepseekForCausalLM

        return DeepseekForCausalLM(config)
    return Qwen2ForCausalLM(config)


def load_model(model_dir: str | os.PathLike, device: torch.device | str = "cpu",
               dtype: torch.dtype = torch.float32) -> nn.Module:
    """Build the model for an HF snapshot directory, weights loaded, in eval mode.

    Parameters are created directly on `device` in `dtype`: a 16B MoE model built in fp32 on
    the host first (the obvious `Model(config).to(device, dtype)`) needs 64 GB of RAM before
    a single weight is read. Under tensor parallelism the model is this rank's shard
    (`ModelConfig.shard`) and every checkpoint tensor is sliced as it is read."""
    from pagedserve import dist as tpdist

    config = ModelConfig.from_hf_dir(model_dir).shard(tpdist.get_tp().size)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = build_model(config)
    finally:
        torch.set_default_dtype(prev)
    load_hf_weights(model, model_dir, dtype=dtype, device=device)
    if hasattr(model, "fold_rope_permutation"):  # DeepSeek: rope layout into the weights
        model.fold_rope_permutation()
    return model.eval()
