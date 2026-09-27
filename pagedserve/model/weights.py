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


def hf_to_local_name(name: str) -> str | None:
    """Map an HF checkpoint key to our `state_dict` key; `None` means skip the tensor."""
    if name.endswith(_SKIP_SUFFIXES):
        return None
    return name


def load_hf_weights(model: Qwen2ForCausalLM, model_dir: str | os.PathLike,
                    dtype: torch.dtype | None = None,
                    device: torch.device | str | None = None) -> None:
    """Copy every tensor from `model_dir/*.safetensors` into `model` in place.

    Tensors are cast to `dtype` (default: the target parameter's dtype) and moved to
    `device` (default: the target parameter's device) during the copy.
    """
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {model_dir}")

    state = model.state_dict()
    loaded: set[str] = set()
    with torch.no_grad():
        for path in files:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for hf_name in f.keys():
                    local = hf_to_local_name(hf_name)
                    if local is None:
                        continue
                    if local not in state:
                        raise KeyError(f"unexpected checkpoint key {hf_name!r} in {path.name}")
                    if local in loaded:
                        raise KeyError(f"duplicate checkpoint key {hf_name!r} in {path.name}")
                    target = state[local]
                    src = f.get_tensor(hf_name)
                    if src.shape != target.shape:
                        raise ValueError(
                            f"shape mismatch for {hf_name!r}: checkpoint {tuple(src.shape)} "
                            f"vs model {tuple(target.shape)}")
                    target.copy_(src.to(device=device or target.device,
                                        dtype=dtype or target.dtype))
                    loaded.add(local)

    tied = model.config.tie_word_embeddings
    missing = [k for k in state if k not in loaded and not (tied and k == "lm_head.weight")]
    if missing:
        raise KeyError(f"parameters never loaded from {model_dir}: {missing}")


def load_model(model_dir: str | os.PathLike, device: torch.device | str = "cpu",
               dtype: torch.dtype = torch.float32) -> Qwen2ForCausalLM:
    """Build a `Qwen2ForCausalLM` from an HF snapshot directory, weights loaded, in eval mode."""
    config = ModelConfig.from_hf_dir(model_dir)
    model = Qwen2ForCausalLM(config).to(device=device, dtype=dtype)
    load_hf_weights(model, model_dir, dtype=dtype, device=device)
    return model.eval()
