"""Round-trip a tiny model through an HF-style safetensors snapshot and `load_model`."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from emberserve.attn.naive import NaiveAttentionBackend
from emberserve.config import ModelConfig
from emberserve.model.qwen2 import Qwen2ForCausalLM
from emberserve.model.weights import hf_state_dict, hf_to_local, hf_to_local_name, load_hf_weights, load_model
from tests.test_model import make_prefill_meta, tiny_model

torch.set_num_threads(2)


def _hf_config_json(cfg: ModelConfig) -> dict:
    return {
        "model_type": "qwen2",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "eos_token_id": cfg.eos_token_id,
    }


def _dump_snapshot(model: Qwen2ForCausalLM, out_dir: Path, drop: Collection[str] = (),
                   split: bool = False) -> None:
    """Write `config.json` + safetensors in HF layout. HF omits `lm_head.weight` when
    tied; we add a bogus `model.rotary_emb.inv_freq` that the loader must skip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(_hf_config_json(model.config)))
    tensors = {k: v.detach().clone().contiguous() for k, v in hf_state_dict(model).items()
               if k not in drop}
    if model.config.tie_word_embeddings:
        tensors.pop("lm_head.weight", None)
    tensors["model.rotary_emb.inv_freq"] = torch.full((model.config.head_dim // 2,), 123.0)
    if split:
        keys = sorted(tensors)
        half = len(keys) // 2
        save_file({k: tensors[k] for k in keys[:half]}, str(out_dir / "model-00001-of-00002.safetensors"))
        save_file({k: tensors[k] for k in keys[half:]}, str(out_dir / "model-00002-of-00002.safetensors"))
    else:
        save_file(tensors, str(out_dir / "model.safetensors"))


@torch.no_grad()
def _all_logits(model: Qwen2ForCausalLM, prompt: torch.Tensor) -> torch.Tensor:
    backend = NaiveAttentionBackend(model.config, device="cpu", dtype=torch.float32)
    return model.forward_logits_all(prompt, backend, make_prefill_meta([0], [len(prompt)]))


def _assert_same_model(loaded: Qwen2ForCausalLM, ref: Qwen2ForCausalLM) -> None:
    assert not loaded.training
    ref_sd, loaded_sd = ref.state_dict(), loaded.state_dict()
    assert set(ref_sd) == set(loaded_sd)
    for name, tensor in ref_sd.items():
        assert torch.equal(loaded_sd[name], tensor), name
    prompt = torch.randint(0, ref.config.vocab_size, (10,), generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(_all_logits(loaded, prompt), _all_logits(ref, prompt), atol=1e-6, rtol=0)


def test_hf_to_local_name() -> None:
    assert hf_to_local("model.layers.3.self_attn.q_proj.bias") == ("model.layers.3.self_attn.qkv_proj.bias", "q")
    assert hf_to_local("model.layers.3.self_attn.v_proj.weight") == ("model.layers.3.self_attn.qkv_proj.weight", "v")
    assert hf_to_local("model.layers.0.mlp.up_proj.weight") == ("model.layers.0.mlp.gate_up_proj.weight", "up")
    assert hf_to_local_name("model.layers.0.mlp.gate_proj.weight") == "model.layers.0.mlp.gate_up_proj.weight"
    assert hf_to_local_name("lm_head.weight") == "lm_head.weight"
    assert hf_to_local_name("model.rotary_emb.inv_freq") is None


def test_hf_state_dict_splits_fused_projections() -> None:
    model = tiny_model(seed=3)
    sd = hf_state_dict(model)
    attn = model.model.layers[0].self_attn
    q, k, v = attn.qkv_proj.weight.split([attn.q_size, attn.kv_size, attn.kv_size], dim=0)
    assert torch.equal(sd["model.layers.0.self_attn.q_proj.weight"], q)
    assert torch.equal(sd["model.layers.0.self_attn.k_proj.weight"], k)
    assert torch.equal(sd["model.layers.0.self_attn.v_proj.weight"], v)
    mlp = model.model.layers[0].mlp
    assert torch.equal(sd["model.layers.0.mlp.up_proj.weight"],
                       mlp.gate_up_proj.weight[mlp.intermediate_size:])
    assert not any("qkv_proj" in k or "gate_up_proj" in k for k in sd)


@pytest.mark.parametrize("split", [False, True])
def test_round_trip_tied(tmp_path: Path, split: bool) -> None:
    ref = tiny_model(seed=11, tie_word_embeddings=True)
    _dump_snapshot(ref, tmp_path, split=split)
    loaded = load_model(tmp_path)
    assert loaded.config == ref.config
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    _assert_same_model(loaded, ref)


def test_round_trip_untied(tmp_path: Path) -> None:
    ref = tiny_model(seed=12, tie_word_embeddings=False)
    _dump_snapshot(ref, tmp_path)
    loaded = load_model(tmp_path)
    assert loaded.config.tie_word_embeddings is False
    assert loaded.lm_head.weight is not loaded.model.embed_tokens.weight
    _assert_same_model(loaded, ref)


def test_dtype_cast_on_load(tmp_path: Path) -> None:
    ref = tiny_model(seed=13)
    _dump_snapshot(ref, tmp_path)
    loaded = load_model(tmp_path, dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in loaded.parameters())
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight, "tie must survive .to()"
    torch.testing.assert_close(loaded.model.embed_tokens.weight.float(),
                               ref.model.embed_tokens.weight.to(torch.bfloat16).float())


def test_missing_param_raises(tmp_path: Path) -> None:
    ref = tiny_model(seed=14)
    _dump_snapshot(ref, tmp_path, drop={"model.layers.1.mlp.up_proj.weight"})
    with pytest.raises(KeyError, match="model.layers.1.mlp.up_proj.weight"):
        load_model(tmp_path)


def test_missing_untied_lm_head_raises(tmp_path: Path) -> None:
    ref = tiny_model(seed=15, tie_word_embeddings=False)
    _dump_snapshot(ref, tmp_path, drop={"lm_head.weight"})
    with pytest.raises(KeyError, match="lm_head.weight"):
        load_model(tmp_path)


def test_unexpected_key_raises(tmp_path: Path) -> None:
    ref = tiny_model(seed=16)
    _dump_snapshot(ref, tmp_path)
    (tmp_path / "model.safetensors").unlink()
    tensors = {k: v.clone() for k, v in ref.state_dict().items() if k != "lm_head.weight"}
    tensors["model.layers.0.self_attn.bogus.weight"] = torch.zeros(2, 2)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(KeyError, match="bogus"):
        load_hf_weights(Qwen2ForCausalLM(ref.config), tmp_path)


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    ref = tiny_model(seed=17)
    _dump_snapshot(ref, tmp_path)
    wrong = Qwen2ForCausalLM(ModelConfig.tiny(intermediate_size=96))
    with pytest.raises(ValueError, match="shape mismatch"):
        load_hf_weights(wrong, tmp_path)


def test_no_safetensors_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_hf_weights(tiny_model(), tmp_path)
