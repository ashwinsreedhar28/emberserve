"""Llama / Mistral family support: config parsing, attention without bias, several eos ids,
Llama 3 RoPE frequency scaling, and a round trip through an HF-style llama checkpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pagedserve.config import ModelConfig
from pagedserve.model.qwen2 import LlamaForCausalLM, Qwen2ForCausalLM, reset_parameters_deterministic
from pagedserve.model.rope import RotaryEmbedding, llama3_scale_inv_freq
from pagedserve.model.weights import hf_state_dict, load_model
from pagedserve.sampling import check_stop
from pagedserve.sched.request import FinishReason, Request, SamplingParams

LLAMA3_SCALING = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                  "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}


def _write_config(tmp: Path, **cfg) -> Path:
    base = {"vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 512, "rms_norm_eps": 1e-5}
    base.update(cfg)
    (tmp / "config.json").write_text(json.dumps(base))
    return tmp


def test_llama_config_defaults(tmp_path: Path) -> None:
    cfg = ModelConfig.from_hf_dir(_write_config(
        tmp_path, model_type="llama", eos_token_id=[128001, 128008, 128009],
        bos_token_id=128000, rope_theta=500000.0, rope_scaling=LLAMA3_SCALING,
        tie_word_embeddings=True))
    assert cfg.model_type == "llama"
    assert cfg.attention_bias is False, "llama has no attention bias unless the config says so"
    assert cfg.eos_token_id == 128001 and cfg.all_eos_token_ids == {128001, 128008, 128009}
    assert cfg.rope_scaling == LLAMA3_SCALING and cfg.rope_theta == 500000.0


def test_mistral_config(tmp_path: Path) -> None:
    cfg = ModelConfig.from_hf_dir(_write_config(
        tmp_path, model_type="mistral", eos_token_id=2, bos_token_id=1, rope_theta=1000000.0,
        sliding_window=None, tie_word_embeddings=False))
    assert cfg.model_type == "mistral" and cfg.attention_bias is False
    assert cfg.tie_word_embeddings is False and cfg.all_eos_token_ids == {2}
    assert cfg.rope_scaling is None


def test_qwen2_config_keeps_bias_and_defaults(tmp_path: Path) -> None:
    cfg = ModelConfig.from_hf_dir(_write_config(tmp_path, model_type="qwen2", eos_token_id=151645))
    assert cfg.attention_bias is True and cfg.rope_theta == 1_000_000.0


@pytest.mark.parametrize("bad", [
    {"model_type": "gemma2", "eos_token_id": 1},
    {"model_type": "mistral", "eos_token_id": 2, "sliding_window": 4096},
    {"model_type": "llama", "eos_token_id": 2, "rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
    {"model_type": "llama", "eos_token_id": 2, "mlp_bias": True},
])
def test_unsupported_configs_raise(tmp_path: Path, bad: dict) -> None:
    with pytest.raises(ValueError):
        ModelConfig.from_hf_dir(_write_config(tmp_path, **bad))


def test_llama3_rope_scaling_bands() -> None:
    """High-frequency components untouched, low-frequency divided by `factor`, the band in
    between interpolated between the two; matches HF's `_compute_llama3_parameters`."""
    head_dim = 128
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv = 1.0 / (500000.0 ** exponent)
    scaled = llama3_scale_inv_freq(inv, LLAMA3_SCALING)
    wavelen = 2 * math.pi / inv
    high = wavelen < 8192 / 4.0
    low = wavelen > 8192 / 1.0
    torch.testing.assert_close(scaled[high], inv[high])
    torch.testing.assert_close(scaled[low], inv[low] / 32.0)
    mid = ~high & ~low
    assert mid.any()
    assert torch.all(scaled[mid] <= inv[mid]) and torch.all(scaled[mid] >= inv[mid] / 32.0)
    # Same result through the module.
    rope = RotaryEmbedding(head_dim, 512, 500000.0, rope_scaling=LLAMA3_SCALING)
    torch.testing.assert_close(rope.inv_freq(), scaled)
    plain = RotaryEmbedding(head_dim, 512, 500000.0)
    torch.testing.assert_close(plain.inv_freq(), inv)


def test_check_stop_accepts_eos_set() -> None:
    req = Request(request_id="r", prompt_token_ids=[1, 2], sampling_params=SamplingParams(max_tokens=8),
                  seq_id=0)
    req.output_token_ids = [7]
    assert check_stop(req, 7, frozenset({5, 7}), 4096) is FinishReason.STOP
    assert check_stop(req, 7, frozenset({5, 6}), 4096) is None
    assert check_stop(req, 7, 7, 4096) is FinishReason.STOP


def test_llama_checkpoint_round_trip(tmp_path: Path) -> None:
    """A tiny llama-family model (no attention bias, untied head, several eos ids) written
    under HF names and loaded back through `load_model`; logits identical."""
    cfg = ModelConfig.tiny(model_type="llama", attention_bias=False, tie_word_embeddings=False,
                           eos_token_id=3, eos_token_ids=(3, 5), rope_theta=500000.0,
                           rope_scaling=LLAMA3_SCALING)
    ref = LlamaForCausalLM(cfg)
    reset_parameters_deterministic(ref, 5)
    ref.eval()
    assert not any(n.endswith("qkv_proj.bias") for n, _ in ref.named_parameters())
    hf_cfg = {"model_type": "llama", "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
              "intermediate_size": cfg.intermediate_size, "num_hidden_layers": cfg.num_hidden_layers,
              "num_attention_heads": cfg.num_attention_heads,
              "num_key_value_heads": cfg.num_key_value_heads,
              "max_position_embeddings": cfg.max_position_embeddings, "rms_norm_eps": cfg.rms_norm_eps,
              "rope_theta": cfg.rope_theta, "rope_scaling": LLAMA3_SCALING,
              "tie_word_embeddings": False, "eos_token_id": [3, 5], "bos_token_id": 1}
    (tmp_path / "config.json").write_text(json.dumps(hf_cfg))
    save_file({k: v.detach().clone().contiguous() for k, v in hf_state_dict(ref).items()},
              str(tmp_path / "model.safetensors"))
    loaded = load_model(tmp_path)
    assert isinstance(loaded, Qwen2ForCausalLM) and loaded.config.attention_bias is False
    assert loaded.config.all_eos_token_ids == {3, 5}
    from pagedserve.attn.naive import NaiveAttentionBackend
    from tests.test_model import make_prefill_meta

    ids = torch.randint(0, cfg.vocab_size, (9,), generator=torch.Generator().manual_seed(2))
    meta = make_prefill_meta([0], [9])
    with torch.no_grad():
        a = ref.forward_logits_all(ids, NaiveAttentionBackend(cfg, device="cpu", dtype=torch.float32), meta)
        b = loaded.forward_logits_all(ids, NaiveAttentionBackend(cfg, device="cpu", dtype=torch.float32), meta)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
