"""DeepSeek-V3-style model (MLA + MoE) on the CPU: the absorbed paged attention against an
independent HF-style transcription (non-absorbed, full recompute, interleaved RoPE);
incremental decode == full forward; batched == alone; config parsing; checkpoint round trip."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pagedserve.config import EngineConfig, MLAConfig, ModelConfig, MoEConfig
from pagedserve.engine import LLMEngine
from pagedserve.llm import LLM
from pagedserve.model.deepseek import DeepseekForCausalLM, MLAAttention
from pagedserve.model.qwen2 import reset_parameters_deterministic
from pagedserve.model.weights import hf_state_dict, load_model
from pagedserve.sched.request import SamplingParams

torch.set_num_threads(2)

H, HID, NOPE, ROPE, VH, KVR = 4, 64, 16, 16, 16, 32


def tiny_cfg(q_lora_rank: int | None = None, layers: int = 3, moe: bool = True) -> ModelConfig:
    return ModelConfig.tiny(
        model_type="deepseek_v3", hidden_size=HID, num_attention_heads=H, num_key_value_heads=H,
        intermediate_size=96, num_hidden_layers=layers, attention_bias=False, rope_theta=10000.0,
        max_position_embeddings=256, tie_word_embeddings=True, eos_token_id=1,
        mla=MLAConfig(q_lora_rank=q_lora_rank, kv_lora_rank=KVR, qk_nope_head_dim=NOPE,
                      qk_rope_head_dim=ROPE, v_head_dim=VH),
        moe=MoEConfig(hidden_size=HID, moe_intermediate_size=32, n_routed_experts=4,
                      num_experts_per_tok=2, n_shared_experts=1, routed_scaling_factor=1.5) if moe else None,
        first_k_dense_replace=1 if moe else 0,
    )


def tiny_model(seed: int = 0, **kw) -> DeepseekForCausalLM:
    model = DeepseekForCausalLM(tiny_cfg(**kw))
    reset_parameters_deterministic(model, seed)
    return model.eval()


def make_engine(model: DeepseekForCausalLM, block_size: int = 4, num_blocks: int = 128) -> LLMEngine:
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=block_size,
                        num_gpu_blocks=num_blocks, max_num_seqs=16, max_num_batched_tokens=256,
                        max_model_len=128, attn_backend="mla_torch")
    return LLMEngine(model, model.config, ecfg, tokenizer=None)


# ---- reference: a transcription of HF DeepseekV3Attention, full sequence, no cache ----------

def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _rope_tables(positions: torch.Tensor, dim: int, base: float):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(positions.float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def reference_attention(attn: MLAAttention, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """HF semantics for one sequence `x: [T, hidden]`: materialize k_nope/v per head, apply
    DeepSeek's interleaved RoPE, dense causal softmax attention."""
    t = x.shape[0]
    w = attn.qkv_a_proj.weight  # HF's q_proj (or q_a_proj) and kv_a_proj_with_mqa, stacked
    q = x @ w[:attn.q_size].T
    if attn.q_lora_rank is not None:
        q = attn.q_b_proj(attn.q_a_layernorm(q))
    q = q.view(t, H, NOPE + ROPE)
    q_nope, q_pe = q[..., :NOPE], q[..., NOPE:]
    ckv = x @ w[attn.q_size:].T
    c, k_pe = ckv[:, :KVR], ckv[:, KVR:]
    c = attn.kv_a_layernorm(c)
    kv = attn.kv_b_proj(c).view(t, H, NOPE + VH)
    k_nope, v = kv[..., :NOPE], kv[..., NOPE:]
    cos, sin = _rope_tables(positions, ROPE, attn.rotary_emb.base)

    def rope(z):  # HF apply_rotary_pos_emb for deepseek: interleave permutation then rotate-half
        *lead, d = z.shape
        z = z.view(*lead, d // 2, 2).transpose(-1, -2).reshape(*lead, d)
        c_ = cos.view(t, 1, ROPE) if z.dim() == 3 else cos
        s_ = sin.view(t, 1, ROPE) if z.dim() == 3 else sin
        return z * c_ + _rotate_half(z) * s_

    q_pe = rope(q_pe)
    k_pe = rope(k_pe).unsqueeze(1).expand(t, H, ROPE)
    qf = torch.cat([q_nope, q_pe], -1)  # [T, H, 24]
    kf = torch.cat([k_nope, k_pe], -1)
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * (NOPE + ROPE) ** -0.5
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khv->qhv", p, v)
    return attn.o_proj(out.reshape(t, H * VH))


@pytest.mark.parametrize("q_lora_rank", [None, 24])
def test_attention_prefill_matches_hf_transcription(q_lora_rank):
    model = tiny_model(seed=1, q_lora_rank=q_lora_rank, layers=1, moe=False)
    attn = model.model.layers[0].self_attn
    engine = make_engine(model)
    t = 11
    x = torch.randn(t, HID, generator=torch.Generator().manual_seed(2))
    positions = torch.arange(t)
    want = reference_attention(attn, x, positions)
    # drive the backend like the engine: one sequence, blocks allocated for t tokens
    req = engine.add_request("r", list(range(2, 2 + t)), SamplingParams.greedy(1))
    so = engine.scheduler.schedule()
    _, meta = engine._build_inputs(so)
    with torch.no_grad():
        got = attn(x, engine.backend, meta)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    engine.abort_request(req.request_id)


def test_incremental_decode_consistent_with_full_forward():
    """Generate greedily through the engine (prefill, then one cached decode step per token)
    and check that a single full-sequence forward over prompt + output predicts exactly
    those tokens at every position: the latent cache, the absorbed scores and the rope
    positions of the incremental path agree with the from-scratch computation."""
    model = tiny_model(seed=3)
    prompt = torch.randint(2, 256, (9,), generator=torch.Generator().manual_seed(4)).tolist()
    out = LLM.from_engine(make_engine(model)).generate([prompt], SamplingParams.greedy(10, ignore_eos=True))[0]
    ids = prompt + out.output_token_ids
    e = make_engine(model)
    e.add_request("full", ids, SamplingParams.greedy(1))
    so = e.scheduler.schedule()
    input_ids, meta = e._build_inputs(so)
    with torch.inference_mode():
        full = model.forward_logits_all(input_ids, e.backend, meta).float()
    predicted = full.argmax(-1).tolist()
    for pos in range(len(prompt), len(ids)):
        top2 = full[pos - 1].topk(2).values
        assert top2[0] - top2[1] > 1e-3, "tie in the reference; pick another seed"
        assert predicted[pos - 1] == ids[pos], pos


def test_generate_batched_equals_alone():
    model = tiny_model(seed=5)
    prompts = [torch.randint(2, 256, (n,), generator=torch.Generator().manual_seed(n)).tolist()
               for n in (3, 9, 6, 12)]
    sp = SamplingParams.greedy(8, ignore_eos=True)
    together = LLM.from_engine(make_engine(model)).generate(prompts, sp)
    for p, out in zip(prompts, together):
        alone = LLM.from_engine(make_engine(model)).generate([p], sp)[0]
        assert out.output_token_ids == alone.output_token_ids


def test_moe_layers_placed_by_config():
    model = tiny_model(seed=6)
    from pagedserve.model.moe import DeepseekMoE
    from pagedserve.model.qwen2 import Qwen2MLP

    assert isinstance(model.model.layers[0].mlp, Qwen2MLP)  # first_k_dense_replace = 1
    assert isinstance(model.model.layers[1].mlp, DeepseekMoE)
    assert isinstance(model.model.layers[2].mlp, DeepseekMoE)


def test_cache_bytes_per_token():
    cfg = tiny_cfg()
    assert cfg.kv_bytes_per_token(torch.float16) == 3 * (KVR + ROPE) * 2


def test_config_from_hf_dir(tmp_path: Path):
    hf = {"model_type": "deepseek_v3", "vocab_size": 256, "hidden_size": HID, "intermediate_size": 96,
          "num_hidden_layers": 3, "num_attention_heads": H, "num_key_value_heads": H,
          "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
          "tie_word_embeddings": True, "eos_token_id": 1, "attention_bias": False,
          "q_lora_rank": None, "kv_lora_rank": KVR, "qk_nope_head_dim": NOPE, "qk_rope_head_dim": ROPE,
          "v_head_dim": VH, "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
          "moe_intermediate_size": 32, "first_k_dense_replace": 1, "moe_layer_freq": 1,
          "n_group": 1, "topk_group": 1, "norm_topk_prob": True, "routed_scaling_factor": 1.5,
          "scoring_func": "sigmoid", "topk_method": "noaux_tc"}
    (tmp_path / "config.json").write_text(json.dumps(hf))
    cfg = ModelConfig.from_hf_dir(tmp_path)
    assert cfg == tiny_cfg()


def test_checkpoint_round_trip(tmp_path: Path):
    ref = tiny_model(seed=7)
    sd = hf_state_dict(ref)
    assert "model.layers.0.self_attn.q_proj.weight" in sd  # HF names out of the fused qkv_a_proj
    assert "model.layers.0.self_attn.kv_a_proj_with_mqa.weight" in sd
    assert sd["model.layers.0.self_attn.q_proj.weight"].shape == (H * (NOPE + ROPE), HID)
    assert sd["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"].shape == (KVR + ROPE, HID)
    assert "model.layers.1.mlp.experts.3.down_proj.weight" in sd
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in sd
    assert "model.layers.0.mlp.gate_proj.weight" in sd  # dense first layer
    hf = {"model_type": "deepseek_v3", "vocab_size": 256, "hidden_size": HID, "intermediate_size": 96,
          "num_hidden_layers": 3, "num_attention_heads": H, "num_key_value_heads": H,
          "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
          "tie_word_embeddings": True, "eos_token_id": 1, "attention_bias": False,
          "q_lora_rank": None, "kv_lora_rank": KVR, "qk_nope_head_dim": NOPE, "qk_rope_head_dim": ROPE,
          "v_head_dim": VH, "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
          "moe_intermediate_size": 32, "first_k_dense_replace": 1, "routed_scaling_factor": 1.5}
    (tmp_path / "config.json").write_text(json.dumps(hf))
    tensors = {k: v.detach().clone().contiguous() for k, v in sd.items() if k != "lm_head.weight"}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    loaded = load_model(tmp_path)
    assert isinstance(loaded, DeepseekForCausalLM)
    assert loaded.rope_folded and not ref.rope_folded  # load_model folds the rope layout in
    sd2 = hf_state_dict(loaded)  # ... and the HF view is unchanged by it
    for name, t in sd.items():
        assert torch.equal(t, sd2[name]), name
    assert not torch.equal(ref.model.layers[0].self_attn.qkv_a_proj.weight,
                           loaded.model.layers[0].self_attn.qkv_a_proj.weight)
    ids = list(range(2, 12))
    a = LLM.from_engine(make_engine(ref)).generate([ids], SamplingParams.greedy(6, ignore_eos=True))[0]
    b = LLM.from_engine(make_engine(loaded)).generate([ids], SamplingParams.greedy(6, ignore_eos=True))[0]
    assert a.output_token_ids == b.output_token_ids


def test_scale_is_qk_head_dim():
    attn = tiny_model(seed=8, layers=1, moe=False).model.layers[0].self_attn
    assert math.isclose(attn.softmax_scale, (NOPE + ROPE) ** -0.5)


@pytest.mark.parametrize("q_lora_rank", [None, 24])
def test_qkv_a_round_trip_and_fold_with_q_lora(q_lora_rank, tmp_path: Path):
    """The fused first projection loads from HF's q_proj / q_a_proj + kv_a_proj_with_mqa
    names and exports them back; the rope fold lands in q_b_proj when q is low-rank."""
    ref = tiny_model(seed=9, q_lora_rank=q_lora_rank, layers=1, moe=False)
    sd = hf_state_dict(ref)
    q_name = "q_a_proj" if q_lora_rank else "q_proj"
    assert f"model.layers.0.self_attn.{q_name}.weight" in sd
    assert "model.layers.0.self_attn.qkv_a_proj.weight" not in sd
    hf = {"model_type": "deepseek_v3", "vocab_size": 256, "hidden_size": HID, "intermediate_size": 96,
          "num_hidden_layers": 1, "num_attention_heads": H, "num_key_value_heads": H,
          "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
          "tie_word_embeddings": True, "eos_token_id": 1, "attention_bias": False,
          "q_lora_rank": q_lora_rank, "kv_lora_rank": KVR, "qk_nope_head_dim": NOPE,
          "qk_rope_head_dim": ROPE, "v_head_dim": VH, "first_k_dense_replace": 0}
    (tmp_path / "config.json").write_text(json.dumps(hf))
    save_file({k: v.detach().clone().contiguous() for k, v in sd.items() if k != "lm_head.weight"},
              str(tmp_path / "model.safetensors"))
    loaded = load_model(tmp_path)
    assert loaded.rope_folded
    for name, t in hf_state_dict(loaded).items():
        assert torch.equal(t, sd[name]), name
    ids = list(range(2, 20))
    sp = SamplingParams.greedy(5, ignore_eos=True)
    a = LLM.from_engine(make_engine(ref)).generate([ids], sp)[0].output_token_ids
    b = LLM.from_engine(make_engine(loaded)).generate([ids], sp)[0].output_token_ids
    assert a == b


def test_rope_fold_is_exact():
    """Folding the halves permutation into the projections changes the weights, not the
    math: prompt logits agree to float precision and greedy tokens are identical; unfolding
    restores the original weights bit for bit."""
    m = tiny_model(seed=31)
    ids = list(range(2, 30))
    sp = SamplingParams.greedy(8, ignore_eos=True)

    def prompt_logits():
        e = make_engine(m)
        e.add_request("r", ids, SamplingParams.greedy(1))
        so = e.scheduler.schedule()
        input_ids, meta = e._build_inputs(so)
        with torch.inference_mode():
            return m.forward_logits_all(input_ids, e.backend, meta).clone()

    before_logits = prompt_logits()
    before = LLM.from_engine(make_engine(m)).generate([ids], sp)[0].output_token_ids
    w0 = m.model.layers[0].self_attn.qkv_a_proj.weight.clone()
    m.fold_rope_permutation()
    assert m.rope_folded
    assert not torch.equal(w0, m.model.layers[0].self_attn.qkv_a_proj.weight)
    torch.testing.assert_close(prompt_logits(), before_logits, atol=1e-5, rtol=1e-5)
    assert LLM.from_engine(make_engine(m)).generate([ids], sp)[0].output_token_ids == before
    m.fold_rope_permutation(False)
    assert not m.rope_folded and torch.equal(w0, m.model.layers[0].self_attn.qkv_a_proj.weight)
