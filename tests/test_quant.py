"""Weight-only int8: the quantizer, the torch reference GEMM, the module swap, and a
quantized tiny model running through the engine (CPU; the Triton kernel has its own
interpreter test in tests/test_int8_triton.py and GPU test in tests/test_int8_gpu.py)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from emberserve.config import EngineConfig, ModelConfig
from emberserve.engine import LLMEngine
from emberserve.llm import LLM
from emberserve.model.quant import Int8Linear, int8_gemm_torch, quantize_int8_weight, quantize_model
from emberserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from emberserve.sched.request import SamplingParams
from tests.test_deepseek import tiny_model as tiny_deepseek
from tests.test_engine import prompts

torch.set_num_threads(2)


def test_quantize_roundtrip_error_bound():
    w = torch.randn(64, 96, generator=torch.Generator().manual_seed(0)) * 0.1
    q, s = quantize_int8_weight(w)
    assert q.dtype == torch.int8 and s.dtype == torch.float32 and s.shape == (64,)
    deq = q.float() * s[:, None]
    assert (deq - w).abs().max() <= (s / 2 + 1e-6)[:, None].max()
    assert q.abs().max() == 127  # the max of every row hits the top code


def test_reference_gemm_matches_dequantized_linear():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(40, 96, generator=g) * 0.1
    x = torch.randn(7, 96, generator=g)
    b = torch.randn(40, generator=g)
    q, s = quantize_int8_weight(w)
    got = int8_gemm_torch(x, q, s, b)
    want = F.linear(x, q.float() * s[:, None], b)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_int8_linear_module_swap_and_forward():
    lin = nn.Linear(96, 40)
    torch.manual_seed(2)
    lin.weight.data.normal_(0, 0.1)
    q = Int8Linear.from_linear(lin)
    x = torch.randn(3, 5, 96)
    out = q(x)
    assert out.shape == (3, 5, 40)
    torch.testing.assert_close(out, F.linear(x, q.weight, q.bias), atol=1e-4, rtol=1e-4)
    # close to the fp32 layer, not equal
    assert (out - lin(x)).abs().max() < 0.05 * lin(x).abs().max()


def _engine(model, backend="paged_torch"):
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=4, num_gpu_blocks=256,
                        max_num_seqs=16, max_num_batched_tokens=512, max_model_len=256,
                        attn_backend=backend)
    return LLMEngine(model, model.config, ecfg, tokenizer=None)


def test_quantize_model_dense_and_run():
    cfg = ModelConfig.tiny()
    m = Qwen2ForCausalLM(cfg)
    reset_parameters_deterministic(m, 3)
    m.eval()
    ps = prompts(4, seed=5)
    sp = SamplingParams.greedy(8, ignore_eos=True)
    ref = LLM.from_engine(_engine(m)).generate(ps, sp)
    n = quantize_model(m)
    layers = cfg.num_hidden_layers
    assert n == layers * 4 + 1  # qkv, o, gate_up, down per layer + lm_head
    assert isinstance(m.lm_head, Int8Linear) and isinstance(m.model.embed_tokens, nn.Embedding)
    assert isinstance(m.model.layers[0].self_attn.qkv_proj, Int8Linear)
    assert m.model.layers[0].self_attn.qkv_proj.bias is not None  # Qwen2's qkv bias kept
    out = LLM.from_engine(_engine(m)).generate(ps, sp)
    assert all(len(o.output_token_ids) == 8 for o in out)
    # a quality check, not exactness: most first tokens agree with the fp32 model
    agree = sum(o.output_token_ids[0] == r.output_token_ids[0] for o, r in zip(out, ref))
    assert agree >= len(ps) // 2


def test_quantize_model_deepseek_skips_kv_b_and_experts():
    m = tiny_deepseek(seed=4)
    n = quantize_model(m)
    attn = m.model.layers[0].self_attn
    assert isinstance(attn.qkv_a_proj, Int8Linear) and isinstance(attn.o_proj, Int8Linear)
    assert isinstance(attn.kv_b_proj, nn.Linear)  # read directly by the absorbed attention
    moe = m.model.layers[1].mlp
    assert moe.experts_gate_up.dtype == torch.float32  # stacked experts untouched
    assert isinstance(moe.shared_experts.down_proj, Int8Linear)
    assert n > 0
    ps = [list(range(2, 20)), list(range(30, 45))]
    out = LLM.from_engine(_engine(m, "mla_torch")).generate(ps, SamplingParams.greedy(6, ignore_eos=True))
    assert all(len(o.output_token_ids) == 6 for o in out)
