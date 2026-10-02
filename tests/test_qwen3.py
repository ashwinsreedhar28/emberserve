"""Qwen3 (dense) against Hugging Face's own implementation: a tiny random Qwen3 saved with
`save_pretrained` (per-head q/k RMSNorm, no attention bias, head_dim != hidden / heads,
untied head), loaded by `load_model`, gives the same logits at every position. Skipped
when `transformers` is not installed."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from emberserve.attn.naive import NaiveAttentionBackend
from emberserve.config import ModelConfig
from emberserve.model.weights import load_model
from tests.test_model import make_prefill_meta

transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("tied", [False, True])
def test_qwen3_matches_hf(tmp_path: Path, tied: bool) -> None:
    cfg = transformers.Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,  # 4 x 32 != 64
        max_position_embeddings=256, rms_norm_eps=1e-6, rope_theta=1_000_000.0,
        tie_word_embeddings=tied, attention_bias=False, eos_token_id=1, torch_dtype="float32")
    torch.manual_seed(0)
    hf = transformers.Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():  # random q/k norm weights, so a missing norm cannot pass
        for layer in hf.model.layers:
            layer.self_attn.q_norm.weight.uniform_(0.5, 1.5)
            layer.self_attn.k_norm.weight.uniform_(0.5, 1.5)
    hf.save_pretrained(tmp_path, safe_serialization=True)
    mc = ModelConfig.from_hf_dir(tmp_path)
    assert mc.model_type == "qwen3" and mc.qk_norm and mc.head_dim == 32 and not mc.attention_bias
    ours = load_model(tmp_path)
    ids = torch.randint(2, 256, (13,), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = hf(ids[None]).logits[0]
        got = ours.forward_logits_all(ids, NaiveAttentionBackend(mc, device="cpu", dtype=torch.float32),
                                      make_prefill_meta([0], [len(ids)]))
    torch.testing.assert_close(got, ref, atol=2e-5, rtol=1e-4)


def test_qwen3_engine_batch_equals_alone() -> None:
    """The q/k norm path through the paged engine: a batch of prompts decodes exactly as
    each prompt alone (continuous batching, paged KV)."""
    from emberserve.config import EngineConfig
    from emberserve.engine import LLMEngine
    from emberserve.llm import LLM
    from emberserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
    from emberserve.sched.request import SamplingParams

    cfg = ModelConfig.tiny(model_type="qwen3", qk_norm=True, head_dim=32, attention_bias=False)

    def engine() -> LLMEngine:
        m = Qwen2ForCausalLM(cfg)
        reset_parameters_deterministic(m, 3)
        ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=4, num_gpu_blocks=256,
                            max_num_seqs=16, max_num_batched_tokens=256, max_model_len=256,
                            attn_backend="paged_torch")
        return LLMEngine(m, cfg, ecfg, tokenizer=None)

    g = torch.Generator().manual_seed(4)
    ps = [torch.randint(2, 256, (int(n),), generator=g).tolist() for n in (5, 9, 13)]
    sp = SamplingParams.greedy(8, ignore_eos=True)
    together = LLM.from_engine(engine()).generate(ps, sp)
    for p, r in zip(ps, together):
        assert r.output_token_ids == LLM.from_engine(engine()).generate([p], sp)[0].output_token_ids
