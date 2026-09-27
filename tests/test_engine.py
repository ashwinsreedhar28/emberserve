"""End-to-end LLMEngine tests on a tiny deterministic random model (no weights, no GPU).

These are the integration gates: continuous batching + paged cache + preemption must
produce exactly the tokens a single request alone would produce.
"""

from __future__ import annotations

import pytest
import torch

from pagedserve.config import EngineConfig, ModelConfig
from pagedserve.engine import LLMEngine
from pagedserve.llm import LLM
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from pagedserve.sched.request import FinishReason, SamplingParams

torch.set_num_threads(2)

CFG = ModelConfig.tiny()


def make_engine(backend: str = "paged_torch", num_blocks: int = 256, block_size: int = 4,
                max_num_seqs: int = 64, max_batched: int = 512, seed: int = 0,
                max_model_len: int = 256) -> LLMEngine:
    model = Qwen2ForCausalLM(CFG)
    reset_parameters_deterministic(model, seed)
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=block_size,
                        num_gpu_blocks=num_blocks, max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=max_batched, max_model_len=max_model_len,
                        attn_backend=backend)
    return LLMEngine(model, CFG, ecfg, tokenizer=None)


def prompts(n: int, seed: int = 1) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        length = int(torch.randint(3, 20, (1,), generator=g))
        # avoid eos (1) in prompts
        out.append((torch.randint(2, CFG.vocab_size, (length,), generator=g)).tolist())
    return out


def run_alone(backend: str, prompt: list[int], max_tokens: int, **kw) -> list[int]:
    eng = make_engine(backend, **kw)
    res = LLM.from_engine(eng).generate([prompt], SamplingParams.greedy(max_tokens, ignore_eos=True))
    return res[0].output_token_ids


@pytest.mark.parametrize("backend", ["naive", "paged_torch"])
def test_batch_equals_alone(backend: str) -> None:
    ps = prompts(6)
    eng = make_engine(backend)
    res = LLM.from_engine(eng).generate(ps, SamplingParams.greedy(16, ignore_eos=True))
    for p, r in zip(ps, res):
        assert r.output_token_ids == run_alone(backend, p, 16)
        assert r.finish_reason == FinishReason.LENGTH
        assert len(r.output_token_ids) == 16
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks


def test_naive_and_paged_agree() -> None:
    ps = prompts(4, seed=7)
    a = LLM.from_engine(make_engine("naive")).generate(ps, SamplingParams.greedy(20, ignore_eos=True))
    b = LLM.from_engine(make_engine("paged_torch")).generate(ps, SamplingParams.greedy(20, ignore_eos=True))
    assert [x.output_token_ids for x in a] == [x.output_token_ids for x in b]


def test_preemption_preserves_outputs() -> None:
    """A block budget too small for all requests at once forces recompute-preemption;
    outputs must still be identical to running each request alone."""
    ps = prompts(5, seed=3)
    max_tokens = 24
    # Each request needs up to ceil((20 + 24) / 4) = 11 blocks; 5 requests need 55. Give 20.
    eng = make_engine("paged_torch", num_blocks=20, block_size=4)
    res = LLM.from_engine(eng).generate(ps, SamplingParams.greedy(max_tokens, ignore_eos=True))
    assert any(s.num_preempted > 0 for s in eng.stats), "expected at least one preemption"
    for p, r in zip(ps, res):
        assert r.output_token_ids == run_alone("paged_torch", p, max_tokens)
    assert eng.block_manager.num_free_blocks == 20


def test_continuous_admission_mid_stream() -> None:
    eng = make_engine("paged_torch")
    ps = prompts(3, seed=11)
    sp = SamplingParams.greedy(12, ignore_eos=True)
    eng.add_request("a", ps[0], sp)
    eng.add_request("b", ps[1], sp)
    eng.step()  # prefill a, b
    eng.step()  # decode
    eng.add_request("c", ps[2], sp)
    out = eng.step()  # prefill priority: c gets prefilled now
    assert [o.request_id for o in out] == ["c"]
    collected: dict[str, list[int]] = {}
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                collected[o.request_id] = o.output_token_ids
    for rid, p in zip("abc", ps):
        assert collected[rid] == run_alone("paged_torch", p, 12)


def test_eos_and_stop_token_ids() -> None:
    eng = make_engine("paged_torch")
    p = prompts(1)[0]
    # Find what greedy generates, then use its 3rd token as a stop token.
    ref = run_alone("paged_torch", p, 10)
    stop = ref[2]
    sp = SamplingParams.greedy(10, stop_token_ids=[stop])
    res = LLM.from_engine(eng).generate([p], sp)[0]
    assert res.output_token_ids == ref[: ref.index(stop) + 1]
    assert res.finish_reason == FinishReason.STOP


def test_abort_frees_blocks() -> None:
    eng = make_engine("paged_torch", num_blocks=32)
    ps = prompts(2)
    eng.add_request("a", ps[0], SamplingParams.greedy(50, ignore_eos=True))
    eng.add_request("b", ps[1], SamplingParams.greedy(50, ignore_eos=True))
    eng.step()
    eng.step()
    used_before = eng.block_manager.num_blocks - eng.block_manager.num_free_blocks
    eng.abort_request("a")
    assert eng.block_manager.num_blocks - eng.block_manager.num_free_blocks < used_before
    while eng.has_unfinished_requests():
        eng.step()
    assert eng.block_manager.num_free_blocks == 32


def test_max_model_len_terminates() -> None:
    eng = make_engine("paged_torch", max_model_len=32)
    p = prompts(1)[0]
    res = LLM.from_engine(eng).generate([p], SamplingParams.greedy(100, ignore_eos=True))[0]
    assert len(p) + len(res.output_token_ids) == 32
    assert res.finish_reason == FinishReason.LENGTH


def test_sampled_seeded_reproducible() -> None:
    p = prompts(1)[0]
    sp = SamplingParams(max_tokens=16, temperature=0.8, top_p=0.9, seed=123, ignore_eos=True)
    a = LLM.from_engine(make_engine()).generate([p], sp)[0].output_token_ids
    b = LLM.from_engine(make_engine()).generate([p], sp)[0].output_token_ids
    assert a == b
    sp2 = SamplingParams(max_tokens=16, temperature=0.8, top_p=0.9, seed=124, ignore_eos=True)
    c = LLM.from_engine(make_engine()).generate([p], sp2)[0].output_token_ids
    assert c != a


def test_step_stats_recorded() -> None:
    eng = make_engine()
    LLM.from_engine(eng).generate(prompts(2), SamplingParams.greedy(4, ignore_eos=True))
    assert eng.stats[0].is_prefill and not eng.stats[1].is_prefill
    assert all(0.0 < s.kv_utilization <= 1.0 for s in eng.stats)
