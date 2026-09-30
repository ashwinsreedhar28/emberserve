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
                max_model_len: int = 256, enable_chunked_prefill: bool = False,
                enable_prefix_caching: bool = False) -> LLMEngine:
    model = Qwen2ForCausalLM(CFG)
    reset_parameters_deterministic(model, seed)
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=block_size,
                        num_gpu_blocks=num_blocks, max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=max_batched, max_model_len=max_model_len,
                        attn_backend=backend, enable_chunked_prefill=enable_chunked_prefill,
                        enable_prefix_caching=enable_prefix_caching)
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


# ---- chunked prefill ----------------------------------------------------------------

def _check_token_split(eng: LLMEngine, budget: int | None = None) -> None:
    assert eng.stats, "no steps recorded"
    for s in eng.stats:
        assert s.num_prefill_tokens + s.num_decode_tokens == s.num_tokens, s
        if budget is not None:
            assert s.num_tokens <= budget, s


@pytest.mark.parametrize("backend", ["naive", "paged_torch"])
def test_chunked_prefill_matches_unchunked(backend: str) -> None:
    """6 prompts of 3-20 tokens through an 8-token budget: chunks straddle the budget,
    several prompts share steps, and the outputs must equal the unchunked run."""
    ps = prompts(6)
    sp = SamplingParams.greedy(12, ignore_eos=True)
    eng = make_engine(backend, max_batched=8, enable_chunked_prefill=True)
    res = LLM.from_engine(eng).generate(ps, sp)
    ref = LLM.from_engine(make_engine(backend)).generate(ps, sp)
    assert [r.output_token_ids for r in res] == [r.output_token_ids for r in ref]
    assert all(r.finish_reason == FinishReason.LENGTH for r in res)
    assert any(s.num_prefill_tokens > 0 and s.num_decode_tokens > 0 for s in eng.stats), \
        "expected at least one mixed prefill+decode step"
    _check_token_split(eng, budget=8)
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks


@pytest.mark.parametrize("backend", ["naive", "paged_torch"])
def test_chunked_prefill_mixed_with_decoding_requests(backend: str) -> None:
    """4 short requests decoding, then a 40-token prompt with budget 8: no step exceeds
    the budget, decodes keep flowing, every output equals `run_alone`."""
    budget = 8
    eng = make_engine(backend, max_batched=budget, enable_chunked_prefill=True)
    shorts = [p[:2] for p in prompts(4, seed=5)]
    g = torch.Generator().manual_seed(9)
    long_p = torch.randint(2, CFG.vocab_size, (40,), generator=g).tolist()
    sp = SamplingParams.greedy(20, ignore_eos=True)
    for i, p in enumerate(shorts):
        eng.add_request(f"s{i}", p, sp)
    eng.step()  # prefill all four (8 tokens == budget)
    assert eng.stats[-1].num_prefill_tokens == 8
    eng.step()  # pure decode
    assert eng.stats[-1].num_decode_tokens == 4 and not eng.stats[-1].is_prefill
    eng.add_request("long", long_p, SamplingParams.greedy(6, ignore_eos=True))
    long_req = eng.scheduler.get_request("long")
    collected: dict[str, list[int]] = {}
    chunk_steps = 0
    while long_req.num_computed_tokens < len(long_p):
        for o in eng.step():
            if o.finished:
                collected[o.request_id] = o.output_token_ids
        st = eng.stats[-1]
        chunk_steps += 1
        assert st.num_decode_tokens == 4 and 1 <= st.num_prefill_tokens <= 4
        assert st.is_prefill
    assert chunk_steps == -(-len(long_p) // 4)
    assert len(long_req.output_token_ids) == 1  # sampled once, at the last chunk
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                collected[o.request_id] = o.output_token_ids
    _check_token_split(eng, budget=budget)
    for i, p in enumerate(shorts):
        assert collected[f"s{i}"] == run_alone(backend, p, 20)
    assert collected["long"] == run_alone(backend, long_p, 6)


def test_chunked_prefill_with_prefix_caching() -> None:
    """A shared 16-token prefix (4 blocks of 4) with chunking: the second request skips
    the cached blocks and chunks only its tail; outputs equal the plain run."""
    g = torch.Generator().manual_seed(21)
    shared = torch.randint(2, CFG.vocab_size, (16,), generator=g).tolist()
    tails = [torch.randint(2, CFG.vocab_size, (n,), generator=g).tolist() for n in (9, 13, 5)]
    ps = [shared + t for t in tails]
    sp = SamplingParams.greedy(8, ignore_eos=True)
    ref = LLM.from_engine(make_engine("paged_torch")).generate(ps, sp)
    eng = make_engine("paged_torch", max_batched=6, block_size=4, enable_chunked_prefill=True,
                      enable_prefix_caching=True)
    llm = LLM.from_engine(eng)
    first = llm.generate([ps[0]], sp)
    assert first[0].output_token_ids == ref[0].output_token_ids
    prefill_before = sum(s.num_prefill_tokens for s in eng.stats)
    assert prefill_before == len(ps[0])
    rest = llm.generate(ps[1:], sp)
    assert [r.output_token_ids for r in rest] == [r.output_token_ids for r in ref[1:]]
    prefill_after = sum(s.num_prefill_tokens for s in eng.stats) - prefill_before
    # Each later prompt hits the 4 shared blocks (16 tokens) and chunks only its tail.
    assert prefill_after == sum(len(p) - 16 for p in ps[1:])
    assert eng.block_manager.stats().prefix_cache.hits >= 8
    _check_token_split(eng, budget=6)


@pytest.mark.parametrize("chunked", [False, True])
def test_step_stats_token_split(chunked: bool) -> None:
    eng = make_engine("paged_torch", max_batched=16, enable_chunked_prefill=chunked)
    LLM.from_engine(eng).generate(prompts(5, seed=13), SamplingParams.greedy(6, ignore_eos=True))
    _check_token_split(eng, budget=16 if chunked else None)
    assert sum(s.num_decode_tokens for s in eng.stats) == 5 * 5  # 5 reqs x (6 - 1) decodes
    assert sum(s.num_prefill_tokens for s in eng.stats) == sum(len(p) for p in prompts(5, seed=13))


def test_kv_budget_does_not_subtract_resident_weights() -> None:
    """`free` is read after the weights are loaded; charging them again put the 7B on a
    24 GB card at the 64-block floor (the Serverless 4090 run: ~30 sequences, 1,118 tok/s),
    and a hand-set 512 blocks on the same card OOMed at graph capture, so the reserve has
    to cover the busiest step's activations."""
    from pagedserve.engine import (KV_WORKSPACE_BYTES, MIN_GPU_BLOCKS, activation_reserve_bytes,
                                   kv_blocks_for)

    qwen7b = ModelConfig(vocab_size=152064, hidden_size=3584, intermediate_size=18944,
                         num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4)
    cfg = EngineConfig(device="cuda", dtype=torch.float16, block_size=256, max_num_seqs=256,
                       max_num_batched_tokens=2048, enable_chunked_prefill=True)  # the CLI's 7B setup
    bytes_per_block = qwen7b.kv_bytes_per_token(torch.float16) * 256
    assert bytes_per_block == 28 * 2 * 4 * 128 * 2 * 256  # 14.0 MiB per 256-token block
    reserve = activation_reserve_bytes(qwen7b, cfg)
    assert reserve == KV_WORKSPACE_BYTES + 2048 * 18944 * 2 * 2 + 256 * 152064 * 6
    # RTX 4090 as the container sees it: 22.04 GiB, 14.18 GiB of weights, ~0.55 GiB context.
    free_after_load = int(7.3 * (1 << 30))
    blocks = kv_blocks_for(free_after_load, 0.90, bytes_per_block, reserve)
    assert 300 <= blocks < 512
    assert free_after_load - blocks * bytes_per_block >= reserve  # room for capture + prefill
    assert 512 * bytes_per_block > free_after_load - (512 << 20)  # what OOMed: 71 MiB left
    old_formula = max((int(free_after_load * 0.90) - (15_231 << 20) - (512 << 20))
                      // bytes_per_block, MIN_GPU_BLOCKS)
    assert old_formula == MIN_GPU_BLOCKS
    assert kv_blocks_for(0, 0.90, bytes_per_block, reserve) == MIN_GPU_BLOCKS
    # 80 GB A100 after the same weights: most of the card, as before.
    assert kv_blocks_for(int(64 * (1 << 30)), 0.90, bytes_per_block, reserve) > 3900
    # Without chunking a preempted request re-prefills its whole history (up to
    # max_model_len) in one step, over the budget: the reserve covers that prefill.
    unchunked = EngineConfig(device="cuda", dtype=torch.float16, block_size=256, max_num_seqs=256,
                             max_num_batched_tokens=512, max_model_len=32768)
    assert activation_reserve_bytes(qwen7b, unchunked) == \
        KV_WORKSPACE_BYTES + 32768 * 18944 * 2 * 2 + 256 * 152064 * 6


def test_finished_text_retention_is_bounded_and_reset(monkeypatch) -> None:
    """Every finished request's text used to stay in `_final_text` forever (nothing in the
    server or LLM.generate pops it) and survived `reset()`."""
    from pagedserve import engine as engine_mod
    from pagedserve.sched.request import SamplingParams

    monkeypatch.setattr(engine_mod, "FINAL_TEXT_KEEP", 5)
    eng = make_engine()
    for i, p in enumerate(prompts(12)):
        eng.add_request(f"r{i}", p, SamplingParams.greedy(2, ignore_eos=True))
    while eng.has_unfinished_requests():
        eng.step()
    assert list(eng._final_text) == [f"r{i}" for i in range(7, 12)]
    eng.reset()
    assert not eng._final_text
