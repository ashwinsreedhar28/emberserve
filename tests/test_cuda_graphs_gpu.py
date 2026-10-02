"""CUDA-graph decode must reproduce eager decode exactly (GPU only).

Tiny deterministic model, fp16 on CUDA, `paged_flash` backend (block_size 256 because
of flash-attn's paged page-size constraint). Run: `python -m pytest -m gpu -q`.
"""

from __future__ import annotations

import pytest
import torch

from emberserve.config import EngineConfig, ModelConfig
from emberserve.engine import LLMEngine
from emberserve.llm import LLM
from emberserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from emberserve.sched.request import SamplingParams

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("flash_attn")

CFG = ModelConfig.tiny()
BLOCK = 256


def make_engine(graphs: bool, num_blocks: int = 64, max_num_seqs: int = 64,
                max_model_len: int = 512, async_scheduling: bool = False,
                chunked: bool = False, max_batched: int = 4096, piecewise: bool = False,
                spec_k: int = 0) -> LLMEngine:
    model = Qwen2ForCausalLM(CFG)
    reset_parameters_deterministic(model, 0)
    model = model.to("cuda", torch.float16)
    ecfg = EngineConfig(device="cuda", dtype=torch.float16, block_size=BLOCK,
                        num_gpu_blocks=num_blocks, max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=max_batched, max_model_len=max_model_len,
                        attn_backend="paged_flash", enable_cuda_graphs=graphs,
                        async_scheduling=async_scheduling, enable_chunked_prefill=chunked,
                        piecewise_cuda_graphs=piecewise,
                        speculative_ngram=3 if spec_k else 0, num_speculative_tokens=spec_k)
    return LLMEngine(model, CFG, ecfg, tokenizer=None)


def prompts(n: int, seed: int = 1) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, CFG.vocab_size, (int(torch.randint(3, 40, (1,), generator=g)),),
                          generator=g).tolist() for _ in range(n)]


def gen(eng: LLMEngine, ps, max_tokens=16):
    return [r.output_token_ids for r in
            LLM.from_engine(eng).generate(ps, SamplingParams.greedy(max_tokens, ignore_eos=True))]


def test_graphs_match_eager_8_prompts():
    ps = prompts(8)
    eager = gen(make_engine(False), ps)
    graphed = make_engine(True)
    assert graphed.graph_runner is not None and graphed.scratch_block == 63
    assert gen(graphed, ps) == eager
    assert graphed.block_manager.num_blocks == 63


@pytest.mark.parametrize("n", [5, 13])
def test_batch_between_buckets(n):
    ps = prompts(n, seed=n)
    assert gen(make_engine(True), ps) == gen(make_engine(False), ps)


def test_scratch_block_never_in_block_tables():
    eng = make_engine(True)
    scratch = eng.scratch_block
    ps = prompts(13, seed=4)
    sp = SamplingParams.greedy(12, ignore_eos=True)
    for i, p in enumerate(ps):
        eng.add_request(str(i), p, sp)
    while eng.has_unfinished_requests():
        eng.step()
        for sid in range(len(ps)):
            if eng.block_manager.has_sequence(sid):
                assert scratch not in eng.block_manager.get_block_table(sid)
    # Padding rows of the static tables only reference the scratch block or block 0.
    assert eng.graph_runner.block_tables[:, 0].max().item() <= scratch



@pytest.mark.parametrize("graphs", [False, True])
def test_async_scheduling_matches_sync_on_cuda(graphs):
    """The device-side token gather + pinned read-back path: same greedy tokens as the
    synchronous engine, with and without graphs, plain and chunked, and under
    EOS-style stop tokens (the discarded extra token never surfaces)."""
    ps = prompts(13, seed=3)
    ref = gen(make_engine(graphs), ps)
    assert gen(make_engine(graphs, async_scheduling=True), ps) == ref
    assert gen(make_engine(graphs, async_scheduling=True, chunked=True, max_batched=24), ps) == ref
    sps = [SamplingParams.greedy(16, stop_token_ids=[r[5]]) for r in ref]
    eng = make_engine(graphs, async_scheduling=True)
    got = [r.output_token_ids for r in LLM.from_engine(eng).generate(ps, sps)]
    for r, sp, g in zip(ref, sps, got):
        assert g == r[: r.index(sp.stop_token_ids[0]) + 1]
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks


@pytest.mark.parametrize("chunked", [False, True])
def test_piecewise_graphs_match_eager(chunked):
    """Prefill and mixed steps replayed piecewise (per-layer graphs, eager attention) give
    the same greedy tokens as the eager engine; decode steps still use the full graph."""
    ps = prompts(11, seed=6)
    ref = gen(make_engine(False, chunked=chunked, max_batched=40 if chunked else 4096), ps)
    eng = make_engine(True, chunked=chunked, max_batched=40 if chunked else 4096, piecewise=True,
                      async_scheduling=True)
    assert eng.piecewise_runner is not None and eng.piecewise_runner.buckets[-1] == max(64, 40 if chunked else 4096)
    assert gen(eng, ps) == ref
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks


@pytest.mark.parametrize("piecewise", [False, True])
def test_speculative_decoding_matches_greedy_on_cuda(piecewise):
    """Draft rows go through flash's mixed-step path (decode rows batched, draft rows padded)
    and, with piecewise graphs, through per-layer replays; tokens equal plain greedy."""
    ps = prompts(9, seed=8) + [prompts(1, seed=9)[0] * 3]
    ref = gen(make_engine(False), ps, max_tokens=20)
    eng = make_engine(True, piecewise=piecewise, chunked=piecewise, max_batched=64 if piecewise else 4096,
                      spec_k=4)
    assert gen(eng, ps, max_tokens=20) == ref
    assert eng.spec_drafted > 0
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks
