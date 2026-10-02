"""Tensor parallelism on two GPUs (NCCL): the tiny model with CUDA graphs (full-step and
piecewise, async scheduling) must reproduce the single-GPU engine, and Qwen2.5-0.5B on two
GPUs must agree with one GPU on greedy text up to fp16 tie-breaks.

Needs two CUDA devices; skips otherwise. Run: `python -m pytest tests/test_tp_gpu.py -q`.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch

from emberserve.config import EngineConfig, ModelConfig
from emberserve.dist import WorkerSpec
from emberserve.engine import LLMEngine
from emberserve.llm import LLM
from emberserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from emberserve.sched.request import SamplingParams

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    pytest.skip("needs two CUDA devices", allow_module_level=True)
pytest.importorskip("flash_attn")

CFG = ModelConfig.tiny()
MODEL = Path("models/Qwen2.5-0.5B-Instruct")


def ecfg(tp: int, backend: str = "paged_flash", block: int = 256, graphs: bool = True,
         piecewise: bool = False, async_scheduling: bool = False, chunked: bool = False,
         num_blocks: int | None = 64, max_batched: int = 4096) -> EngineConfig:
    return EngineConfig(device="cuda", dtype=torch.float16, block_size=block,
                        num_gpu_blocks=num_blocks, max_num_seqs=64, max_num_batched_tokens=max_batched,
                        max_model_len=512, attn_backend=backend, enable_cuda_graphs=graphs,
                        piecewise_cuda_graphs=piecewise, async_scheduling=async_scheduling,
                        enable_chunked_prefill=chunked, tensor_parallel_size=tp)


def prompts(n: int, seed: int = 1) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, CFG.vocab_size, (int(torch.randint(3, 40, (1,), generator=g)),),
                          generator=g).tolist() for _ in range(n)]


def tiny_single(cfg: EngineConfig) -> LLMEngine:
    model = Qwen2ForCausalLM(CFG, tp_size=1)
    reset_parameters_deterministic(model, 0)
    return LLMEngine(model.to("cuda", torch.float16), CFG, cfg, tokenizer=None)


def gen(eng, ps, max_tokens=16):
    return [r.output_token_ids for r in
            LLM.from_engine(eng).generate(ps, SamplingParams.greedy(max_tokens, ignore_eos=True))]


@pytest.mark.parametrize("mode", ["graphs", "piecewise_async", "eager_chunked"])
def test_tiny_tp2_matches_single_gpu(mode):
    kw = {"graphs": dict(graphs=True),
          "piecewise_async": dict(graphs=True, piecewise=True, async_scheduling=True, chunked=True,
                                  max_batched=64),
          "eager_chunked": dict(graphs=False, chunked=True, max_batched=64)}[mode]
    ps = prompts(12)
    ref = gen(tiny_single(ecfg(1, **kw)), ps)
    eng = LLMEngine.launch_tp(WorkerSpec(ecfg(2, **kw), tiny=True, tiny_seed=0))
    try:
        assert eng.backend.cache.num_kv_heads == 1 and eng.model.lm_head_sharded
        # fp16 partial sums differ in order (o_proj / down_proj split + all-reduce): allow
        # tie-break drift, but the bulk of a random tiny model's greedy path must agree
        got = gen(eng, ps)
        agree = sum(a == b for x, y in zip(got, ref) for a, b in zip(x, y))
        assert agree >= 0.9 * sum(len(x) for x in ref), (agree, got[:2], ref[:2])
        assert gen(eng, ps[:3]) == got[:3]  # and it is deterministic step to step
    finally:
        eng.shutdown()


@pytest.mark.hf
@pytest.mark.parametrize("backend,block", [("paged_flash", 256), ("paged_triton", 16)])
def test_qwen_0p5b_tp2_greedy_text(backend, block):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} not downloaded")
    texts = ["The capital of France is", "def fibonacci(n):", "In 1969, humans first",
             "List three primary colors:"]
    sp = SamplingParams.greedy(32)
    single = LLM(MODEL, ecfg(1, backend, block, num_blocks=None))
    ref = single.generate(texts, sp)
    del single  # it reserved 90% of cuda:0; the TP driver needs that memory
    gc.collect()
    torch.cuda.empty_cache()
    llm = LLM(MODEL, ecfg(2, backend, block, num_blocks=None, async_scheduling=True))
    try:
        got = llm.generate(texts, sp)
    finally:
        llm.engine.shutdown()
    for r, g in zip(ref, got):
        # identical up to the first fp16 tie-break, and the prefix is most of the output
        n = next((i for i, (a, b) in enumerate(zip(r.output_token_ids, g.output_token_ids)) if a != b),
                 len(r.output_token_ids))
        assert n >= min(12, len(r.output_token_ids)), (r.text, g.text)
