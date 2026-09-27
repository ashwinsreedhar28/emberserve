"""CUDA-graph decode must reproduce eager decode exactly (GPU only).

Tiny deterministic model, fp16 on CUDA, `paged_flash` backend (block_size 256 because
of flash-attn's paged page-size constraint). Run: `python -m pytest -m gpu -q`.
"""

from __future__ import annotations

import pytest
import torch

from pagedserve.config import EngineConfig, ModelConfig
from pagedserve.engine import LLMEngine
from pagedserve.llm import LLM
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from pagedserve.sched.request import SamplingParams

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("flash_attn")

CFG = ModelConfig.tiny()
BLOCK = 256


def make_engine(graphs: bool, num_blocks: int = 64, max_num_seqs: int = 64,
                max_model_len: int = 512) -> LLMEngine:
    model = Qwen2ForCausalLM(CFG)
    reset_parameters_deterministic(model, 0)
    model = model.to("cuda", torch.float16)
    ecfg = EngineConfig(device="cuda", dtype=torch.float16, block_size=BLOCK,
                        num_gpu_blocks=num_blocks, max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=4096, max_model_len=max_model_len,
                        attn_backend="paged_flash", enable_cuda_graphs=graphs)
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

