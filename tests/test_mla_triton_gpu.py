"""Triton MLA decode kernel and the mla_triton backend on CUDA: kernel vs the absorbed
torch reference at Moonlight's geometry, flash-varlen prefill vs the reference, and the
tiny DeepSeek model generating identical tokens through mla_torch and mla_triton."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from pagedserve.attn.mla_torch import mla_attention_absorbed  # noqa: E402
from pagedserve.attn.mla_triton import mla_decode  # noqa: E402
from pagedserve.config import EngineConfig  # noqa: E402
from pagedserve.engine import LLMEngine  # noqa: E402
from pagedserve.llm import LLM  # noqa: E402
from pagedserve.sched.request import SamplingParams  # noqa: E402
from tests.test_deepseek import tiny_model  # noqa: E402

DEV = "cuda"
H, DL, DR = 16, 512, 64  # Moonlight / DeepSeek-V2-Lite latent geometry
BLOCK = 16


def _setup(seed, batch, ctx_lens, dtype):
    g = torch.Generator().manual_seed(seed)
    max_blocks = max((c + BLOCK - 1) // BLOCK for c in ctx_lens)
    num_blocks = batch * max_blocks + 5
    latent = (torch.randn(num_blocks, BLOCK, DL + DR, generator=g) * 0.5).to(DEV, dtype)
    perm = torch.randperm(num_blocks, generator=g).tolist()
    tables = torch.zeros(batch, max_blocks, dtype=torch.int32)
    for b in range(batch):
        for j in range((ctx_lens[b] + BLOCK - 1) // BLOCK):
            tables[b, j] = perm.pop()
    q_abs = (torch.randn(batch, H, DL + DR, generator=g) * 0.5).to(DEV, dtype)
    return latent, tables.to(DEV), torch.tensor(ctx_lens, dtype=torch.int32, device=DEV), q_abs


def _reference(latent, tables, ctx_lens, q_abs, scale):
    eye = torch.eye(DL, device=DEV).expand(H, DL, DL)
    outs = []
    for b in range(q_abs.shape[0]):
        n = int(ctx_lens[b])
        rows = latent[tables[b, :(n + BLOCK - 1) // BLOCK].long()].reshape(-1, DL + DR)[:n]
        out_c = mla_attention_absorbed(q_abs[b, :, :DL][None].float(), q_abs[b, :, DL:][None].float(),
                                       rows.float(), eye, eye, scale, DL)[0]
        outs.append(out_c)
    return torch.stack(outs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ctx_lens", [[1], [16, 17], [300, 5, 1024, 77], [2048] * 8])
def test_kernel_matches_reference(dtype, ctx_lens):
    latent, tables, ctx, q_abs = _setup(1, len(ctx_lens), ctx_lens, dtype)
    scale = (128 + 64) ** -0.5
    for splits in (1, None, 4):
        got = mla_decode(q_abs, latent, tables, ctx, scale, DL, num_splits=splits).float()
        want = _reference(latent, tables, ctx, q_abs, scale)
        tol = dict(atol=2e-2, rtol=2e-2) if dtype == torch.float16 else dict(atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(got, want, **tol)


def _engine(model, backend: str, dtype, graphs: bool = False, chunk: int | None = None,
            piecewise: bool = False) -> LLMEngine:
    ecfg = EngineConfig(device=DEV, dtype=dtype, block_size=BLOCK, num_gpu_blocks=256,
                        max_num_seqs=16, max_num_batched_tokens=chunk or 512, max_model_len=256,
                        attn_backend=backend, enable_cuda_graphs=graphs,
                        enable_chunked_prefill=chunk is not None, piecewise_cuda_graphs=piecewise)
    return LLMEngine(model, model.config, ecfg, tokenizer=None)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tiny_model_same_tokens_through_both_backends(dtype):
    """Prefill (flash varlen, non-absorbed) + decode (Triton kernel) vs the torch absorbed
    reference for both phases: greedy tokens must agree on several prompts at once."""
    model = tiny_model(seed=21).to(DEV, dtype)
    prompts = [torch.randint(2, 256, (n,), generator=torch.Generator().manual_seed(n)).tolist()
               for n in (4, 17, 33, 9)]
    sp = SamplingParams.greedy(12, ignore_eos=True)
    ref = LLM.from_engine(_engine(model, "mla_torch", dtype)).generate(prompts, sp)
    got = LLM.from_engine(_engine(model, "mla_triton", dtype)).generate(prompts, sp)
    for r, g in zip(ref, got):
        assert r.output_token_ids == g.output_token_ids
    # and with CUDA graphs capturing the decode step (Triton MLA kernel + fused MoE inside)
    graphed = LLM.from_engine(_engine(model, "mla_triton", dtype, graphs=True)).generate(prompts, sp)
    for r, g in zip(ref, graphed):
        assert r.output_token_ids == g.output_token_ids


def test_prefill_paths_agree(dtype=torch.float16):
    """flash-varlen fresh-prompt prefill vs the absorbed torch reference on the tiny model:
    prompt logits within fp16 tolerance."""
    model = tiny_model(seed=22).to(DEV, dtype)
    ids = torch.randint(2, 256, (23,), generator=torch.Generator().manual_seed(3)).tolist()
    logits = []
    for backend in ("mla_torch", "mla_triton"):
        e = _engine(model, backend, dtype)
        e.add_request("r", ids, SamplingParams.greedy(1))
        so = e.scheduler.schedule()
        input_ids, meta = e._build_inputs(so)
        with torch.inference_mode():
            logits.append(model.forward_logits_all(input_ids, e.backend, meta).float().cpu())
    torch.testing.assert_close(logits[1], logits[0], atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_prefill_mixed_steps(dtype):
    """Chunked prefill with a 12-token cap: prompts of 4..33 tokens are split across steps
    and every later chunk attends through the cache beside decode rows (`_prefill_mixed`:
    absorbed kernel for the decode rows, gathered latent -> per-head k/v -> flash varlen
    for the chunk). Tokens must match mla_torch under the same schedule."""
    model = tiny_model(seed=23).to(DEV, dtype)
    prompts = [torch.randint(2, 256, (n,), generator=torch.Generator().manual_seed(n + 7)).tolist()
               for n in (4, 17, 33, 9, 25)]
    sp = SamplingParams.greedy(10, ignore_eos=True)
    ref = LLM.from_engine(_engine(model, "mla_torch", dtype, chunk=12)).generate(prompts, sp)
    got = LLM.from_engine(_engine(model, "mla_triton", dtype, chunk=12)).generate(prompts, sp)
    for r, g in zip(ref, got):
        assert r.output_token_ids == g.output_token_ids


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_fold_matches_unfolded_on_cuda(dtype):
    """With the rope permutation folded into the weights the fused RoPE kernel runs in
    place on strided views of q and of kv_a's output; tokens must not change."""
    model = tiny_model(seed=24).to(DEV, dtype)
    prompts = [torch.randint(2, 256, (n,), generator=torch.Generator().manual_seed(n + 3)).tolist()
               for n in (5, 19, 30)]
    sp = SamplingParams.greedy(12, ignore_eos=True)
    ref = LLM.from_engine(_engine(model, "mla_triton", dtype)).generate(prompts, sp)
    model.fold_rope_permutation()
    got = LLM.from_engine(_engine(model, "mla_triton", dtype, graphs=True)).generate(prompts, sp)
    for r, g in zip(ref, got):
        assert r.output_token_ids == g.output_token_ids


def test_piecewise_graphs_on_the_mla_moe_model():
    """Per-layer graphs around the eager MLA attention, MoE layers included, with chunked
    prefill: same tokens as mla_torch."""
    dtype = torch.bfloat16
    model = tiny_model(seed=25).to(DEV, dtype)
    prompts = [torch.randint(2, 256, (n,), generator=torch.Generator().manual_seed(n + 11)).tolist()
               for n in (6, 21, 33, 14)]
    sp = SamplingParams.greedy(10, ignore_eos=True)
    ref = LLM.from_engine(_engine(model, "mla_torch", dtype, chunk=16)).generate(prompts, sp)
    eng = _engine(model, "mla_triton", dtype, graphs=True, chunk=16, piecewise=True)
    assert eng.piecewise_runner is not None
    got = LLM.from_engine(eng).generate(prompts, sp)
    for r, g in zip(ref, got):
        assert r.output_token_ids == g.output_token_ids
