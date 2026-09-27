"""Piecewise graph runner: the layer decomposition and the buffer orchestration, on the CPU.

`PiecewiseGraphRunner` off CUDA captures nothing (each piece "replays" by running eagerly),
which is exactly the orchestration a graph replay performs on the device: padded static
buffers, per-layer pre -> attend -> post, copies into and out of the buffers. So the CPU
runner must reproduce the eager forward's logits exactly, for prefill steps, mixed steps,
and decode steps, on both the dense and the latent-attention model."""

from __future__ import annotations

import pytest
import torch

from pagedserve.attn.piecewise_graphs import PiecewiseGraphRunner
from pagedserve.config import EngineConfig, ModelConfig
from pagedserve.engine import LLMEngine
from pagedserve.llm import LLM
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from pagedserve.sched.request import SamplingParams
from tests.test_deepseek import tiny_model as tiny_deepseek

torch.set_num_threads(2)


def dense_model():
    cfg = ModelConfig.tiny()
    m = Qwen2ForCausalLM(cfg)
    reset_parameters_deterministic(m, 3)
    return m.eval()


def engine(model, backend="paged_torch", chunked=False, budget=512):
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=4, num_gpu_blocks=256,
                        max_num_seqs=16, max_num_batched_tokens=budget, max_model_len=256,
                        attn_backend=backend, enable_chunked_prefill=chunked)
    return LLMEngine(model, model.config, ecfg, tokenizer=None)


def prompts(n, seed=1, lo=3, hi=30):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, 200, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g).tolist()
            for _ in range(n)]


@pytest.mark.parametrize("family", ["dense", "mla"])
def test_pieces_compose_to_forward(family):
    """pre -> attend -> post equals the layer's forward, for a prefill step and a decode step."""
    model = dense_model() if family == "dense" else tiny_deepseek(seed=4)
    eng = engine(model, "paged_torch" if family == "dense" else "mla_torch")
    ps = prompts(3, seed=2)
    sp = SamplingParams.greedy(2, ignore_eos=True)
    for i, p in enumerate(ps):
        eng.add_request(str(i), p, sp)
    for _ in range(2):  # a prefill step, then a decode step
        so = eng.scheduler.schedule()
        input_ids, meta = eng._build_inputs(so)
        inner = model.model
        with torch.inference_mode():
            ref = model(input_ids, eng.backend, meta)
            # the pieces write the same K/V into the same slots again: idempotent
            hidden = inner.embed_tokens(input_ids)
            residual = None
            for layer in inner.layers:
                pre, residual = layer.pre(hidden, residual, meta.positions)
                hidden, residual = layer.post(layer.attend(pre, eng.backend, meta), residual)
            got, _ = inner.norm.forward_with_residual(hidden, residual)
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
        eng._advance(so)
        for r in so.scheduled:
            r.append_output(7)


@pytest.mark.parametrize("family", ["dense", "mla"])
@pytest.mark.parametrize("chunked", [False, True])
def test_runner_matches_eager_forward(family, chunked):
    """The runner (buckets, padded static buffers, copies) reproduces the eager logits step
    for step through prefill, mixed and decode steps, with rows padded up to a bucket."""
    model = dense_model() if family == "dense" else tiny_deepseek(seed=5)
    backend = "paged_torch" if family == "dense" else "mla_torch"
    eng = engine(model, backend, chunked=chunked, budget=24 if chunked else 512)
    runner = PiecewiseGraphRunner(model, eng.backend, max_tokens=64, buckets=(8, 16, 32, 64),
                                  device="cpu")
    runner.capture()
    assert runner.buckets == [8, 16, 32, 64] and runner.bucket_for(9) == 16 and runner.bucket_for(65) is None
    ps = prompts(5, seed=3)
    sp = SamplingParams.greedy(3, ignore_eos=True)
    for i, p in enumerate(ps):
        eng.add_request(str(i), p, sp)
    steps = 0
    while eng.has_unfinished_requests() and steps < 12:
        so = eng.scheduler.schedule()
        if so.is_empty:
            break
        input_ids, meta = eng._build_inputs(so)
        with torch.inference_mode():
            ref = model.compute_logits(model(input_ids, eng.backend, meta), meta)
            got = runner.run(input_ids, meta)
        # padded rows change the GEMM's blocking, so fp32 summation order differs by an ulp
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
        # padded rows of the static buffers never leak into real rows: run again with junk
        # in the padding and the result must not change
        n = meta.num_tokens
        runner.hidden[n:].fill_(float("nan"))
        runner.residual[n:].fill_(float("nan"))
        with torch.inference_mode():
            again = runner.run(input_ids, meta)
        assert not torch.isnan(again).any()
        # (with MoE the junk rows change which rows share an expert GEMM: ulp-level noise)
        torch.testing.assert_close(again, got, atol=1e-5, rtol=1e-5)
        eng._advance(so)
        toks = torch.argmax(ref, dim=-1).tolist()
        j = 0
        for r, done in zip(so.scheduled, so.prefill_complete):
            if done:
                r.append_output(toks[j])
                j += 1
        steps += 1
    assert steps >= 4


def test_token_buckets():
    from pagedserve.attn.piecewise_graphs import DEFAULT_TOKEN_BUCKETS, token_buckets

    assert token_buckets(2048, 0) == DEFAULT_TOKEN_BUCKETS
    assert token_buckets(2048, 256) == (16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048)
    assert token_buckets(1000, 128) == (16, 32, 64, 128, 256, 384, 512, 640, 768, 896)
    runner = PiecewiseGraphRunner(dense_model(), engine(dense_model()).backend, max_tokens=1000,
                                  buckets=token_buckets(1000, 128), device="cpu")
    assert runner.buckets[-1] == 1000 and runner.bucket_for(897) == 1000 and runner.bucket_for(300) == 384


def test_engine_flag_is_ignored_off_cuda():
    model = dense_model()
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=4, num_gpu_blocks=64,
                        max_num_seqs=8, max_num_batched_tokens=64, max_model_len=128,
                        attn_backend="paged_torch", enable_cuda_graphs=True, piecewise_cuda_graphs=True)
    with pytest.warns(UserWarning):
        eng = LLMEngine(model, model.config, ecfg, tokenizer=None)
    assert eng.piecewise_runner is None
    out = LLM.from_engine(eng).generate(prompts(2), SamplingParams.greedy(4, ignore_eos=True))
    assert all(len(o.output_token_ids) == 4 for o in out)
