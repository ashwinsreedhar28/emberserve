"""Tensor parallelism (dist.py) on the CPU: the checkpoint sharding, and a real two-process
gloo group whose output must equal the single-process engine's, through the plain, async,
chunked and speculative step paths and through the engine-core process."""

from __future__ import annotations

import asyncio

import pytest
import torch

from pagedserve import dist as tpdist
from pagedserve.config import EngineConfig, ModelConfig
from pagedserve.dist import TPState, WorkerSpec, shard_tensor, vocab_shard
from pagedserve.engine import LLMEngine
from pagedserve.llm import LLM
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic
from pagedserve.model.weights import hf_state_dict, load_hf_state_dict
from pagedserve.sched.request import SamplingParams
from tests.test_engine import prompts

torch.set_num_threads(2)
CFG = ModelConfig.tiny()


def ecfg(tp: int = 1, **kw) -> EngineConfig:
    base = dict(device="cpu", dtype=torch.float32, block_size=4, num_gpu_blocks=256,
                max_num_seqs=16, max_num_batched_tokens=512, max_model_len=256,
                attn_backend="paged_torch", tensor_parallel_size=tp)
    base.update(kw)
    return EngineConfig(**base)


def full_model(seed: int = 0) -> Qwen2ForCausalLM:
    m = Qwen2ForCausalLM(CFG, tp_size=1)
    reset_parameters_deterministic(m, seed)
    return m.eval()


# ---- sharding, no process group ---------------------------------------------------------------
def test_config_shard():
    s = CFG.shard(2)
    assert (s.num_attention_heads, s.num_key_value_heads, s.intermediate_size) == (2, 1, 64)
    assert s.head_dim == CFG.head_dim == 16 and s.hidden_size == 64 and s.vocab_size == 256
    assert CFG.shard(1) is CFG
    with pytest.raises(ValueError):
        ModelConfig.tiny(num_key_value_heads=1).shard(2)
    assert vocab_shard(256, 2) == 128 and vocab_shard(255, 2) is None and vocab_shard(256, 1) is None


def test_shard_tensor_rules():
    w = torch.arange(24.0).view(4, 6)
    assert torch.equal(shard_tensor("model.layers.3.self_attn.q_proj.weight", w, 1, 2), w[2:])
    assert torch.equal(shard_tensor("model.layers.3.self_attn.k_proj.bias", w, 0, 2), w[:2])
    assert torch.equal(shard_tensor("model.layers.3.mlp.up_proj.weight", w, 1, 2), w[2:])
    assert torch.equal(shard_tensor("lm_head.weight", w, 0, 2), w[:2])
    assert torch.equal(shard_tensor("model.layers.3.self_attn.o_proj.weight", w, 1, 2), w[:, 3:])
    assert torch.equal(shard_tensor("model.layers.3.mlp.down_proj.weight", w, 0, 2), w[:, :3])
    assert shard_tensor("model.embed_tokens.weight", w, 1, 2) is w
    assert shard_tensor("model.layers.3.input_layernorm.weight", w, 1, 2) is w
    assert shard_tensor("lm_head.weight", w, 0, 1) is w


def test_sharded_loader_round_trip():
    """Each rank's shard, loaded from the full model's HF-layout state dict, holds exactly
    the rows/columns Megatron-style sharding assigns it; the tied lm_head becomes a real
    parameter holding its slice of the embedding."""
    full = full_model()
    sd = hf_state_dict(full)
    q, kv, inter = 4 * 16, 2 * 16, 128
    for rank in (0, 1):
        shard = Qwen2ForCausalLM(CFG.shard(2), tp_size=2)
        load_hf_state_dict(shard, sd, tp=TPState(rank, 2))
        r = slice(rank * q // 2, (rank + 1) * q // 2)
        rk = slice(rank * kv // 2, (rank + 1) * kv // 2)
        ri = slice(rank * inter // 2, (rank + 1) * inter // 2)
        f, s = full.model.layers[1].self_attn, shard.model.layers[1].self_attn
        fq = f.qkv_proj.weight
        want = torch.cat([fq[:q][r], fq[q:q + kv][rk], fq[q + kv:][rk]])
        assert torch.equal(s.qkv_proj.weight, want)
        fb = f.qkv_proj.bias
        assert torch.equal(s.qkv_proj.bias, torch.cat([fb[:q][r], fb[q:q + kv][rk], fb[q + kv:][rk]]))
        assert torch.equal(s.o_proj.weight, f.o_proj.weight[:, r])
        fm, sm = full.model.layers[1].mlp, shard.model.layers[1].mlp
        assert torch.equal(sm.gate_up_proj.weight,
                           torch.cat([fm.gate_up_proj.weight[:inter][ri], fm.gate_up_proj.weight[inter:][ri]]))
        assert torch.equal(sm.down_proj.weight, fm.down_proj.weight[:, ri])
        assert torch.equal(shard.model.embed_tokens.weight, full.model.embed_tokens.weight)
        assert shard.lm_head_sharded and shard.lm_head.weight.shape == (128, 64)
        assert torch.equal(shard.lm_head.weight, full.model.embed_tokens.weight[rank * 128:(rank + 1) * 128])
        assert shard.lm_head.weight is not shard.model.embed_tokens.weight
        assert torch.equal(shard.model.norm.weight, full.model.norm.weight)


def test_engine_refuses_a_mismatched_group():
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        LLMEngine(full_model(), CFG, ecfg(tp=2), tokenizer=None)


# ---- two processes over gloo -------------------------------------------------------------------
def same(a, b):
    assert [r.output_token_ids for r in a] == [r.output_token_ids for r in b]
    assert [r.finish_reason for r in a] == [r.finish_reason for r in b]


@pytest.mark.parametrize("mode", ["plain", "async", "chunked", "spec"])
def test_tp2_matches_single_process(mode):
    kw = {"plain": {}, "async": {"async_scheduling": True},
          "chunked": {"enable_chunked_prefill": True, "max_num_batched_tokens": 24,
                      "async_scheduling": True},
          "spec": {"speculative_ngram": 3, "num_speculative_tokens": 4}}[mode]
    ps = prompts(5, seed=3) + [prompts(1, seed=9)[0] * 3]
    sps = [SamplingParams.greedy(16, ignore_eos=True)] * 4 + [
        SamplingParams(max_tokens=16, temperature=0.7, top_p=0.9, seed=7, ignore_eos=True)] * 2
    ref = LLM.from_engine(LLMEngine(full_model(), CFG, ecfg(1, **kw), tokenizer=None)).generate(ps, sps)
    eng = LLMEngine.launch_tp(WorkerSpec(ecfg(2, **kw), tiny=True, tiny_seed=0))
    try:
        assert eng.tp.size == 2 and eng.tp.is_driver and len(eng._tp_workers) == 1
        assert eng.local_config.num_attention_heads == 2 and eng.model.lm_head_sharded
        assert eng.backend.cache.num_kv_heads == 1  # one KV head per rank
        got = eng and LLM.from_engine(eng).generate(ps, sps)
        same(got, ref)
        if mode == "spec":
            assert eng.spec_drafted > 0
        # a second batch on the same engine (the workers keep following)
        same(LLM.from_engine(eng).generate(ps[:2], sps[:2]), ref[:2])
        assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks
    finally:
        procs = list(eng._tp_workers)
        eng.shutdown()
    assert not tpdist.is_initialized() and eng.tp.size == 1
    assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]


def test_driver_failure_mid_step_does_not_hang_shutdown():
    """A step that raises on the driver after the plan went out leaves the worker waiting
    on collectives; `shutdown()` must kill it and return instead of blocking forever in a
    "stop" broadcast (which would also hide the original exception)."""
    import time

    eng = LLMEngine.launch_tp(WorkerSpec(ecfg(2), tiny=True, tiny_seed=0))
    procs = list(eng._tp_workers)
    orig = eng._forward

    def boom(input_ids, meta, so):
        raise RuntimeError("kernel failed")

    eng._forward = boom
    try:
        with pytest.raises(RuntimeError, match="kernel failed"):
            LLM.from_engine(eng).generate(prompts(2, seed=3), SamplingParams.greedy(4, ignore_eos=True))
        assert eng._tp_step_open
    finally:
        eng._forward = orig
        t0 = time.monotonic()
        eng.shutdown()
        took = time.monotonic() - t0
    assert took < 20.0, took
    assert not tpdist.is_initialized() and all(p.returncode is not None for p in procs)


def test_tp2_through_the_engine_core_process():
    """The core process (itself daemonic) starts the worker as a subprocess and stops it."""
    from pagedserve.server.async_engine import AsyncEngineCoreClient
    from pagedserve.server.engine_core import EngineSpec
    from tests.stub_tokenizer import StubTokenizer

    tiny = dict(num_hidden_layers=CFG.num_hidden_layers, num_attention_heads=CFG.num_attention_heads,
                num_key_value_heads=CFG.num_key_value_heads, hidden_size=CFG.hidden_size,
                intermediate_size=CFG.intermediate_size, vocab_size=CFG.vocab_size)
    ps = prompts(4, seed=5)
    sp = SamplingParams.greedy(10, ignore_eos=True)
    ref = LLM.from_engine(LLMEngine(full_model(), CFG, ecfg(1), tokenizer=None)).generate(ps, sp)

    async def run():
        c = AsyncEngineCoreClient(EngineSpec(ecfg(2), tiny=True, tiny_seed=0, tiny_overrides=tiny),
                                  StubTokenizer())
        c.start()
        try:
            outs = await asyncio.gather(*(_collect(c, f"r{i}", p, sp) for i, p in enumerate(ps)))
        finally:
            c.stop()
        return outs

    outs = asyncio.run(run())
    for r, chunks in zip(ref, outs):
        assert chunks[-1].output_token_ids == r.output_token_ids


async def _collect(c, rid, prompt, sp):
    return [out async for out in c.generate(rid, prompt, sp)]
