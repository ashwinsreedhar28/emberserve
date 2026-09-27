"""The Runpod Serverless handler's job contract, on the tiny CPU engine (no SDK needed)."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from pagedserve.llm import LLM
from pagedserve.sched.request import SamplingParams
from pagedserve.server.async_engine import AsyncLLMEngine
from tests.stub_tokenizer import install
from tests.test_engine import make_engine, prompts

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _load_handler_module():
    path = Path(__file__).resolve().parents[1] / "deploy" / "runpod" / "handler.py"
    spec = importlib.util.spec_from_file_location("runpod_handler", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


async def _collect(handler, job):
    return [c async for c in handler(job)]


async def test_streams_text_and_final_usage() -> None:
    mod = _load_handler_module()
    eng = install(make_engine())
    ref = LLM.from_engine(install(make_engine())).generate(prompts(1), SamplingParams.greedy(6, ignore_eos=True))[0]
    aeng = AsyncLLMEngine(eng)
    aeng.start()
    try:
        handler = mod.make_handler(aeng, eng.tokenizer)
        job = {"id": "job-1", "input": {"prompt_token_ids": prompts(1)[0], "max_tokens": 6,
                                        "ignore_eos": True}}
        chunks, bad = await asyncio.gather(
            _collect(handler, job),
            _collect(handler, {"id": "job-2", "input": {"max_tokens": 3}}))
    finally:
        aeng.stop()
    assert "".join(c["text"] for c in chunks) == ref.text
    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["usage"] == {"prompt_tokens": len(prompts(1)[0]), "completion_tokens": 6,
                                   "total_tokens": len(prompts(1)[0]) + 6}
    assert all("usage" not in c for c in chunks[:-1])
    assert bad == [{"error": "input needs one of: prompt, messages, prompt_token_ids"}]


def test_sampling_params_mapping() -> None:
    mod = _load_handler_module()
    sp = mod.to_sampling_params({"max_tokens": 5, "temperature": 0.5, "top_p": 0.9, "top_k": 40,
                                 "seed": 7, "stop": "END", "stop_token_ids": [3], "ignore_eos": True})
    assert (sp.max_tokens, sp.temperature, sp.top_p, sp.top_k, sp.seed) == (5, 0.5, 0.9, 40, 7)
    assert sp.stop == ["END"] and sp.stop_token_ids == [3] and sp.ignore_eos
    cfg = mod.engine_config()
    assert cfg.device == "cuda" and cfg.attn_backend == "paged_flash" and cfg.enable_chunked_prefill
