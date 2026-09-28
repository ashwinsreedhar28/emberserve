"""The Runpod Serverless worker's job contract: the proxy handler against the real app on
the tiny CPU engine (httpx ASGI transport, no SDK, no port)."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from pagedserve.server.app import create_app
from pagedserve.server.async_engine import AsyncLLMEngine
from tests.stub_tokenizer import install
from tests.test_engine import make_engine

pytestmark = pytest.mark.anyio
MODEL = "tiny"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "deploy" / "runpod" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"runpod_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
async def proxied() -> AsyncIterator[tuple[httpx.AsyncClient, AsyncLLMEngine]]:
    aeng = AsyncLLMEngine(install(make_engine(max_model_len=512, num_blocks=512)))
    app = create_app(aeng, MODEL, manage_lifespan=False)
    aeng.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://worker") as client:
            yield client, aeng
    finally:
        aeng.stop()


async def _collect(handler, job):
    return [c async for c in handler(job)]


def _sse_events(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[6:]))
    return out


def test_normalize_job_input() -> None:
    n = _load("handler").normalize_job_input
    assert n({"openai_route": "/v1/chat/completions", "openai_input": {"messages": []}}) == \
        ("/v1/chat/completions", "POST", {"messages": []})
    assert n({"openai_input": {"prompt": "x"}}) == ("/v1/chat/completions", "POST", {"prompt": "x"})
    assert n({"openai_route": "/v1/models"}) == ("/v1/models", "GET", None)
    assert n({"route": "/v1/completions", "body": {"prompt": "x"}}) == ("/v1/completions", "POST", {"prompt": "x"})
    assert n({"route": "/health"}) == ("/health", "GET", None)
    assert n({"prompt": "hi", "sampling_params": {"max_tokens": 3}}) == \
        ("/v1/completions", "POST", {"max_tokens": 3, "stream": False, "prompt": "hi"})
    assert n({"messages": [{"role": "user", "content": "hi"}], "stream": True}) == \
        ("/v1/chat/completions", "POST", {"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    with pytest.raises(ValueError):
        n({})


async def test_openai_passthrough_non_stream_and_stream(proxied) -> None:
    client, _ = proxied
    handler = _load("handler").make_handler(client, served_model=MODEL)
    body = {"prompt": "hello world", "max_tokens": 6, "temperature": 0, "ignore_eos": True}
    full = await _collect(handler, {"id": "j1", "input": {"openai_route": "/v1/completions",
                                                           "openai_input": body}})
    assert len(full) == 1 and full[0]["object"] == "text_completion"
    assert full[0]["usage"]["completion_tokens"] == 6 and full[0]["model"] == MODEL
    chunks = await _collect(handler, {"id": "j2", "input": {"openai_route": "/v1/completions",
                                                             "openai_input": {**body, "stream": True}}})
    assert all(isinstance(c, str) for c in chunks)
    events = _sse_events("".join(chunks))
    assert "".join(e["choices"][0]["text"] for e in events) == full[0]["choices"][0]["text"]
    assert "".join(chunks).rstrip().endswith("data: [DONE]")


async def test_shorthand_routes_and_errors(proxied) -> None:
    client, _ = proxied
    handler = _load("handler").make_handler(client, served_model=MODEL)
    out = await _collect(handler, {"id": "j3", "input": {"prompt": "hi", "sampling_params": {
        "max_tokens": 2, "ignore_eos": True}}})
    assert out[0]["usage"]["completion_tokens"] == 2
    models = await _collect(handler, {"id": "j4", "input": {"openai_route": "/v1/models"}})
    assert models[0]["data"][0]["id"] == MODEL
    bad = await _collect(handler, {"id": "j5", "input": {}})
    assert bad[0]["error"]["type"] == "worker_error"
    http = await _collect(handler, {"id": "j6", "input": {"route": "/v1/completions",
                                                          "body": {"model": MODEL, "prompt": ["a", "b"]}}})
    assert "HTTP 400" in http[0]["error"]["message"]
    dead = _load("handler").make_handler(client, served_model=MODEL, alive=lambda: False)
    assert "not running" in (await _collect(dead, {"id": "j7", "input": {"prompt": "x"}}))[0]["error"]["message"]


def test_serve_command_from_env(monkeypatch) -> None:
    main = _load("main")
    for k in ("DTYPE", "CUDA_GRAPHS", "PREFIX_CACHING", "QUANTIZATION", "TENSOR_PARALLEL_SIZE",
              "EXTRA_SERVE_ARGS", "SERVED_MODEL_NAME"):
        monkeypatch.delenv(k, raising=False)
    cmd = main.serve_command("/models/m")
    assert cmd[1:4] == ["-m", "pagedserve.cli", "serve"] and "--enable-cuda-graphs" in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "/models/m"
    monkeypatch.setenv("QUANTIZATION", "int8")
    monkeypatch.setenv("TENSOR_PARALLEL_SIZE", "2")
    monkeypatch.setenv("CUDA_GRAPHS", "0")
    monkeypatch.setenv("EXTRA_SERVE_ARGS", "--no-chunked-prefill --piecewise-bucket-step 256")
    cmd = main.serve_command("/models/m")
    assert "--enable-cuda-graphs" not in cmd
    assert cmd[cmd.index("--quantization") + 1] == "int8"
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    assert cmd[-3:] == ["--no-chunked-prefill", "--piecewise-bucket-step", "256"]
