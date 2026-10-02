"""The Runpod Serverless worker's job contract: the proxy handler against the real app on
the tiny CPU engine (httpx ASGI transport, no SDK, no port)."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from emberserve.server.app import create_app
from emberserve.server.async_engine import AsyncLLMEngine
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
    assert cmd[1:4] == ["-m", "emberserve.cli", "serve"] and "--enable-cuda-graphs" in cmd
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


async def test_stream_chunks_are_coalesced_by_time() -> None:
    import asyncio

    mod = _load("handler")

    async def source():
        for i in range(6):
            yield f"data: {i}\n\n"
            await asyncio.sleep(0.03)

    out = [c async for c in mod._coalesced(source(), 0.1)]
    assert "".join(out) == "".join(f"data: {i}\n\n" for i in range(6))
    assert 1 < len(out) < 6  # ~0.1 s windows over 0.18 s of chunks: fewer yields than chunks
    assert [c async for c in mod._coalesced(source(), 0)] == [f"data: {i}\n\n" for i in range(6)]


async def test_stream_flush_does_not_wait_for_the_next_chunk() -> None:
    """The first chunk goes out at once, and a buffered chunk goes out when its window ends
    even if the next chunk is late (it used to wait for that chunk: a 100 ms window held a
    token ~350 ms in a reviewer's repro)."""
    import asyncio

    mod = _load("handler")
    loop = asyncio.get_running_loop()
    t0 = loop.time()

    async def source():
        yield "a"
        await asyncio.sleep(0.02)
        yield "b"
        await asyncio.sleep(0.5)  # a slow step
        yield "c"

    seen = []
    async for c in mod._coalesced(source(), 0.1):
        seen.append((c, loop.time() - t0))
    assert [c for c, _ in seen] == ["a", "b", "c"]
    assert seen[0][1] < 0.05  # first token: immediately
    assert seen[1][1] < 0.25  # "b": when its 0.1 s window ends, not when "c" arrives (~0.52 s)


async def test_stream_coalescer_passes_source_errors_and_stops_early() -> None:
    import asyncio

    mod = _load("handler")

    async def failing():
        yield "a"
        await asyncio.sleep(0.01)
        raise RuntimeError("upstream broke")

    got = []
    with pytest.raises(RuntimeError, match="upstream broke"):
        async for c in mod._coalesced(failing(), 0.1):
            got.append(c)
    assert got == ["a"]

    async def endless():
        while True:
            yield "x"
            await asyncio.sleep(0.001)

    gen = mod._coalesced(endless(), 0.05)
    assert await gen.__anext__() == "x"
    await gen.aclose()  # the pump task is cancelled, nothing left running


async def test_ping_and_lb_command(proxied, monkeypatch) -> None:
    client, _ = proxied
    assert (await client.get("/ping")).status_code == 200
    main = _load("main")
    monkeypatch.setenv("PORT", "9000")
    cmd = main.serve_command("/models/m", host="0.0.0.0", port=9000)
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0" and cmd[cmd.index("--port") + 1] == "9000"


async def test_timeline_job_returns_worker_marks(proxied) -> None:
    client, _ = proxied
    h = _load("handler")
    tl = h.timeline
    tl.reset()
    tl.record_process_starts()
    tl.mark("serve_healthy")
    handler = h.make_handler(client, served_model=MODEL)
    out = await _collect(handler, {"id": "t1", "input": {"prompt": "hi", "timeline": True,
                                                         "sampling_params": {"max_tokens": 2, "ignore_eos": True}}})
    assert len(out) == 1 and set(out[0]) == {"timeline", "output"}
    marks = out[0]["timeline"]["marks"]
    assert {"serve_healthy", "first_job"} <= set(marks)
    assert out[0]["output"]["usage"]["completion_tokens"] == 2
    plain = await _collect(handler, {"id": "t2", "input": {"prompt": "hi", "sampling_params": {"max_tokens": 2}}})
    assert "timeline" not in plain[0]
    tl.reset()


def test_proc_start_wall_is_in_the_past() -> None:
    import os
    import time

    tl = _load("timeline")
    t = tl.proc_start_wall("self")
    if not os.path.exists("/proc/self/stat"):
        assert t is None
        return
    assert t is not None and time.time() - 3600 * 24 * 365 < t <= time.time() + 1
    assert tl.proc_start_wall(1) is not None and tl.proc_start_wall(1) <= t + 1


def test_coldstart_phases_from_marks() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import serverless_coldstart as sc

    tl = {"marks": {"container_start": 100.0, "worker_start": 101.0, "worker_main": 102.5,
                    "serve_spawned": 102.6, "engine_boot": 107.0, "serve_healthy": 107.8,
                    "sdk_ready": 107.9, "first_job": 108.2}, "notes": {}}
    ph = sc.phases(90.0, tl)
    assert ph["schedule_pull_create"] == 10.0 and ph["engine_boot"] == 4.4
    assert ph["submit_to_first_job"] == 18.2
    assert sc._timeline_of([{"timeline": tl, "output": {}}]) == tl
    assert sc.phases(90.0, None) is None


def test_fetcher_streams_a_checkpoint_into_the_loader(tmp_path, monkeypatch) -> None:
    """Small files first, shards in the background with a delay each, and the engine's
    loader (waiting via EMBERSERVE_WAIT_WEIGHTS_S) ends up with the same weights."""
    import shutil
    import sys
    import time

    import torch

    from emberserve.model.weights import load_model
    from tests.test_model import tiny_model
    from tests.test_weights import _dump_snapshot

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "runpod"))
    fetch = _load("fetch")
    repo = tmp_path / "repo"
    repo.mkdir()
    _dump_snapshot(tiny_model(seed=8), repo, split=True)
    import json

    from safetensors import safe_open

    wm = {}
    for sh in sorted(repo.glob("*.safetensors")):
        with safe_open(str(sh), framework="pt") as f:
            wm.update({k: sh.name for k in f.keys()})
    (repo / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    monkeypatch.setenv("EMBERSERVE_LOADER", "safetensors")
    ref = load_model(repo, dtype=torch.float32)
    monkeypatch.delenv("EMBERSERVE_LOADER")

    def download(r, name, local_dir, revision):
        if name.endswith(".safetensors"):
            time.sleep(0.2)
        out = Path(local_dir) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(r) / name, out)
        return str(out)

    dst = tmp_path / "model"
    f = fetch.Fetcher(str(repo), dst, workers=2, download=download,
                      list_files=lambda r, rev: sorted(p.name for p in Path(r).iterdir()))
    shards = f.fetch_small()
    assert (dst / "config.json").exists() and not list(dst.glob("*.safetensors"))
    assert shards == sorted(p.name for p in repo.glob("*.safetensors"))
    f.start_shards(shards)
    monkeypatch.setenv("EMBERSERVE_WAIT_WEIGHTS_S", "10")
    m = load_model(dst, dtype=torch.float32)
    assert f.done.wait(10) and f.error is None
    assert m.load_stats.wait_seconds > 0.1
    for k, v in ref.state_dict().items():
        assert torch.equal(v, m.state_dict()[k]), k
    assert not (dst / ".incoming").exists()
    assert "weights_downloaded" in fetch.timeline.snapshot()["marks"]


def test_fetcher_reports_a_failed_download(tmp_path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "runpod"))
    fetch = _load("fetch")

    def download(r, name, local_dir, revision):
        if name.endswith(".safetensors"):
            raise OSError("HTTP 503")
        out = Path(local_dir) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return str(out)

    f = fetch.Fetcher("r", tmp_path / "m", download=download,
                      list_files=lambda r, rev: ["config.json", "model-1.safetensors"])
    f.start_shards(f.fetch_small())
    assert f.done.wait(5) and isinstance(f.error, OSError)


def test_snapshot_complete_needs_every_indexed_shard(tmp_path) -> None:
    """config.json is fetched first, so a directory left by an interrupted download has it
    and no shards; it used to count as a finished snapshot and skip the fetch."""
    main = _load("main")
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text("{}")
    assert not main.snapshot_complete(d)
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    (d / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert not main.snapshot_complete(d)
    (d / "model-00002-of-00002.safetensors").write_bytes(b"x")
    assert main.snapshot_complete(d)
    single = tmp_path / "s"
    single.mkdir()
    (single / "config.json").write_text("{}")
    (single / "model.safetensors").write_bytes(b"x")
    assert main.snapshot_complete(single) and not main.snapshot_complete(tmp_path / "nothing")


@pytest.mark.parametrize("env", [{"TENSOR_PARALLEL_SIZE": "2"}, {"EMBERSERVE_LOADER": "safetensors"},
                                 {"EXTRA_SERVE_ARGS": "--no-chunked-prefill --tensor-parallel-size 2"},
                                 {"EXTRA_SERVE_ARGS": "--tensor-parallel-size=2"},
                                 {"EXTRA_SERVE_ARGS": "--tensor-parallel 2"},
                                 {"EXTRA_SERVE_ARGS": "--tensor-parallel=2"}])
def test_weights_download_first_when_the_loader_cannot_wait(tmp_path, monkeypatch, env) -> None:
    """Only the streaming loader on one rank waits for shards still downloading; with tensor
    parallelism or the reference loader the engine read an empty directory and failed."""
    import huggingface_hub

    main = _load("main")
    calls = []

    def fake_snapshot_download(repo, local_dir, revision=None, allow_patterns=None):
        calls.append(repo)
        Path(local_dir, "config.json").write_text("{}")
        Path(local_dir, "model.safetensors").write_bytes(b"x")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    for k in ("TENSOR_PARALLEL_SIZE", "EMBERSERVE_LOADER", "WEIGHTS_STREAM", "EMBERSERVE_WAIT_WEIGHTS_S",
              "EXTRA_SERVE_ARGS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MODEL_REPO", "org/model")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
    (tmp_path / "m").mkdir()
    assert main.resolve_model_dir() == str(tmp_path / "m")
    assert calls == ["org/model"] and main.snapshot_complete(tmp_path / "m")
    import os

    assert "EMBERSERVE_WAIT_WEIGHTS_S" not in os.environ
