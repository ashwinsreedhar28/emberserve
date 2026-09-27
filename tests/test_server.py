"""OpenAI-compatible server tests: in-process via httpx ASGI transport, plus a live uvicorn
server exercised with the real `openai` client."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import AsyncIterator

import httpx
import openai
import pytest
import uvicorn

from pagedserve.server.app import create_app
from pagedserve.server.async_engine import AsyncLLMEngine
from tests.stub_tokenizer import EOS, install
from tests.test_engine import make_engine

pytestmark = pytest.mark.anyio
MODEL = "tiny"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def served() -> AsyncIterator[tuple[httpx.AsyncClient, AsyncLLMEngine]]:
    aeng = AsyncLLMEngine(install(make_engine(max_model_len=512, num_blocks=512)))
    app = create_app(aeng, MODEL, manage_lifespan=False)
    aeng.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            yield client, aeng
    finally:
        aeng.stop()


def sse_events(body: str) -> list[str]:
    """Data payloads of each SSE event, in order."""
    out = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        data = [ln[5:].strip() for ln in block.split("\n") if ln.startswith("data:")]
        if data:
            out.append("\n".join(data))
    return out


def text_of(ids: list[int]) -> str:
    return bytes(i for i in ids if i != EOS).decode(errors="replace")


async def test_health_and_models(served) -> None:
    client, _ = served
    assert (await client.get("/health")).json() == {"status": "ok"}
    body = (await client.get("/v1/models")).json()
    assert body["object"] == "list" and body["data"][0]["id"] == MODEL


async def test_completion_non_stream(served) -> None:
    client, aeng = served
    r = await client.post("/v1/completions", json={
        "model": MODEL, "prompt": "hello world", "max_tokens": 10, "temperature": 0,
        "ignore_eos": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion" and body["id"].startswith("cmpl-")
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 10, "total_tokens": 21}
    # Same ids through the async engine directly -> the text is their byte decode.
    from pagedserve.sched.request import SamplingParams
    outs = [o async for o in aeng.generate("ref", list(b"hello world"),
                                           SamplingParams.greedy(10, ignore_eos=True))]
    assert body["choices"][0]["text"] == text_of(outs[-1].output_token_ids)


async def test_completion_token_ids_prompt_and_max_tokens(served) -> None:
    client, _ = served
    r = await client.post("/v1/completions", json={
        "model": MODEL, "prompt": [10, 20, 30, 40], "max_tokens": 3, "temperature": 0,
        "ignore_eos": True})
    assert r.status_code == 200
    assert r.json()["usage"]["completion_tokens"] == 3
    r = await client.post("/v1/completions", json={"model": MODEL, "prompt": ["a", "b"]})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


async def test_completion_stream(served) -> None:
    client, _ = served
    payload = {"model": MODEL, "prompt": "stream me", "max_tokens": 8, "temperature": 0,
               "ignore_eos": True}
    full = (await client.post("/v1/completions", json=payload)).json()
    r = await client.post("/v1/completions", json={**payload, "stream": True})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert len(chunks) == 8
    assert "".join(c["choices"][0]["text"] for c in chunks) == full["choices"][0]["text"]
    assert [c["choices"][0]["finish_reason"] for c in chunks[:-1]] == [None] * 7
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"] == full["usage"]


async def test_chat_non_stream_and_stream(served) -> None:
    client, _ = served
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    payload = {"model": MODEL, "messages": msgs, "max_tokens": 6, "temperature": 0,
               "ignore_eos": True}
    r = await client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    prompt = "<system>be brief\n<user>hi\n<assistant>"
    assert body["usage"]["prompt_tokens"] == len(prompt.encode())
    text = body["choices"][0]["message"]["content"]

    r = await client.post("/v1/chat/completions", json={**payload, "stream": True})
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[1:]) == text
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 6


async def test_stop_string_truncates(served) -> None:
    client, _ = served
    base = {"model": MODEL, "prompt": "abc", "max_tokens": 24, "temperature": 1.5, "seed": 1,
            "ignore_eos": True}
    text = (await client.post("/v1/completions", json=base)).json()["choices"][0]["text"]
    stop = "LF"  # seed 1 on the tiny model yields "...[pLFm0..." a few tokens in
    assert 3 < text.find(stop) < 12
    r = (await client.post("/v1/completions", json={**base, "stop": [stop]})).json()
    out = r["choices"][0]["text"]
    assert out == text[:text.find(stop)]
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"]["completion_tokens"] < 24


async def test_rejects_n_and_echo(served) -> None:
    client, _ = served
    r = await client.post("/v1/completions", json={"model": MODEL, "prompt": "x", "n": 2})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    r = await client.post("/v1/completions", json={"model": MODEL, "prompt": "x", "echo": True})
    assert r.status_code == 400


async def test_seeded_determinism(served) -> None:
    client, _ = served
    payload = {"model": MODEL, "prompt": "seed test", "max_tokens": 12, "temperature": 0.9,
               "seed": 7, "ignore_eos": True}
    a = (await client.post("/v1/completions", json=payload)).json()["choices"][0]["text"]
    b = (await client.post("/v1/completions", json=payload)).json()["choices"][0]["text"]
    assert a == b


async def test_metrics(served) -> None:
    client, _ = served
    await client.post("/v1/completions", json={"model": MODEL, "prompt": "m", "max_tokens": 2,
                                               "ignore_eos": True})
    m = (await client.get("/metrics")).json()
    for k in ("requests_running", "requests_waiting", "steps_total", "prefill_steps_total",
              "decode_steps_total", "generated_tokens_total", "kv_blocks_free",
              "kv_blocks_total", "requests_finished_total"):
        assert k in m
    assert m["requests_finished_total"] >= 1 and m["requests_running"] == 0
    prom = (await client.get("/metrics", params={"format": "prometheus"})).text
    assert "pagedserve_steps_total " in prom and "# TYPE pagedserve_steps_total counter" in prom


async def test_engine_not_started_is_503() -> None:
    aeng = AsyncLLMEngine(install(make_engine()))
    app = create_app(aeng, MODEL, manage_lifespan=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test") as client:
        r = await client.post("/v1/completions", json={"model": MODEL, "prompt": "x"})
    assert r.status_code == 503 and r.json()["error"]["type"] == "service_unavailable"


async def test_step_error_is_500(served, monkeypatch: pytest.MonkeyPatch) -> None:
    _, aeng = served

    def bad_step():
        raise RuntimeError("bad")

    monkeypatch.setattr(aeng.engine, "step", bad_step)
    app = create_app(aeng, MODEL, manage_lifespan=False)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/v1/completions", json={"model": MODEL, "prompt": "x"})
    assert r.status_code == 500 and "bad" in r.json()["error"]["message"]


# ---- live server + real openai client -------------------------------------------------
class LiveServer:
    def __init__(self) -> None:
        self.aeng = AsyncLLMEngine(install(make_engine(max_model_len=2048, num_blocks=1024)))
        app = create_app(self.aeng, MODEL)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                                    log_level="warning", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> "LiveServer":
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture(scope="module")
def live() -> AsyncIterator[LiveServer]:
    with LiveServer() as srv:
        yield srv


def test_openai_client_completions(live: LiveServer) -> None:
    client = openai.OpenAI(base_url=live.base_url, api_key="x")
    full = client.completions.create(model=MODEL, prompt="hello", max_tokens=6, temperature=0,
                                     extra_body={"ignore_eos": True})
    assert full.choices[0].finish_reason == "length" and full.usage.completion_tokens == 6
    parts = []
    for chunk in client.completions.create(model=MODEL, prompt="hello", max_tokens=6,
                                           temperature=0, stream=True,
                                           extra_body={"ignore_eos": True}):
        parts.append(chunk.choices[0].text)
    assert "".join(parts) == full.choices[0].text


def test_openai_client_chat_stream(live: LiveServer) -> None:
    client = openai.OpenAI(base_url=live.base_url, api_key="x")
    msgs = [{"role": "user", "content": "hey"}]
    full = client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=5,
                                          temperature=0, extra_body={"ignore_eos": True})
    assert full.choices[0].message.role == "assistant"
    roles, parts, finish = [], [], None
    for chunk in client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=5,
                                                temperature=0, stream=True,
                                                extra_body={"ignore_eos": True}):
        d = chunk.choices[0].delta
        if d.role:
            roles.append(d.role)
        parts.append(d.content or "")
        finish = chunk.choices[0].finish_reason or finish
    assert roles == ["assistant"] and finish == "length"
    assert "".join(parts) == full.choices[0].message.content


def test_client_disconnect_aborts(live: LiveServer) -> None:
    client = openai.OpenAI(base_url=live.base_url, api_key="x")
    stream = client.completions.create(model=MODEL, prompt="never ending", max_tokens=1500,
                                       temperature=0, stream=True,
                                       extra_body={"ignore_eos": True})
    it = iter(stream)
    next(it)
    next(it)
    stream.close()
    deadline = time.time() + 1.0
    while time.time() < deadline:
        m = httpx.get(f"http://127.0.0.1:{live.port}/metrics").json()
        if m["requests_running"] == 0 and m["requests_aborted_total"] >= 1:
            break
        time.sleep(0.02)
    assert m["requests_running"] == 0 and m["requests_aborted_total"] >= 1
    assert m["kv_blocks_free"] == m["kv_blocks_total"]
