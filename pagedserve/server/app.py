"""FastAPI app exposing the OpenAI-compatible routes over an `AsyncLLMEngine`.

`create_app` is a factory so tests can inject a tiny engine; `build_app_from_args` loads
a real model. Errors use OpenAI's `{"error": {...}}` shape everywhere.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse

try:  # ~5x faster than json.dumps for the small per-token SSE bodies; optional
    import orjson

    def _dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode()
except ImportError:  # pragma: no cover - depends on the environment
    def _dumps(obj: Any) -> str:
        return json.dumps(obj, separators=(",", ":"))

from pagedserve.config import EngineConfig
from pagedserve.engine import LLMEngine
from pagedserve.sched.request import RequestOutput
from pagedserve.server.async_engine import AsyncLLMEngine, EngineNotRunningError
from pagedserve.server.openai_types import (ChatCompletionChoice, ChatCompletionChunk,
                                            ChatCompletionChunkChoice, ChatCompletionMessage,
                                            ChatCompletionRequest, ChatCompletionResponse,
                                            ChatDelta, CompletionChoice, CompletionRequest,
                                            CompletionResponse, ErrorResponse, ModelCard,
                                            ModelList, Usage, new_id, now, to_sampling_params)

Chunk = Callable[[RequestOutput], dict[str, Any]]


def _error(status: int, message: str, type_: str) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content=ErrorResponse.of(message, type_, status).model_dump())


def _finish(out: RequestOutput) -> str | None:
    return out.finish_reason.value if out.finish_reason is not None else None


def _prometheus(metrics: dict[str, int | float | bool]) -> str:
    lines = []
    for k, v in metrics.items():
        lines.append(f"# TYPE pagedserve_{k} {'counter' if k.endswith('_total') else 'gauge'}")
        lines.append(f"pagedserve_{k} {float(v) if isinstance(v, bool) else v}")
    return "\n".join(lines) + "\n"


def create_app(async_engine: AsyncLLMEngine, model_name: str,
               manage_lifespan: bool = True) -> FastAPI:
    """Build the app. With `manage_lifespan` the async engine starts/stops with the server."""

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if manage_lifespan:
            async_engine.start()
        try:
            yield
        finally:
            if manage_lifespan:
                async_engine.stop()

    app = FastAPI(title="pagedserve", lifespan=lifespan)
    created = now()

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(400, str(exc.errors()[0].get("msg", exc)), "invalid_request_error")

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        types = {400: "invalid_request_error", 503: "service_unavailable"}
        return _error(exc.status_code, str(exc.detail), types.get(exc.status_code, "api_error"))

    @app.exception_handler(Exception)
    async def _internal(_: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, EngineNotRunningError):
            return _error(503, str(exc), "service_unavailable")
        return _error(500, f"{type(exc).__name__}: {exc}", "internal_error")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=model_name, created=created)])

    @app.get("/metrics")
    async def metrics(format: str | None = None):
        snap = async_engine.metrics()
        if format == "prometheus":
            return PlainTextResponse(_prometheus(snap), media_type="text/plain; version=0.0.4")
        return snap

    def tokenizer():
        tok = async_engine.engine.tokenizer
        if tok is None:
            raise HTTPException(503, "engine has no tokenizer; text prompts are unavailable")
        return tok

    def require_running() -> None:
        if not async_engine.is_running:
            raise HTTPException(503, "engine is not running")

    async def stream(request: Request, request_id: str, prompt_ids: list[int],
                     req: CompletionRequest | ChatCompletionRequest, chunk: Chunk,
                     first: dict[str, Any] | None = None) -> AsyncIterator[dict[str, str]]:
        """SSE body: an optional leading chunk, one chunk per output, a `[DONE]` sentinel.

        Client disconnects are detected by sse-starlette (it watches the ASGI receive channel
        and cancels this generator), so the `finally` below aborts the engine request; no
        per-token `request.is_disconnected()` poll, which cost a cancel scope and a receive
        await per token per stream."""
        if first is not None:
            yield {"data": _dumps(first)}
        n_out = 0
        gen = async_engine.generate(request_id, prompt_ids, to_sampling_params(req))
        try:
            async for out in gen:
                n_out = len(out.output_token_ids)
                body = chunk(out)
                if out.finished:
                    body["usage"] = Usage.of(len(prompt_ids), n_out).model_dump()
                yield {"data": _dumps(body)}
        finally:
            await gen.aclose()
        yield {"data": "[DONE]"}

    async def collect(request_id: str, prompt_ids: list[int],
                      req: CompletionRequest | ChatCompletionRequest) -> tuple[str, RequestOutput]:
        parts: list[str] = []
        async for out in async_engine.generate(request_id, prompt_ids, to_sampling_params(req)):
            parts.append(out.text_delta)
        return "".join(parts), out

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        require_running()
        p = req.prompt
        if isinstance(p, str):
            prompt_ids = tokenizer().encode(p)
        elif p and all(isinstance(x, int) for x in p):
            prompt_ids = list(p)
        else:
            raise HTTPException(400, "prompt must be a string or a list of token ids")
        rid = new_id("cmpl-")
        base = {"id": rid, "object": "text_completion", "created": now(), "model": req.model}
        if req.stream:
            def chunk(out: RequestOutput) -> dict[str, Any]:
                return {**base, "choices": [{"index": 0, "text": out.text_delta,
                                             "finish_reason": _finish(out)}]}
            return EventSourceResponse(stream(request, rid, prompt_ids, req, chunk))
        text, last = await collect(rid, prompt_ids, req)
        return CompletionResponse(
            id=rid, created=base["created"], model=req.model,
            choices=[CompletionChoice(text=text, finish_reason=_finish(last))],
            usage=Usage.of(len(prompt_ids), len(last.output_token_ids)))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        require_running()
        messages = [m.model_dump() for m in req.messages]
        prompt_ids = tokenizer().apply_chat_template(messages, add_generation_prompt=True)
        rid = new_id("chatcmpl-")
        created_at = now()
        if req.stream:
            def mk(delta: ChatDelta, finish: str | None = None) -> dict[str, Any]:
                return ChatCompletionChunk(
                    id=rid, created=created_at, model=req.model,
                    choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=finish)],
                ).model_dump(exclude_none=True)

            def chunk(out: RequestOutput) -> dict[str, Any]:
                return mk(ChatDelta(content=out.text_delta), _finish(out))
            return EventSourceResponse(stream(request, rid, prompt_ids, req, chunk,
                                              first=mk(ChatDelta(role="assistant", content=""))))
        text, last = await collect(rid, prompt_ids, req)
        return ChatCompletionResponse(
            id=rid, created=created_at, model=req.model,
            choices=[ChatCompletionChoice(message=ChatCompletionMessage(content=text),
                                          finish_reason=_finish(last))],
            usage=Usage.of(len(prompt_ids), len(last.output_token_ids)))

    return app


def build_app_from_args(model_dir: str, engine_config: EngineConfig,
                        served_model_name: str | None = None) -> FastAPI:
    """Load a model from `model_dir` and wrap it in a served app."""
    engine = LLMEngine.from_pretrained(model_dir, engine_config)
    return create_app(AsyncLLMEngine(engine), served_model_name or model_dir)
