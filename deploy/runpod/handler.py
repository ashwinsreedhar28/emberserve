"""Runpod Serverless worker for pagedserve.

One engine per worker (built at cold start), continuous batching across the jobs the
worker is handling concurrently (`concurrency_modifier`), streamed output.

Job input, any of:

    {"input": {"prompt": "Once upon a time", "max_tokens": 64}}
    {"input": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64, "temperature": 0.7}}
    {"input": {"prompt_token_ids": [1, 2, 3], "max_tokens": 8}}

plus the OpenAI sampling fields (`temperature`, `top_p`, `top_k`, `seed`, `stop`,
`stop_token_ids`, `ignore_eos`, `repetition_penalty`). Streamed chunks are
`{"text": delta}`; the last one adds `finish_reason` and `usage`. `runsync` / `run` get
the chunks aggregated (`return_aggregate_stream`).

Environment (all optional): MODEL_DIR (a snapshot baked into the image; default
/models/model), MODEL_REPO (download at cold start instead), DTYPE (float16), ATTN_BACKEND
(paged_flash), BLOCK_SIZE (256), MAX_MODEL_LEN (4096), MAX_NUM_SEQS (256), CUDA_GRAPHS (1),
CHUNKED_PREFILL (default: on for checkpoints >= 4 GB), ASYNC_SCHEDULING (1), MAX_CONCURRENCY
(jobs per worker, 64).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pagedserve.config import EngineConfig  # noqa: E402
from pagedserve.sched.request import SamplingParams  # noqa: E402


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def engine_config(model_dir: str | None = None) -> EngineConfig:
    """Same defaults as `pagedserve serve --device cuda`: async scheduling on, chunked
    prefill on for checkpoints >= 4 GB (CHUNKED_PREFILL=0/1 forces it)."""
    from pagedserve.cli import CHUNKED_PREFILL_MIN_BYTES, checkpoint_bytes

    forced = os.environ.get("CHUNKED_PREFILL")
    if forced is not None:
        chunked = forced == "1"
    else:
        chunked = checkpoint_bytes(model_dir) >= CHUNKED_PREFILL_MIN_BYTES
    return EngineConfig(
        device="cuda", dtype=EngineConfig.dtype_from_str(env("DTYPE", "float16")),
        attn_backend=env("ATTN_BACKEND", "paged_flash"), block_size=int(env("BLOCK_SIZE", "256")),
        max_model_len=int(env("MAX_MODEL_LEN", "4096")), max_num_seqs=int(env("MAX_NUM_SEQS", "256")),
        max_num_batched_tokens=2048 if chunked else 8192, enable_chunked_prefill=chunked,
        enable_cuda_graphs=env("CUDA_GRAPHS", "1") == "1",
        async_scheduling=env("ASYNC_SCHEDULING", "1") == "1")


def resolve_model_dir() -> str:
    """The snapshot to serve: baked into the image, or fetched at cold start."""
    model_dir = env("MODEL_DIR", "/models/model")
    repo = os.environ.get("MODEL_REPO")
    if repo and not (Path(model_dir) / "config.json").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_dir=model_dir,
                          allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json",
                                          "*.txt", "*.py", "*.model", "*.tiktoken"])
    return model_dir


def to_sampling_params(inp: dict[str, Any]) -> SamplingParams:
    stop = inp.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    return SamplingParams(
        max_tokens=int(inp.get("max_tokens", 128)),
        temperature=float(inp.get("temperature", 0.0)),
        top_p=float(inp.get("top_p", 1.0)),
        top_k=int(inp.get("top_k", -1)),
        seed=inp.get("seed"),
        stop=list(stop or []),
        stop_token_ids=list(inp.get("stop_token_ids") or []),
        ignore_eos=bool(inp.get("ignore_eos", False)),
        repetition_penalty=float(inp.get("repetition_penalty", 1.0)),
    )


def prompt_ids_for(inp: dict[str, Any], tokenizer) -> list[int]:
    if "prompt_token_ids" in inp:
        return [int(t) for t in inp["prompt_token_ids"]]
    if "messages" in inp:
        if tokenizer is None:
            raise ValueError("this model has no tokenizer; send prompt_token_ids")
        return tokenizer.apply_chat_template(inp["messages"], add_generation_prompt=True)
    if "prompt" in inp:
        if tokenizer is None:
            raise ValueError("this model has no tokenizer; send prompt_token_ids")
        return tokenizer.encode(inp["prompt"])
    raise ValueError("input needs one of: prompt, messages, prompt_token_ids")


def make_handler(async_engine, tokenizer):
    """The job handler as an async generator, closed over one engine (testable without the
    Runpod SDK)."""

    async def handler(job: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        inp = job.get("input") or {}
        try:
            prompt_ids = prompt_ids_for(inp, tokenizer)
            params = to_sampling_params(inp)
        except (ValueError, TypeError, KeyError) as exc:
            yield {"error": str(exc)}
            return
        request_id = str(job.get("id") or id(job))
        n_out = 0
        gen = async_engine.generate(request_id, prompt_ids, params)
        try:
            async for out in gen:
                n_out = len(out.output_token_ids)
                chunk: dict[str, Any] = {"text": out.text_delta}
                if out.finished:
                    chunk["finish_reason"] = out.finish_reason.value if out.finish_reason else None
                    chunk["usage"] = {"prompt_tokens": len(prompt_ids), "completion_tokens": n_out,
                                      "total_tokens": len(prompt_ids) + n_out}
                yield chunk
        finally:
            await gen.aclose()

    return handler


def main() -> None:
    import runpod

    from pagedserve.engine import LLMEngine
    from pagedserve.server.async_engine import AsyncLLMEngine

    model_dir = resolve_model_dir()
    engine = LLMEngine.from_pretrained(model_dir, engine_config(model_dir))
    async_engine = AsyncLLMEngine(engine)
    async_engine.start()
    max_concurrency = int(env("MAX_CONCURRENCY", "64"))
    runpod.serverless.start({
        "handler": make_handler(async_engine, engine.tokenizer),
        "return_aggregate_stream": True,
        "concurrency_modifier": lambda current: max_concurrency,
    })


if __name__ == "__main__":
    main()
