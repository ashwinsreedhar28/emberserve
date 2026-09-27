"""`pagedserve serve ...` (OpenAI-compatible server) and `pagedserve generate ...` (one-shot)."""

from __future__ import annotations

import argparse
import sys
import time

from pagedserve.config import EngineConfig


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF snapshot directory")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--device", default="cpu")
    p.add_argument("--attn-backend", choices=["naive", "paged_torch", "paged_flash", "paged_triton", "mla_torch", "mla_triton"],
                   default="paged_torch")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=None,
                   help="prefill token budget per step (default 8192; 2048 when chunked prefill is on: "
                        "512 tightens the TPOT tail at saturation for ~6%% throughput at 7B)")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--enable-prefix-caching", action="store_true")
    p.add_argument("--enable-cuda-graphs", action="store_true")
    p.add_argument("--enable-chunked-prefill", dest="enable_chunked_prefill", action="store_true",
                   default=None,
                   help="mix decode tokens and prompt chunks in every step; --max-num-batched-tokens "
                        "becomes the per-step cap. Default: on for --device cuda (with a 2048-token "
                        "cap unless --max-num-batched-tokens is given), off otherwise")
    p.add_argument("--no-chunked-prefill", dest="enable_chunked_prefill", action="store_false")
    p.add_argument("--async-scheduling", dest="async_scheduling", action="store_true", default=False,
                   help="launch step N+1 before reading step N's tokens back (vLLM v1 style): the "
                        "CPU work of a step overlaps the GPU work of the previous one. Outputs of "
                        "a step arrive one step() later; EOS-ended requests compute one discarded "
                        "token. Default: off")
    p.add_argument("--no-async-scheduling", dest="async_scheduling", action="store_false")


def engine_config_from_args(args: argparse.Namespace) -> EngineConfig:
    chunked = args.enable_chunked_prefill
    if chunked is None:
        chunked = args.device.startswith("cuda")
    budget = args.max_num_batched_tokens
    if budget is None:
        budget = 2048 if chunked else 8192
    return EngineConfig(model_dir=args.model, device=args.device,
                        dtype=EngineConfig.dtype_from_str(args.dtype),
                        block_size=args.block_size, num_gpu_blocks=args.num_blocks,
                        max_num_seqs=args.max_num_seqs,
                        max_num_batched_tokens=budget,
                        max_model_len=args.max_model_len, attn_backend=args.attn_backend,
                        enable_prefix_caching=args.enable_prefix_caching,
                        enable_cuda_graphs=args.enable_cuda_graphs,
                        enable_chunked_prefill=chunked,
                        async_scheduling=bool(getattr(args, "async_scheduling", False)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pagedserve")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the OpenAI-compatible HTTP server")
    _add_engine_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", default=None)
    serve.add_argument("--engine-process", dest="engine_process", action="store_true", default=None,
                       help="run the engine in its own process (tokenizer + HTTP stay here); removes "
                            "the GIL contention between SSE delivery and the step loop. Default: on "
                            "for --device cuda, off otherwise")
    serve.add_argument("--no-engine-process", dest="engine_process", action="store_false")

    gen = sub.add_parser("generate", help="generate a completion for one prompt")
    _add_engine_args(gen)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--chat", action="store_true", help="wrap the prompt as a user chat turn")
    gen.add_argument("--max-tokens", type=int, default=128)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=-1)
    gen.add_argument("--seed", type=int, default=None)
    return parser


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from pagedserve.server.app import build_app_from_args

    engine_process = args.engine_process
    if engine_process is None:
        engine_process = args.device.startswith("cuda")
    app = build_app_from_args(args.model, engine_config_from_args(args), args.served_model_name,
                              engine_process=engine_process)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from pagedserve.llm import LLM
    from pagedserve.sched.request import SamplingParams

    llm = LLM(args.model, engine_config_from_args(args))
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                            top_p=args.top_p, top_k=args.top_k, seed=args.seed)
    t0 = time.perf_counter()
    if args.chat:
        res = llm.chat([[{"role": "user", "content": args.prompt}]], params)[0]
    else:
        res = llm.generate([args.prompt], params)[0]
    dt = time.perf_counter() - t0
    n = len(res.output_token_ids)
    print(res.text)
    print(f"\n[{n} tokens in {dt:.2f}s, {n / dt:.1f} tok/s, finish={res.finish_reason}]",
          file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_serve(args) if args.command == "serve" else cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
