"""Ablation: the same trace through one engine per configuration.

    naive              per-sequence growing KV, continuous batching
    static             paged_torch + static batching (batch runs until its longest member ends)
    paged_torch        paged KV (gather attention), continuous batching
    paged_torch+prefix   + prefix caching
    paged_flash        flash-attn paged decode / varlen prefill (GPU)
    paged_flash+graphs   + CUDA graphs for decode steps

    python -m pagedserve.bench.ablation --model models/Qwen2.5-0.5B-Instruct --device cuda \
        --dtype float16 --trace-n 200 --request-rate 8 --out results/ablation.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pagedserve.bench.metrics import BenchSummary, records_to_json, summarize
from pagedserve.bench.offline import run_offline_benchmark
from pagedserve.bench.trace import TraceRequest, generate_trace, trace_summary, trace_to_json
from pagedserve.config import EngineConfig, ModelConfig

ALL_CONFIGS = ("naive", "static", "paged_torch", "paged_torch+prefix", "paged_flash",
               "paged_flash+graphs")
DEFAULT_CONFIGS = ("naive", "static", "paged_torch", "paged_torch+prefix")


@dataclass(frozen=True)
class AblationConfig:
    name: str
    attn_backend: str
    enable_prefix_caching: bool = False
    enable_cuda_graphs: bool = False
    static_batching: bool = False

    @classmethod
    def parse(cls, name: str) -> "AblationConfig":
        if name == "static":
            return cls(name, "paged_torch", static_batching=True)
        base, _, flags = name.partition("+")
        if base not in ("naive", "paged_torch", "paged_flash"):
            raise ValueError(f"unknown config {name!r}; choose from {ALL_CONFIGS}")
        opts = set(flags.split("+")) if flags else set()
        unknown = opts - {"prefix", "graphs"}
        if unknown:
            raise ValueError(f"unknown flags {unknown} in config {name!r}")
        return cls(name, base, enable_prefix_caching="prefix" in opts,
                   enable_cuda_graphs="graphs" in opts)

    def engine_config(self, args: argparse.Namespace) -> EngineConfig:
        return EngineConfig(
            device=args.device, dtype=EngineConfig.dtype_from_str(args.dtype),
            block_size=args.block_size, num_gpu_blocks=args.num_blocks,
            max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len, attn_backend=self.attn_backend,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_cuda_graphs=self.enable_cuda_graphs, seed=args.seed)


def build_engine(cfg: AblationConfig, args: argparse.Namespace):
    from pagedserve.engine import LLMEngine

    ecfg = cfg.engine_config(args)
    if args.tiny:
        from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic

        mcfg = ModelConfig.tiny()
        model = Qwen2ForCausalLM(mcfg)
        reset_parameters_deterministic(model, args.seed)
        if ecfg.num_gpu_blocks is None:
            ecfg.num_gpu_blocks = 512
        return LLMEngine(model, mcfg, ecfg, tokenizer=None)
    if not args.model:
        raise SystemExit("--model DIR is required unless --tiny")
    return LLMEngine.from_pretrained(args.model, ecfg)


def run_config(cfg: AblationConfig, trace: list[TraceRequest], args: argparse.Namespace,
               ) -> dict[str, Any]:
    t_build = time.perf_counter()
    engine = build_engine(cfg, args)
    build_s = time.perf_counter() - t_build
    if args.warmup:
        # One short request to page in kernels / capture graphs before timing.
        warm = generate_trace(1, seed=999, vocab_size=engine.model_config.vocab_size,
                              max_prompt_len=32, max_output_len=8)
        run_offline_benchmark(engine, warm, static_batching=cfg.static_batching)
        engine.reset()
    records, steps = run_offline_benchmark(engine, trace, static_batching=cfg.static_batching,
                                           progress=args.progress)
    summary = summarize(records, slo_ttft_ms=args.slo_ttft_ms, slo_tpot_ms=args.slo_tpot_ms)
    kv_util = float(np.mean([s.kv_utilization for s in steps])) if steps else float("nan")
    decode = [s for s in steps if not s.is_prefill]
    prefill = [s for s in steps if s.is_prefill]
    result = {
        "config": cfg.name, "attn_backend": cfg.attn_backend,
        "enable_prefix_caching": cfg.enable_prefix_caching,
        "enable_cuda_graphs": cfg.enable_cuda_graphs, "static_batching": cfg.static_batching,
        "summary": summary.to_dict(),
        "kv_utilization_mean": kv_util,
        "num_steps": len(steps), "num_prefill_steps": len(prefill),
        "num_decode_steps": len(decode),
        "num_preempted": int(sum(s.num_preempted for s in steps)),
        "mean_decode_batch": float(np.mean([s.num_seqs for s in decode])) if decode else 0.0,
        "decode_forward_ms_mean": float(np.mean([s.forward_ms for s in decode])) if decode else 0.0,
        "prefill_forward_ms_mean": (float(np.mean([s.forward_ms for s in prefill]))
                                    if prefill else 0.0),
        "build_s": build_s,
    }
    if args.save_records:
        result["records"] = records_to_json(records)
    del engine
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def markdown_table(results: list[dict[str, Any]]) -> str:
    head = ("| config | tok/s | req/s | TTFT p50/p99 ms | TPOT p50/p99 ms | KV util |\n"
            "|---|---:|---:|---:|---:|---:|")
    rows = []
    for r in results:
        s = BenchSummary(**{k: v for k, v in r["summary"].items()
                            if k not in ("ttft_ms", "tpot_ms", "e2e_ms")})
        ttft, tpot = r["summary"]["ttft_ms"], r["summary"]["tpot_ms"]
        rows.append(f"| {r['config']} | {s.throughput_tok_s:.1f} | {s.requests_per_s:.2f} | "
                    f"{ttft['p50']:.1f} / {ttft['p99']:.1f} | {tpot['p50']:.1f} / {tpot['p99']:.1f} | "
                    f"{100 * r['kv_utilization_mean']:.1f}% |")
    return "\n".join([head, *rows])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pagedserve.bench.ablation", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="HF snapshot dir (omit with --tiny)")
    p.add_argument("--tiny", action="store_true", help="2-layer random model, no weights")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                   help=f"comma list from {','.join(ALL_CONFIGS)}")
    p.add_argument("--trace-n", type=int, default=100)
    p.add_argument("--request-rate", type=float, default=None,
                   help="Poisson arrival rate req/s; omit or 'inf' for all at t=0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shared-prefix-len", type=int, default=0)
    p.add_argument("--max-prompt-len", type=int, default=None)
    p.add_argument("--max-output-len", type=int, default=None)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--slo-ttft-ms", type=float, default=None)
    p.add_argument("--slo-tpot-ms", type=float, default=None)
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--save-records", action="store_true", help="keep per-request rows in JSON")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--out", default="results/ablation.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configs = [AblationConfig.parse(c.strip()) for c in args.configs.split(",") if c.strip()]
    vocab = ModelConfig.tiny().vocab_size if args.tiny else ModelConfig.from_hf_dir(
        args.model).vocab_size
    max_prompt = args.max_prompt_len or (64 if args.tiny else min(2048, args.max_model_len // 2))
    max_output = args.max_output_len or (16 if args.tiny else min(2048, args.max_model_len // 2))
    trace = generate_trace(args.trace_n, seed=args.seed, request_rate=args.request_rate,
                           shared_prefix_len=args.shared_prefix_len, vocab_size=vocab,
                           max_prompt_len=max_prompt, max_output_len=max_output)
    if args.tiny:
        # Keep the toy run fast: median lengths from the default lognormal are too long.
        for r in trace:
            r.prompt_len = min(r.prompt_len, max_prompt)
            r.output_len = min(r.output_len, max_output)
            r.prompt_ids = (r.prompt_ids or [])[: r.prompt_len]
    results: list[dict[str, Any]] = []
    for cfg in configs:
        print(f"[ablation] running {cfg.name} ...", file=sys.stderr, flush=True)
        res = run_config(cfg, trace, args)
        s = res["summary"]
        print(f"[ablation] {cfg.name}: {s['throughput_tok_s']:.1f} tok/s, "
              f"{res['num_steps']} steps, KV util {100 * res['kv_utilization_mean']:.1f}%",
              file=sys.stderr, flush=True)
        results.append(res)
    payload = {
        "kind": "ablation",
        "model": "tiny" if args.tiny else args.model,
        "device": args.device, "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "trace": {"n": args.trace_n, "seed": args.seed, "request_rate": args.request_rate,
                  "shared_prefix_len": args.shared_prefix_len, **trace_summary(trace)},
        "trace_requests": trace_to_json(trace, include_ids=False),
        "args": {k: v for k, v in vars(args).items()},
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(markdown_table(results))
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
