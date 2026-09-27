"""Rate sweep against an OpenAI-compatible server: vLLM, pagedserve, or any live endpoint.

Launches the server as a subprocess (never imports vllm), waits for `/health`, replays
the same seeded trace at each request rate through `run_http_benchmark`, writes
`results/<name>.json`, and kills the server.

    # vLLM baseline on the pod
    python -m pagedserve.bench.run_vllm_baseline --server vllm --model Qwen/Qwen2.5-0.5B-Instruct \
        --rates 1,2,4,8,16,inf --trace-n 200 --name vllm
    # pagedserve through its own HTTP server
    python -m pagedserve.bench.run_vllm_baseline --server pagedserve --model models/Qwen2.5-0.5B-Instruct \
        --server-args "--device cuda --dtype float16 --attn-backend paged_flash --enable-cuda-graphs" \
        --name pagedserve
    # an endpoint that is already up (e.g. a Runpod vLLM pod)
    python -m pagedserve.bench.run_vllm_baseline --base-url https://<pod>-8000.proxy.runpod.net \
        --model Qwen/Qwen2.5-0.5B-Instruct --name runpod_vllm
    # a hosted API (OpenRouter): text prompts, no /health, no ignore_eos, its own route
    OPENAI_API_KEY=sk-or-... python -m pagedserve.bench.run_vllm_baseline --hosted \
        --base-url https://openrouter.ai/api/v1 --model deepseek/deepseek-chat \
        --tokenizer models/Qwen2.5-0.5B-Instruct --rates 1,2,4 --trace-n 50 --name openrouter_deepseek
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pagedserve.bench.load import fetch_metrics, run_http_benchmark, run_http_benchmark_procs, wait_for_health
from pagedserve.bench.metrics import records_to_json, summarize
from pagedserve.bench.trace import generate_trace, sharegpt_trace, trace_summary


def parse_rates(s: str) -> list[float | None]:
    out: list[float | None] = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(None if tok in ("inf", "none", "0") else float(tok))
    return out


def server_command(args: argparse.Namespace) -> list[str]:
    extra = shlex.split(args.server_args) if args.server_args else []
    if args.server == "vllm":
        cmd = [args.vllm_bin, "serve", args.model, "--port", str(args.port), "--host", args.host,
               "--served-model-name", args.served_model_name or args.model]
        if args.dtype:
            cmd += ["--dtype", args.dtype]
        if args.max_model_len:
            cmd += ["--max-model-len", str(args.max_model_len)]
        return cmd + extra
    if args.server == "pagedserve":
        cmd = [sys.executable, "-m", "pagedserve.cli", "serve", "--model", args.model,
               "--port", str(args.port), "--host", args.host]
        if args.served_model_name:
            cmd += ["--served-model-name", args.served_model_name]
        if args.dtype:
            cmd += ["--dtype", args.dtype]
        if args.max_model_len:
            cmd += ["--max-model-len", str(args.max_model_len)]
        return cmd + extra
    raise ValueError(args.server)


def _vocab_size_for(model: str) -> int:
    """`vocab_size` from a local snapshot's config.json; the Qwen2.5 default otherwise.
    Synthetic prompts are random token ids, and ids past the model's vocabulary are
    rejected by vLLM ("Token id ... is out of vocabulary") and by pagedserve."""
    cfg = Path(model) / "config.json"
    if cfg.exists():
        try:
            return int(json.loads(cfg.read_text())["vocab_size"])
        except (KeyError, ValueError, OSError):
            pass
    return 151_936


def launch(cmd: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")  # noqa: SIM115 - handed to Popen, closed in kill()
    print(f"[baseline] launching: {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)
    env = dict(os.environ)
    # A vLLM in its own venv (/opt/vllm/bin/vllm) JIT-compiles FlashInfer sampling kernels
    # and needs that venv's `ninja` on PATH; put the executable's directory first.
    bin_dir = os.path.dirname(os.path.abspath(cmd[0]))
    if os.path.isdir(bin_dir):
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)


def kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc.stdout is not None:
        proc.stdout.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pagedserve.bench.run_vllm_baseline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", choices=["vllm", "pagedserve"], default="vllm")
    p.add_argument("--vllm-bin", default="vllm",
                   help="path to the vllm executable, e.g. /opt/vllm/bin/vllm when vLLM lives in its own venv")
    p.add_argument("--base-url", default=None, help="use a running endpoint; no subprocess")
    p.add_argument("--hosted", action="store_true",
                   help="a hosted OpenAI-compatible API: implies --completions-path /completions, "
                        "--no-health and --no-ignore-eos (requires --base-url and --tokenizer)")
    p.add_argument("--completions-path", default=None,
                   help="completions route relative to --base-url (default /v1/completions)")
    p.add_argument("--client-procs", type=int, default=1,
                   help="split the load generator across N processes (one event loop each): a "
                        "single process parsing SSE tops out near 20k events/s, which a 200-stream "
                        "burst on a fast server reaches, and then TTFT/TPOT measure the client")
    p.add_argument("--no-health", dest="health", action="store_false",
                   help="do not poll <base-url>/health before the sweep")
    p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                   help="leave the vLLM/pagedserve `ignore_eos` field out of requests")
    p.add_argument("--model", required=True, help="model name/dir; also the `model` field sent")
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--server-args", default="", help="extra args appended to the server cmd")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--rates", default="1,2,4,8,16,inf")
    p.add_argument("--trace-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shared-prefix-len", type=int, default=0)
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--max-output-len", type=int, default=512)
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--startup-timeout-s", type=float, default=900.0)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "x"))
    p.add_argument("--vocab-size", type=int, default=None,
                   help="range of the synthetic prompt ids; default: vocab_size from <model>/config.json "
                        "when --model is a local directory, else 151936 (Qwen2.5)")
    p.add_argument("--tokenizer", default=None,
                   help="HF dir for text prompts (default: send token ids)")
    p.add_argument("--sharegpt", default=None,
                   help="ShareGPT JSON dump: real conversations as the trace (first human turn as the "
                        "prompt, first reply's length as output_len); needs --tokenizer for the "
                        "length accounting. scripts/download_sharegpt.py fetches the file")
    p.add_argument("--slo-ttft-ms", type=float, default=None)
    p.add_argument("--slo-tpot-ms", type=float, default=None)
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--save-records", action="store_true")
    p.add_argument("--name", default=None, help="results/<name>.json (default: --server)")
    p.add_argument("--out-dir", default="results")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    name = args.name or args.server
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    if args.tokenizer:
        from pagedserve.tokenizer import Tokenizer

        tokenizer = Tokenizer(args.tokenizer)
    model_name = args.served_model_name or args.model
    if args.hosted:
        if args.base_url is None or tokenizer is None:
            raise SystemExit("--hosted needs --base-url and --tokenizer (hosted APIs take text prompts)")
        args.completions_path = args.completions_path or "/completions"
        args.health = False
        args.ignore_eos = False
    path = args.completions_path or "/v1/completions"
    ignore_eos = True if args.ignore_eos else None
    vocab_size = args.vocab_size or _vocab_size_for(args.tokenizer if args.hosted else args.model)
    print(f"[baseline] synthetic prompt ids drawn from [2, {vocab_size})", file=sys.stderr)
    proc: subprocess.Popen | None = None
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    try:
        if args.base_url is None:
            proc = launch(server_command(args), out_dir / f"{name}.server.log")
        if args.health:
            print(f"[baseline] waiting for {base_url}/health ...", file=sys.stderr)
            if not asyncio.run(wait_for_health(base_url, args.startup_timeout_s)):
                print("[baseline] server never became healthy", file=sys.stderr)
                return 1
        runs: list[dict[str, Any]] = []
        if args.warmup:
            warm = generate_trace(4, seed=1234, max_prompt_len=64, max_output_len=16,
                                  vocab_size=vocab_size)
            asyncio.run(run_http_benchmark(base_url, model_name, warm, tokenizer=tokenizer,
                                           api_key=args.api_key, progress=False, path=path,
                                           ignore_eos=ignore_eos))
        for rate in parse_rates(args.rates):
            if args.sharegpt:
                if tokenizer is None:
                    raise SystemExit("--sharegpt needs --tokenizer (to count prompt/output tokens)")
                trace = sharegpt_trace(args.sharegpt, args.trace_n, tokenizer, seed=args.seed,
                                       request_rate=rate, max_prompt_len=args.max_prompt_len,
                                       max_output_len=args.max_output_len)
            else:
                trace = generate_trace(args.trace_n, seed=args.seed, request_rate=rate,
                                       shared_prefix_len=args.shared_prefix_len,
                                       max_prompt_len=args.max_prompt_len,
                                       max_output_len=args.max_output_len, vocab_size=vocab_size)
            label = "inf" if rate is None else f"{rate:g}"
            print(f"[baseline] {name} @ rate={label} req/s, n={len(trace)}", file=sys.stderr)
            before = asyncio.run(fetch_metrics(base_url)) if not args.hosted else None
            t0 = time.perf_counter()
            if args.client_procs > 1:
                records = run_http_benchmark_procs(
                    base_url, model_name, trace, args.client_procs,
                    max_concurrency=args.max_concurrency, timeout_s=args.timeout_s,
                    api_key=args.api_key, path=path, ignore_eos=ignore_eos)
            else:
                records = asyncio.run(run_http_benchmark(
                    base_url, model_name, trace, max_concurrency=args.max_concurrency,
                    timeout_s=args.timeout_s, api_key=args.api_key, tokenizer=tokenizer,
                    path=path, ignore_eos=ignore_eos))
            wall = time.perf_counter() - t0
            summary = summarize(records, slo_ttft_ms=args.slo_ttft_ms,
                                slo_tpot_ms=args.slo_tpot_ms)
            print(summary.one_line(f"{name}@{label}"), file=sys.stderr)
            run = {"request_rate": rate, "wall_s": wall, "summary": summary.to_dict(),
                   "trace": {**trace_summary(trace),
                             "source": "sharegpt" if args.sharegpt else "synthetic"}}
            after = asyncio.run(fetch_metrics(base_url)) if before is not None else None
            if after is not None:
                # this run's share of the server's counters (speculation drafted/accepted,
                # server-side latency sums...)
                sc = {k: after[k] - before.get(k, 0) for k in after
                      if k.endswith(("_total", "_sum", "_count")) and isinstance(after[k], (int, float))}
                run["server_counters"] = sc
                d, a = sc.get("spec_drafted_total", 0), sc.get("spec_accepted_total", 0)
                if d:
                    run["spec_acceptance"] = a / d
                    print(f"[baseline] speculation: {a}/{d} drafts accepted ({a / d:.1%})",
                          file=sys.stderr)
                lat = {}
                for metric in ("ttft", "tpot", "e2e"):  # (not `name`: that is the sweep's)
                    if sc.get(f"{metric}_count", 0) > 0:
                        lat[f"{metric}_ms_mean"] = 1e3 * sc[f"{metric}_s_sum"] / sc[f"{metric}_count"]
                if lat:
                    # measured inside the server from the request's arrival: no client queueing
                    run["server_latency"] = lat
                    print("[baseline] server-side means: " + ", ".join(f"{k} {v:.1f}" for k, v in lat.items()),
                          file=sys.stderr)
            if args.save_records:
                run["records"] = records_to_json(records)
            runs.append(run)
            payload = {"kind": "sweep", "system": name, "server": args.server,
                       "base_url": base_url, "model": args.model,
                       "args": vars(args), "runs": runs}
            (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=1))
        print(f"[baseline] wrote {out_dir / f'{name}.json'}", file=sys.stderr)
        return 0
    finally:
        if proc is not None:
            kill(proc)


if __name__ == "__main__":
    sys.exit(main())
