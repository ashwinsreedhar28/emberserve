"""How many tokens per second can the API process deliver, with no model in the way?

    python scripts/bench_api_layer.py [--n 200] [--step-ms 2.5] [--max-tokens 128] [--client-procs 4] [--repeats 3]

Runs the real server (FastAPI + uvicorn, the engine core in its own process) with a
fake engine that hands every running request one token per `--step-ms`, fires `--n`
requests at once from `--client-procs` load-generator processes, and reports delivered
tok/s, TTFT and TPOT. With `--n 200 --step-ms 2.5` the fake core offers 80k tok/s; what
the client sees is the API layer's ceiling on this machine (detokenizer, per-request
queues, SSE encoding, socket writes). The 0.5B saturation point on the A100 is limited
by exactly this path.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import threading
import time
from pathlib import Path

import torch
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedserve.bench.load import run_http_benchmark_procs, wait_for_health  # noqa: E402
from pagedserve.bench.metrics import summarize  # noqa: E402
from pagedserve.bench.trace import generate_trace  # noqa: E402
from pagedserve.config import EngineConfig  # noqa: E402
from pagedserve.server.app import create_app  # noqa: E402
from pagedserve.server.async_engine import AsyncEngineCoreClient  # noqa: E402
from pagedserve.server.engine_core import EngineSpec  # noqa: E402
from pagedserve.server.fake_engine import ByteTokenizer  # noqa: E402
from pagedserve.server.multi import MultiServer  # noqa: E402


_StubTokenizer = ByteTokenizer  # (kept for older notes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--step-ms", type=float, default=2.5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--client-procs", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--api-workers", type=int, default=0,
                    help="N >= 1: run the server as `serve --api-workers N` (supervisor, one core, N API "
                         "processes; ratios between N are the point). 0 (default): the original "
                         "in-process server thread, which also reports API CPU per token")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--cprofile", default=None, metavar="OUT",
                    help="cProfile the server's event-loop thread; prints the top functions and saves the stats")
    args = ap.parse_args()
    port = args.port
    if not port:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, num_gpu_blocks=4096, max_num_seqs=1024,
                        max_model_len=4096)
    espec = EngineSpec(ecfg, tiny=True, fake_step_ms=args.step_ms)
    if args.api_workers >= 1:
        return run_multi(args, espec, port)
    client = AsyncEngineCoreClient(espec, ByteTokenizer())
    app = create_app(client, "fake")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    prof = None
    if args.cprofile:
        import cProfile

        prof = cProfile.Profile()

        def run_profiled() -> None:
            prof.enable()
            try:
                server.run()
            finally:
                prof.disable()

        threading.Thread(target=run_profiled, daemon=True).start()
    else:
        threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    if not asyncio.run(wait_for_health(base, 60)):
        print("server did not start", file=sys.stderr)
        return 1
    offered = args.n * 1e3 / args.step_ms
    print(f"fake core: {args.n} requests x 1 token / {args.step_ms} ms = {offered:,.0f} tok/s offered; "
          f"{args.max_tokens} tokens each; client procs {args.client_procs}")
    def cpu_times() -> tuple[float, float]:
        """This (API) process's user and system CPU seconds so far."""
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime, r.ru_stime

    for i in range(args.repeats):
        u0, s0 = cpu_times()
        trace = generate_trace(args.n, seed=i, request_rate=None, max_prompt_len=64, max_output_len=args.max_tokens,
                               vocab_size=256)
        for r in trace:
            r.output_len = args.max_tokens
        t0 = time.perf_counter()
        recs = run_http_benchmark_procs(base, "fake", trace, args.client_procs, timeout_s=300)
        s = summarize(recs)
        u1, s1 = cpu_times()
        toks = max(s.output_tokens, 1)
        print(f"run {i + 1}: delivered {s.throughput_tok_s:8,.0f} tok/s  ({100 * s.throughput_tok_s / offered:.0f}% of offered)  "
              f"TTFT p50/p99 {s.ttft_ms.p50:.0f}/{s.ttft_ms.p99:.0f} ms  TPOT p50/p99 {s.tpot_ms.p50:.2f}/{s.tpot_ms.p99:.2f} ms  "
              f"ok {s.completed}/{s.num_requests}  wall {time.perf_counter() - t0:.1f}s  "
              f"API cpu/token: user {1e6 * (u1 - u0) / toks:.0f} us, sys {1e6 * (s1 - s0) / toks:.0f} us")
    m = client.metrics() if client.is_running else {}
    if m:
        print(f"server: generated {m.get('generated_tokens_total')} tokens over {m.get('steps_total')} steps")
    server.should_exit = True
    time.sleep(1.5)
    if prof is not None:
        import pstats

        prof.dump_stats(args.cprofile)
        st = pstats.Stats(prof)
        st.sort_stats("tottime").print_stats(30)
    return 0


def _proc_cpu_s(pids: list[int]) -> float | None:
    """utime + stime of these processes from /proc (Linux); None elsewhere."""
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    total = 0.0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            total += (int(parts[11]) + int(parts[12])) / tick
        except OSError:
            return None
    return total


def run_multi(args: argparse.Namespace, espec: EngineSpec, port: int) -> int:
    srv = MultiServer(espec, ByteTokenizer(), "fake", "127.0.0.1", port, args.api_workers,
                      log_level="warning")
    srv.start()
    base = f"http://127.0.0.1:{port}"
    try:
        if not asyncio.run(wait_for_health(base, 120)):
            print("server did not start", file=sys.stderr)
            return 1
        offered = args.n * 1e3 / args.step_ms
        print(f"fake core: {args.n} requests x 1 token / {args.step_ms} ms = {offered:,.0f} tok/s offered; "
              f"{args.max_tokens} tokens each; API workers {args.api_workers}; client procs {args.client_procs}")
        pids = [p.pid for p in srv.workers]
        for i in range(args.repeats):
            trace = generate_trace(args.n, seed=i, request_rate=None, max_prompt_len=64,
                                   max_output_len=args.max_tokens, vocab_size=256)
            for r in trace:
                r.output_len = args.max_tokens
            c0 = _proc_cpu_s(pids)
            t0 = time.perf_counter()
            recs = run_http_benchmark_procs(base, "fake", trace, args.client_procs, timeout_s=300)
            s = summarize(recs)
            c1 = _proc_cpu_s(pids)
            cpu = (f"API cpu/token {1e6 * (c1 - c0) / max(s.output_tokens, 1):.0f} us (all workers)"
                   if c0 is not None and c1 is not None else "API cpu/token n/a")
            print(f"run {i + 1}: delivered {s.throughput_tok_s:8,.0f} tok/s  ({100 * s.throughput_tok_s / offered:.0f}% of offered)  "
                  f"TTFT p50/p99 {s.ttft_ms.p50:.0f}/{s.ttft_ms.p99:.0f} ms  TPOT p50/p99 {s.tpot_ms.p50:.2f}/{s.tpot_ms.p99:.2f} ms  "
                  f"ok {s.completed}/{s.num_requests}  wall {time.perf_counter() - t0:.1f}s  {cpu}")
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
