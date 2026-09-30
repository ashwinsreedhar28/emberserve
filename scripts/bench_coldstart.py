"""Process start to first token, pagedserve against vLLM, on one machine.

    python scripts/bench_coldstart.py --model models/Qwen2.5-7B-Instruct --repeats 3 \\
        --system pagedserve --system vllm --system vllm_tuned --system vllm_eager \\
        --vllm-bin /opt/vllm/bin/vllm --out results/coldstart/local_7b.json

For each system and repeat: start the server as a fresh process (weights already on local
disk, nothing downloaded), then poll `/health` every 50 ms and, the moment it answers, send
one streamed 1-token completion. Recorded per run: seconds from `Popen` to healthy, to the
first streamed byte, and to the end of that request, plus what the server's own log says
about its startup (pagedserve's `[boot]` line; vLLM's "init engine ... took" and
torch.compile lines). The server is stopped before the next run.

The systems (all the same weights, dtype and max-model-len):

  pagedserve   `pagedserve serve --device cuda --attn-backend paged_flash --block-size 256
               --enable-cuda-graphs` (the benchmarked config)
  vllm         `vllm serve` with its defaults: torch.compile + piecewise and full CUDA graphs;
               its compile cache persists across runs in ~/.cache/vllm, so run 1 is a cold
               compile and later runs hit the cache (the realistic warm-image case)
  vllm_tuned   what vLLM's and Runpod's cold-start guides recommend: compile cache (as
               above), `--load-format runai_streamer` when installed, capture sizes trimmed
               to 1-64 with `--max-num-seqs 64`, HF_HUB_OFFLINE=1
  vllm_eager   `--enforce-eager`: no compile, no graphs — vLLM's fastest boot, slower steps

A system's first run is reported separately (cold caches); the claim is made on the rest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedserve.bench.run_vllm_baseline import kill  # noqa: E402


def commands(system: str, args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    m, port = args.model, str(args.port)
    common = ["--host", "127.0.0.1", "--port", port, "--dtype", args.dtype,
              "--max-model-len", str(args.max_model_len)]
    if system == "pagedserve":
        return ([sys.executable, "-m", "pagedserve.cli", "serve", "--model", m, *common,
                 "--device", "cuda", "--attn-backend", "paged_flash", "--block-size", "256",
                 "--enable-cuda-graphs", *shlex.split(args.pagedserve_args)], {})
    v = [args.vllm_bin, "serve", m, "--served-model-name", m, *common]
    if system == "vllm":
        return v, {}
    if system == "vllm_tuned":
        extra = ["--max-num-seqs", "64",
                 "--compilation-config", json.dumps({"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]})]
        if args.runai:
            extra += ["--load-format", "runai_streamer"]
        return v + extra, {"HF_HUB_OFFLINE": "1"}
    if system == "vllm_eager":
        return v + ["--enforce-eager"], {"HF_HUB_OFFLINE": "1"}
    raise ValueError(system)


async def first_token(base: str, model: str, deadline: float,
                      proc: subprocess.Popen | None = None) -> tuple[float, float, float]:
    """Poll /health, then stream a 1-token completion. Perf-counter times: healthy, first
    byte, done."""
    async with httpx.AsyncClient(base_url=base, timeout=httpx.Timeout(30.0, connect=1.0)) as c:
        while True:
            if time.perf_counter() > deadline:
                raise TimeoutError("server did not become healthy")
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"server exited with code {proc.returncode} before it was healthy")
            try:
                if (await c.get("/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.05)
        t_health = time.perf_counter()
        body = {"model": model, "prompt": "The capital of France is", "max_tokens": 1,
                "temperature": 0, "stream": True}
        t_first = None
        async with c.stream("POST", "/v1/completions", json=body) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                if chunk and t_first is None:
                    t_first = time.perf_counter()
        return t_health, t_first or time.perf_counter(), time.perf_counter()


def log_facts(text: str) -> dict:
    facts: dict = {}
    m = re.search(r"\[boot\] (.*)", text)
    if m:
        facts["pagedserve_boot"] = m.group(1).strip()
    m = re.search(r"init engine .*? took ([\d.]+) s(?:econds)?(?: \(compilation: ([\d.]+) s\))?", text)
    if m:
        facts["vllm_init_s"] = float(m.group(1))
        if m.group(2):
            facts["vllm_compile_s"] = float(m.group(2))
    m = re.search(r"torch.compile took ([\d.]+) s", text)
    if m:
        facts["vllm_torch_compile_s"] = float(m.group(1))
    m = re.search(r"Loading weights took ([\d.]+) s", text)
    if m:
        facts["vllm_weights_s"] = float(m.group(1))
    m = re.search(r"Graph capturing finished in ([\d.]+) sec", text)
    if m:
        facts["vllm_graph_capture_s"] = float(m.group(1))
    return facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", action="append", default=[],
                    choices=["pagedserve", "vllm", "vllm_tuned", "vllm_eager"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--vllm-bin", default="vllm")
    ap.add_argument("--pagedserve-args", default="")
    ap.add_argument("--runai", action="store_true", help="vllm_tuned: --load-format runai_streamer")
    ap.add_argument("--timeout-s", type=float, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    systems = args.system or ["pagedserve"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for system in systems:
        for r in range(args.repeats):
            cmd, extra_env = commands(system, args)
            env = {**os.environ, **extra_env}
            bin_dir = os.path.dirname(os.path.abspath(cmd[0]))
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
            log_path = out_path.with_name(f"{out_path.stem}_{system}_{r + 1}.log")
            log = open(log_path, "w")  # noqa: SIM115
            t0 = time.perf_counter()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            rec: dict = {"system": system, "run": r + 1, "cmd": cmd, "env": extra_env}
            try:
                th, tf, td = asyncio.run(first_token(f"http://127.0.0.1:{args.port}", args.model,
                                                     t0 + args.timeout_s, proc))
                rec.update(ok=True, healthy_s=th - t0, first_token_s=tf - t0, done_s=td - t0)
            except Exception as exc:  # noqa: BLE001
                rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            finally:
                kill(proc)
                log.close()
                time.sleep(3)  # let the GPU memory go before the next server
            rec.update(log_facts(log_path.read_text(errors="replace")))
            results.append(rec)
            if rec["ok"]:
                print(f"{system:<11} run {r + 1}: healthy {rec['healthy_s']:6.1f} s  first token "
                      f"{rec['first_token_s']:6.1f} s  | " + ", ".join(
                          f"{k} {v}" for k, v in rec.items() if k.startswith(("pagedserve_boot", "vllm_"))),
                      flush=True)
            else:
                print(f"{system:<11} run {r + 1}: FAILED {rec['error']} (log {log_path})", flush=True)
            out_path.write_text(json.dumps({"model": args.model, "runs": results}, indent=1))
    print("\nfirst token, runs after the first (warm caches):")
    for system in systems:
        xs = [x["first_token_s"] for x in results if x["system"] == system and x["ok"] and x["run"] > 1]
        first = [x["first_token_s"] for x in results if x["system"] == system and x["ok"] and x["run"] == 1]
        if xs:
            print(f"  {system:<11} mean {sum(xs) / len(xs):6.1f} s  (runs {', '.join(f'{v:.1f}' for v in xs)}; "
                  f"first run {first[0]:.1f} s)" if first else f"  {system:<11} mean {sum(xs) / len(xs):6.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
