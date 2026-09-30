"""Runpod Serverless worker entry point: start `pagedserve serve`, wait for it, then serve jobs.

Environment (all optional): MODEL_DIR (a snapshot baked into the image; default /models/model),
MODEL_REPO (download at cold start instead), DTYPE (float16), ATTN_BACKEND (paged_flash),
BLOCK_SIZE (256), MAX_MODEL_LEN (4096), MAX_NUM_SEQS (256), CUDA_GRAPHS (1), PREFIX_CACHING (0),
QUANTIZATION (unset | int8), TENSOR_PARALLEL_SIZE (1), SERVED_MODEL_NAME (the model dir),
EXTRA_SERVE_ARGS (appended verbatim), PAGEDSERVE_PORT (8000), MAX_CONCURRENCY (jobs per
worker, 64), STARTUP_TIMEOUT (600 s), STREAM_FLUSH_MS (100; see handler.py). Chunked prefill,
piecewise graphs, async scheduling and the engine-core process follow the CLI's CUDA defaults
(README, "Run it"). RUNPOD_LB=1 turns the image into a load-balancing worker: `pagedserve serve`
runs on 0.0.0.0:$PORT with no job wrapper (set PORT and PORT_HEALTH on the endpoint).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timeline  # noqa: E402

timeline.record_process_starts()

import httpx  # noqa: E402

from handler import make_handler  # noqa: E402

PORT = int(os.environ.get("PAGEDSERVE_PORT", "8000"))  # the proxy's local server (queue mode)
BASE_URL = f"http://127.0.0.1:{PORT}"


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


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


def serve_command(model_dir: str, host: str = "127.0.0.1", port: int = PORT) -> list[str]:
    cmd = [sys.executable, "-m", "pagedserve.cli", "serve", "--model", model_dir,
           "--device", "cuda", "--dtype", env("DTYPE", "float16"),
           "--attn-backend", env("ATTN_BACKEND", "paged_flash"),
           "--block-size", env("BLOCK_SIZE", "256"),
           "--max-model-len", env("MAX_MODEL_LEN", "4096"),
           "--max-num-seqs", env("MAX_NUM_SEQS", "256"),
           "--host", host, "--port", str(port),
           "--served-model-name", env("SERVED_MODEL_NAME", model_dir)]
    if env("CUDA_GRAPHS", "1") == "1":
        cmd.append("--enable-cuda-graphs")
    if env("PREFIX_CACHING", "0") == "1":
        cmd.append("--enable-prefix-caching")
    if os.environ.get("QUANTIZATION"):
        cmd += ["--quantization", os.environ["QUANTIZATION"]]
    if env("TENSOR_PARALLEL_SIZE", "1") != "1":
        cmd += ["--tensor-parallel-size", env("TENSOR_PARALLEL_SIZE", "1")]
    cmd += shlex.split(env("EXTRA_SERVE_ARGS", ""))
    return cmd


def spawn_serve(cmd: list[str]) -> subprocess.Popen:
    """Start the server with its output relayed line by line to ours (the worker log), noting
    when the engine core prints its `[boot]` line."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    timeline.mark("serve_spawned")

    def relay() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            if "[boot]" in line:
                timeline.mark("engine_boot")
                timeline.note("engine_boot", line.strip())
            sys.stdout.write(line)
            sys.stdout.flush()

    threading.Thread(target=relay, name="serve-log", daemon=True).start()
    return proc


def _import_sdk(box: dict) -> None:
    import runpod

    box["runpod"] = runpod


def wait_for_server(proc: subprocess.Popen, timeout_s: float, base_url: str = BASE_URL) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"pagedserve serve exited with code {proc.returncode} during startup")
        try:
            if httpx.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"pagedserve serve did not become healthy within {timeout_s:.0f} s")


def main() -> None:
    timeline.mark("worker_main")
    model_dir = resolve_model_dir()
    if env("RUNPOD_LB", "0") == "1":
        # Load-balancing endpoint: Runpod routes HTTP straight to this container's $PORT and
        # polls /ping, so the server itself is the worker (no job wrapper, no SDK; streaming
        # is the server's own SSE). Set PORT and PORT_HEALTH on the endpoint to the same value.
        # This process stays as a supervisor so the log carries the start-to-healthy time.
        port = int(env("PORT", "8000"))
        cmd = serve_command(model_dir, host="0.0.0.0", port=port)
        print("[worker] load-balancing mode: " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        t0 = time.monotonic()
        proc = spawn_serve(cmd)
        wait_for_server(proc, float(env("STARTUP_TIMEOUT", "600")), f"http://127.0.0.1:{port}")
        timeline.mark("serve_healthy")
        print(f"[worker] pagedserve up in {time.monotonic() - t0:.1f} s (load-balancing mode)", flush=True)
        sys.exit(proc.wait())
    cmd = serve_command(model_dir)
    print("[worker] starting: " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    t0 = time.monotonic()
    proc = spawn_serve(cmd)
    # the SDK import (~1 s) overlaps the engine's boot instead of following it
    sdk: dict = {}
    importer = threading.Thread(target=_import_sdk, args=(sdk,), name="sdk-import", daemon=True)
    importer.start()
    wait_for_server(proc, float(env("STARTUP_TIMEOUT", "600")))
    timeline.mark("serve_healthy")
    print(f"[worker] pagedserve up in {time.monotonic() - t0:.1f} s", flush=True)
    importer.join()
    runpod = sdk["runpod"]
    client = httpx.AsyncClient(base_url=BASE_URL)
    max_concurrency = int(env("MAX_CONCURRENCY", "64"))
    timeline.mark("sdk_ready")
    print("[worker] timeline " + " ".join(f"{k}={v:.3f}" for k, v in timeline.snapshot()["marks"].items()),
          flush=True)
    runpod.serverless.start({
        "handler": make_handler(client, env("SERVED_MODEL_NAME", model_dir),
                                alive=lambda: proc.poll() is None),
        "return_aggregate_stream": True,
        "concurrency_modifier": lambda current: max_concurrency,
    })


if __name__ == "__main__":
    main()
