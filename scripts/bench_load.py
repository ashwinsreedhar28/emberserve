"""Weight loading, the reference path against the streaming loader (model/fastload.py).

    python scripts/bench_load.py --model models/Qwen2.5-7B-Instruct --device cuda --dtype float16
    python scripts/bench_load.py --model models/Qwen2.5-0.5B-Instruct --device cpu --dtype float16   # a Mac

Alternates the two loaders `--repeats` times each (reference first, so the first reference
run is the only one that can see a cold page cache) and prints seconds and GB/s per run;
`--threads` / `--buffer-mb` sweep the streaming loader's knobs. Each run builds the model
fresh on the device (outside the timed region) and loads into it; on CUDA the timing ends
after a synchronize.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emberserve.config import EngineConfig, ModelConfig  # noqa: E402
from emberserve.model.weights import build_model, load_hf_weights  # noqa: E402


def fresh(model_dir: str, device: str, dtype: torch.dtype):
    cfg = ModelConfig.from_hf_dir(model_dir)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            return build_model(cfg)
    finally:
        torch.set_default_dtype(prev)


def one(model_dir: str, device: str, dtype: torch.dtype, loader: str, threads: int | None,
        buffer_mb: float) -> tuple[float, int]:
    model = fresh(model_dir, device, dtype)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    nbytes = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    t0 = time.perf_counter()
    if loader == "stream":
        from emberserve.model.fastload import stream_weights

        stream_weights(model, model_dir, device, threads=threads, buffer_mb=buffer_mb)
    else:
        os.environ["EMBERSERVE_LOADER"] = "safetensors"
        try:
            load_hf_weights(model, model_dir, dtype=dtype, device=device)
        finally:
            os.environ.pop("EMBERSERVE_LOADER", None)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return dt, nbytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--threads", default="", help="comma list to sweep, e.g. 2,4,8 (default: auto)")
    ap.add_argument("--buffer-mb", default="64", help="comma list to sweep")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()
    dtype = EngineConfig.dtype_from_str(args.dtype)
    threads = [int(x) for x in args.threads.split(",") if x] or [None]
    bufs = [float(x) for x in args.buffer_mb.split(",") if x]
    print(f"{args.model} -> {args.device} {args.dtype}, {os.cpu_count()} CPUs")
    for r in range(args.repeats):
        if not args.skip_reference:
            dt, nb = one(args.model, args.device, dtype, "reference", None, 0)
            print(f"run {r + 1} reference            {dt:7.2f} s  {nb / dt / 1e9:6.2f} GB/s")
        for th in threads:
            for bm in bufs:
                dt, nb = one(args.model, args.device, dtype, "stream", th, bm)
                print(f"run {r + 1} stream t={th or 'auto'!s:>4} buf={bm:g}MB {dt:7.2f} s  {nb / dt / 1e9:6.2f} GB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
