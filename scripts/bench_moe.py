"""Micro-benchmark for the fused MoE layer at Moonlight's geometry: ms per call and the
effective weight bandwidth, for a few token counts and grouped-GEMM tile configs.

    python scripts/bench_moe.py                       # default configs, M in 1,8,32,128,512
    python scripts/bench_moe.py --configs 64,128,4,4 32,128,4,4 128,64,8,3 --tokens 1,8

At M=1 the layer reads the 6 routed experts' weights (104 MB) and nothing else matters,
so `GB/s` against the A100's ~2 TB/s says how far the tiles are from streaming speed.
The winning config goes into `moe_triton.gemm_config`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedserve.config import MoEConfig  # noqa: E402
from pagedserve.model.moe import DeepseekMoE  # noqa: E402
from pagedserve.model.moe_triton import fused_moe_forward  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", default="1,8,32,128,512")
    ap.add_argument("--configs", nargs="*", default=["default", "64,64,4,3", "64,128,4,4", "32,128,4,4",
                                                     "128,64,4,3", "64,128,8,4", "64,256,4,3"],
                    help="bn,bk,warps,stages per config; 'default' = gemm_config()")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    dtype = getattr(torch, args.dtype)
    cfg = MoEConfig(hidden_size=2048, moe_intermediate_size=1408, n_routed_experts=64,
                    num_experts_per_tok=6, n_shared_experts=0, routed_scaling_factor=2.446)
    moe = DeepseekMoE(cfg)
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0, 0.02)
    moe = moe.to("cuda", dtype).eval()
    e_bytes = (moe.experts_gate_up[0].numel() + moe.experts_down[0].numel()) * moe.experts_gate_up.element_size()
    print(f"expert = {e_bytes / 1e6:.1f} MB; configs bn,bk,warps,stages; ms per fused forward, "
          f"GB/s = bytes of the touched experts' weights / time")
    tokens = [int(t) for t in args.tokens.split(",")]
    print(f"{'config':>14} " + " ".join(f"{'M=' + str(m):>16}" for m in tokens))
    for conf in args.configs:
        if conf == "default":
            os.environ.pop("PAGEDSERVE_MOE_CONFIG", None)
        else:
            os.environ["PAGEDSERVE_MOE_CONFIG"] = conf
        cells = []
        for m in tokens:
            x = torch.randn(m, 2048, device="cuda", dtype=dtype)
            idx, w = moe.gate(x)
            touched = int(idx.unique().numel())
            with torch.no_grad():
                for _ in range(5):
                    fused_moe_forward(x, idx, w, moe.experts_gate_up, moe.experts_down)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    fused_moe_forward(x, idx, w, moe.experts_gate_up, moe.experts_down)
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3 / args.iters
            gbs = touched * e_bytes / (ms * 1e-3) / 1e9
            cells.append(f"{ms:7.3f}ms {gbs:5.0f}GB/s")
        print(f"{conf:>14} " + " ".join(f"{c:>16}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
