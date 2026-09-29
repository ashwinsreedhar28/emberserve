"""Where a mixed step's GPU time goes, kernel by kernel, against a decode-only step (GPU).

    python scripts/profile_mixed_step.py --model models/Qwen2.5-7B-Instruct --decode 32 --ctx 600 --chunk 300

B sequences decode at context C; then, `--steps` times, one fresh Q-token prompt joins and
its whole prefill rides in one chunked-prefill step next to the B decode rows (the request
is aborted right after, so the batch shape repeats). torch.profiler records every kernel of
those mixed steps and of as many decode-only steps at batch B; the output is the per-kernel
difference, grouped (GEMM / flash-attn / our Triton / copies & elementwise / other), with
the achieved TFLOP/s of the mixed step's GEMMs. Synchronous scheduling, so each step's
kernels are its own.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedserve.config import EngineConfig  # noqa: E402
from pagedserve.engine import LLMEngine  # noqa: E402
from pagedserve.sched.request import SamplingParams  # noqa: E402


def group(name: str) -> str:
    n = name.lower()
    if "flash" in n:
        return "flash_attn"
    if any(s in n for s in ("gemm", "cutlass", "xmma", "cublas", "sm80_", "ampere_", "gemv", "splitk")):
        return "gemm"
    if any(s in n for s in ("triton", "_kernel_0d", "fused_")):
        return "triton"
    if any(s in n for s in ("copy", "memcpy", "memset", "index", "scatter", "gather", "cat",
                            "elementwise", "vectorized", "fill", "unrolled")):
        return "copy_elementwise"
    return "other"


def profile(engine: LLMEngine, steps: int, body) -> dict[str, tuple[float, float]]:
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile as tprof

    torch.cuda.synchronize()
    with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(steps):
            body(i)
        torch.cuda.synchronize()
    out: dict[str, tuple[float, float]] = {}
    for ev in prof.key_averages():
        if ev.device_type != DeviceType.CUDA:
            continue
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0.0)
        out[ev.key] = (t / steps, ev.count / steps)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    args = ap.parse_args()
    ecfg = EngineConfig(device="cuda", dtype=torch.float16, attn_backend="paged_flash", block_size=256,
                        enable_cuda_graphs=True, enable_chunked_prefill=True,
                        max_num_batched_tokens=args.max_num_batched_tokens, max_model_len=4096,
                        async_scheduling=False)
    engine = LLMEngine.from_pretrained(args.model, ecfg)
    engine.keep_stats = False
    vocab = engine.model_config.vocab_size
    rng = random.Random(0)
    B, C, Q = args.decode, args.ctx, args.chunk
    sp = SamplingParams.greedy(max_tokens=4 * args.steps + 16, ignore_eos=True)
    for i in range(B):
        engine.add_request(f"d{i}", [rng.randrange(2, vocab) for _ in range(C)], sp)
    while engine.scheduler.num_waiting:
        engine.step()
    for _ in range(3):
        engine.step()

    def decode_body(_i: int) -> None:
        outs = engine.step()
        assert len(outs) == B

    def mixed_body(i: int) -> None:
        engine.add_request(f"m{i}", [rng.randrange(2, vocab) for _ in range(Q)],
                           SamplingParams.greedy(max_tokens=8, ignore_eos=True))
        engine.step()
        engine.abort_request(f"m{i}")

    mixed_body(-1)  # warm up the mixed path
    dec = profile(engine, args.steps, decode_body)
    mix = profile(engine, args.steps, mixed_body)
    names = set(dec) | set(mix)
    rows = []
    for n in names:
        d, dc = dec.get(n, (0.0, 0.0))
        m, mc = mix.get(n, (0.0, 0.0))
        rows.append((n, d, m, m - d, mc))
    rows.sort(key=lambda r: -r[3])
    dt, mt = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"batch {B} decoding at ctx {C}; mixed = same + one {Q}-token prompt in the step")
    print(f"GPU kernel time per step: decode-only {dt / 1e3:.2f} ms, mixed {mt / 1e3:.2f} ms, "
          f"excess {(mt - dt) / 1e3:.2f} ms")
    by: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for n, d, m, _, _ in rows:
        g = group(n)
        by[g][0] += d
        by[g][1] += m
    print(f"{'group':<18}{'decode ms':>10}{'mixed ms':>10}{'excess':>9}")
    for g, (d, m) in sorted(by.items(), key=lambda kv: -(kv[1][1] - kv[1][0])):
        print(f"{g:<18}{d / 1e3:>10.2f}{m / 1e3:>10.2f}{(m - d) / 1e3:>9.2f}")
    # GEMM efficiency of the mixed step: the linear layers see B + Q rows.
    cfg = engine.model_config
    h, i_, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    kv = cfg.num_key_value_heads * cfg.head_dim
    per_tok = 2 * L * (h * (h + 2 * kv) + h * h + 3 * h * i_)
    gemm_ms = by["gemm"][1] / 1e3
    if gemm_ms:
        tf = per_tok * (B + Q) / (gemm_ms * 1e-3) / 1e12
        print(f"mixed-step GEMMs: {(B + Q)} rows x {per_tok / 1e9:.1f} GFLOP/row in {gemm_ms:.2f} ms "
              f"= {tf:.0f} TFLOP/s (A100 fp16 peak 312) [lm_head excluded from the FLOPs]")
    print(f"\n{'decode us':>10}{'mixed us':>10}{'excess':>9}{'calls':>7}  kernel (top {args.top} by excess)")
    for n, d, m, x, mc in rows[:args.top]:
        nm = n if len(n) <= 80 else n[:77] + "..."
        print(f"{d:>10.1f}{m:>10.1f}{x:>9.1f}{mc:>7.0f}  {nm}")


if __name__ == "__main__":
    main()
