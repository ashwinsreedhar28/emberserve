"""Where does a decode step's wall time go? Per-phase timing at fixed batch sizes.

    python scripts/profile_step.py --model models/Qwen2.5-0.5B-Instruct --device cuda --dtype float16 \
        --attn-backend paged_flash --block-size 256 --enable-cuda-graphs --batches 1,8,32,128,200
    python scripts/profile_step.py --tiny --batches 1,8,32          # CPU smoke run, random 2-layer model
    ... --kernels 1,128 --top 30     # plus GPU time per kernel at those batch sizes (torch.profiler)

For each batch size N: admit N requests with `--prompt-len` random tokens, run prefill until
all N are decoding, then time `--steps` decode steps split into schedule / build_inputs /
forward / sample / postprocess. `forward` includes a device sync so GPU time lands there and
not in the first phase that happens to touch a result. Prints ms per phase (mean over steps),
the total, and the per-sequence slope (total at N minus total at the smallest N, divided by
the difference in N), which is the number the vLLM gap analysis needs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedserve.config import EngineConfig, ModelConfig  # noqa: E402
from pagedserve.engine import LLMEngine  # noqa: E402
from pagedserve.sched.request import SamplingParams  # noqa: E402

PHASES = ("schedule", "build_inputs", "forward", "sample", "postprocess")


def build_engine(args: argparse.Namespace) -> LLMEngine:
    ecfg = EngineConfig(device=args.device, dtype=EngineConfig.dtype_from_str(args.dtype),
                        attn_backend=args.attn_backend, block_size=args.block_size,
                        enable_cuda_graphs=args.enable_cuda_graphs,
                        enable_prefix_caching=args.enable_prefix_caching,
                        max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
                        max_num_batched_tokens=args.max_num_batched_tokens,
                        num_gpu_blocks=args.num_blocks,
                        quantization=getattr(args, "quantization", None))
    if args.tiny:
        from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic

        mcfg = ModelConfig.tiny()
        model = Qwen2ForCausalLM(mcfg)
        reset_parameters_deterministic(model, 0)
        if ecfg.num_gpu_blocks is None:
            ecfg.num_gpu_blocks = 2048
        return LLMEngine(model, mcfg, ecfg, tokenizer=None)
    if not args.model:
        raise SystemExit("--model DIR is required unless --tiny")
    return LLMEngine.from_pretrained(args.model, ecfg)


class PhaseTimer:
    """Wraps the engine's step phases with timers; `forward` syncs the device.

    Times are accumulated per step (a phase may be entered more than once in a step: the
    eager path calls the model and then `compute_logits`, both counted as `forward`)."""

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self.steps: list[dict[str, float]] = []
        self._cur: dict[str, float] | None = None
        self.enabled = False
        self._sync = (torch.cuda.synchronize if engine.device.type == "cuda"
                      else (torch.mps.synchronize if engine.device.type == "mps" else None))
        self._install()

    def _wrap(self, obj: object, name: str, phase: str, sync: bool = False) -> None:
        fn = getattr(obj, name)

        def wrapped(*a, **kw):
            if not self.enabled or self._cur is None:
                return fn(*a, **kw)
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            if sync and self._sync is not None:
                self._sync()
            self._cur[phase] = self._cur.get(phase, 0.0) + (time.perf_counter() - t0) * 1e3
            return out

        setattr(obj, name, wrapped)

    def _install(self) -> None:
        e = self.engine
        step = e.step

        def timed_step(*a, **kw):
            if not self.enabled:
                return step(*a, **kw)
            self._cur = {}
            t0 = time.perf_counter()
            out = step(*a, **kw)
            self._cur["total"] = (time.perf_counter() - t0) * 1e3
            self.steps.append(self._cur)
            self._cur = None
            return out

        e.step = timed_step
        self._wrap(e.scheduler, "schedule", "schedule")
        self._wrap(e, "_build_inputs", "build_inputs")
        if e.graph_runner is not None:
            self._wrap(e.graph_runner, "run", "forward", sync=True)
        e.model.forward = self._timed_forward(e.model.forward)   # eager path (nn.Module.__call__ -> self.forward)
        self._wrap(e.model, "compute_logits", "forward", sync=True)
        self._wrap(e.sampler, "sample", "sample")
        self._wrap(e, "_postprocess", "postprocess")

    def _timed_forward(self, fn):
        def wrapped(*a, **kw):
            if not self.enabled or self._cur is None:
                return fn(*a, **kw)
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            if self._sync is not None:
                self._sync()
            self._cur["forward"] = self._cur.get("forward", 0.0) + (time.perf_counter() - t0) * 1e3
            return out
        return wrapped

    def reset(self) -> None:
        self.steps.clear()

    def means(self) -> dict[str, float]:
        n = len(self.steps) or 1
        keys = (*PHASES, "total")
        return {k: sum(st.get(k, 0.0) for st in self.steps) / n for k in keys}


def run_batch(engine: LLMEngine, timer: PhaseTimer, n: int, prompt_len: int, steps: int,
              vocab: int, seed: int) -> dict:
    rng = random.Random(seed)
    engine.reset()
    sp = SamplingParams.greedy(max_tokens=steps + 4, ignore_eos=True)
    for i in range(n):
        ids = [rng.randrange(1, vocab) for _ in range(prompt_len)]
        engine.add_request(f"p{n}-{i}", ids, sp)
    # Prefill (possibly several steps under the token budget) until nobody is waiting.
    timer.enabled = False
    while engine.scheduler.num_waiting:
        engine.step()
    # Warm up two decode steps, then measure.
    engine.step()
    engine.step()
    timer.enabled = True
    timer.reset()
    for _ in range(steps):
        outs = engine.step()
        assert len(outs) == n, f"batch drained: {len(outs)} != {n}"
    timer.enabled = False
    m = timer.means()
    m["unaccounted"] = m["total"] - sum(m[p] for p in PHASES)
    engine.reset()
    return m


def profile_kernels(engine: LLMEngine, n: int, prompt_len: int, steps: int, vocab: int,
                    seed: int, top: int) -> list[dict]:
    """GPU time per kernel over `steps` decode steps at batch `n` (torch.profiler / CUPTI;
    kernels replayed inside a CUDA graph are recorded too). Returns rows sorted by time."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    rng = random.Random(seed)
    engine.reset()
    sp = SamplingParams.greedy(max_tokens=steps + 4, ignore_eos=True)
    for i in range(n):
        engine.add_request(f"k{n}-{i}", [rng.randrange(1, vocab) for _ in range(prompt_len)], sp)
    while engine.scheduler.num_waiting:
        engine.step()
    engine.step()
    engine.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            engine.step()
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        if ev.device_type != DeviceType.CUDA:
            continue
        total = getattr(ev, "self_device_time_total", None)
        if total is None:
            total = getattr(ev, "self_cuda_time_total", 0.0)
        rows.append({"name": ev.key, "us_per_step": total / steps, "calls_per_step": ev.count / steps})
    rows.sort(key=lambda r: -r["us_per_step"])
    engine.reset()
    gpu_total = sum(r["us_per_step"] for r in rows)
    print(f"\nGPU kernels per decode step at N={n} (mean of {steps}): {gpu_total / 1e3:.3f} ms of "
          f"kernel time in {sum(r['calls_per_step'] for r in rows):.0f} launches")
    print(f"{'us/step':>9} {'share':>6} {'calls':>6}  kernel")
    for r in rows[:top]:
        name = r["name"] if len(r["name"]) <= 90 else r["name"][:87] + "..."
        print(f"{r['us_per_step']:>9.1f} {100 * r['us_per_step'] / gpu_total:>5.1f}% {r['calls_per_step']:>6.1f}  {name}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--attn-backend", default="paged_torch")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=None)
    ap.add_argument("--enable-cuda-graphs", action="store_true")
    ap.add_argument("--enable-prefix-caching", action="store_true")
    ap.add_argument("--quantization", default=None, choices=["int8"],
                    help="weight-only int8 (per-channel) for every 2-D projection")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--batches", default="1,8,32,128,200")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--kernels", default=None,
                    help="also list GPU time per kernel at these batch sizes, e.g. 1,128 (CUDA only)")
    ap.add_argument("--top", type=int, default=25, help="kernels to print with --kernels")
    args = ap.parse_args()
    if args.tiny:
        args.prompt_len = min(args.prompt_len, 32)
    batches = [int(b) for b in args.batches.split(",")]
    if max(batches) > args.max_num_seqs:
        raise SystemExit(f"--max-num-seqs {args.max_num_seqs} < largest batch {max(batches)}")

    engine = build_engine(args)
    vocab = engine.model_config.vocab_size
    timer = PhaseTimer(engine)
    rows = []
    hdr = f"{'N':>4} " + " ".join(f"{p:>13}" for p in PHASES) + f" {'unaccounted':>12} {'total':>8} {'ms/seq':>8}"
    print(f"decode step breakdown, ms per step (mean of {args.steps}); backend={args.attn_backend} "
          f"block={args.block_size} graphs={args.enable_cuda_graphs} device={args.device} "
          f"prompt_len={args.prompt_len}")
    print(hdr)
    base = None
    for n in batches:
        m = run_batch(engine, timer, n, args.prompt_len, args.steps, vocab, args.seed)
        if base is None:
            base = (n, m["total"])
            slope = float("nan")
        else:
            slope = (m["total"] - base[1]) / (n - base[0]) if n != base[0] else float("nan")
        m["n"] = n
        m["ms_per_seq"] = slope
        rows.append(m)
        print(f"{n:>4} " + " ".join(f"{m[p]:>13.3f}" for p in PHASES)
              + f" {m['unaccounted']:>12.3f} {m['total']:>8.3f} {slope:>8.3f}")
    kernels = {}
    if args.kernels:
        if engine.device.type != "cuda":
            raise SystemExit("--kernels needs --device cuda")
        for n in (int(b) for b in args.kernels.split(",")):
            kernels[n] = profile_kernels(engine, n, args.prompt_len, min(args.steps, 20), vocab,
                                         args.seed, args.top)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        meta = {"kind": "profile_step", "device": args.device, "dtype": args.dtype,
                "attn_backend": args.attn_backend, "block_size": args.block_size,
                "cuda_graphs": args.enable_cuda_graphs, "prompt_len": args.prompt_len,
                "steps": args.steps, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "rows": rows, "kernels": {str(k): v[:60] for k, v in kernels.items()}}
        Path(args.out).write_text(json.dumps(meta, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
