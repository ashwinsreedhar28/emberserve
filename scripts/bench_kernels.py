"""Decode-attention kernel micro-benchmark: paged_torch vs paged_flash vs paged_triton.

    python scripts/bench_kernels.py [--batches 1,8,32,128] [--ctxs 128,512,2048] \
        [--out results/kernels.json] [--iters 20]

Times ONE decode attention call per layer (no model, no cache write, no sampling) at the
real Qwen2.5-0.5B geometry (H=14, Hkv=2, D=64, fp16) over B x ctx, block_size 256 for all
three backends plus an extra paged_triton row at block 16 (the configuration the kernel
exists for). Every sequence has exactly `ctx` tokens and the block tables are scrambled.

Reported per cell: median ms/call over `--iters` timed launches (CUDA events, after
warmup) and the effective K/V bandwidth `B * ctx * Hkv * D * 2 * elem_size / ms` in GB/s,
i.e. the bytes a kernel MUST read from HBM to see every key and value once. A decode
kernel is memory-bound, so this is the number to compare against the card's peak
(RTX 4090: ~1008 GB/s). paged_torch gathers into a padded temporary first, so its
"bandwidth" counts only the useful bytes and is far below what it really moves.

Prints a markdown table and writes `results/kernels.json`. Exits 0 without CUDA.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedserve.attn.base import AttnMetadata  # noqa: E402
from pagedserve.attn.paged_torch import (PagedTorchAttentionBackend,  # noqa: E402
                                         build_block_tables_tensor, build_slot_mapping)
from pagedserve.config import ModelConfig  # noqa: E402
from pagedserve.kv.block_manager import BlockManager  # noqa: E402
from pagedserve.kv.cache import PagedKVCache  # noqa: E402

H, HKV, D = 14, 2, 64
DTYPE = torch.float16
DEV = "cuda"
CFG = ModelConfig.tiny(num_hidden_layers=1, num_attention_heads=H, num_key_value_heads=HKV,
                       hidden_size=H * D)
# (name, block_size)
ROWS = [("paged_torch", 256), ("paged_flash", 256), ("paged_triton", 256), ("paged_triton", 16)]


def kv_bytes(batch: int, ctx: int) -> int:
    return batch * ctx * HKV * D * 2 * torch.tensor([], dtype=DTYPE).element_size()


def make_case(batch: int, ctx: int, block: int, seed: int = 0):
    """A decode step where every sequence holds `ctx` tokens, on scrambled block tables."""
    gen = torch.Generator().manual_seed(seed)
    per_seq = -(-ctx // block)
    num_blocks = batch * per_seq + 1
    bm = BlockManager(num_blocks, block)
    cache = PagedKVCache(CFG, num_blocks, block, device=DEV, dtype=DTYPE)
    cache.k_cache[0].normal_()
    cache.v_cache[0].normal_()
    for sid in range(num_blocks):
        bm.allocate(1000 + sid, block)
    for sid in torch.randperm(num_blocks, generator=gen).tolist():
        bm.free(1000 + sid)
    seqs = list(range(batch))
    for sid in seqs:
        bm.allocate(sid, ctx)
    starts = [ctx - 1] * batch
    meta = AttnMetadata(
        is_prefill=False, seq_ids=seqs, query_lens=[1] * batch, context_lens=[ctx] * batch,
        positions=torch.tensor(starts, device=DEV),
        slot_mapping=build_slot_mapping(bm, seqs, starts, 1, DEV),
        block_tables=build_block_tables_tensor([bm.get_block_table(s) for s in seqs], DEV),
        block_size=block)
    q = torch.randn(batch, H, D, generator=gen).to(DEV, DTYPE)
    return cache, meta, q


def make_backend(name: str, cache: PagedKVCache, splits: int | None = None,
                 variant: str | None = None):
    if name == "paged_torch":
        return PagedTorchAttentionBackend(CFG, cache)
    if name == "paged_flash":
        from pagedserve.attn.paged_flash import PagedFlashAttentionBackend
        return PagedFlashAttentionBackend(CFG, cache)
    if name == "paged_triton":
        from pagedserve.attn.paged_triton import PagedTritonAttentionBackend
        return PagedTritonAttentionBackend(CFG, cache, num_splits=splits, variant=variant)
    raise ValueError(name)


def time_decode(backend, q: torch.Tensor, meta: AttnMetadata, iters: int, warmup: int) -> float:
    """Median ms of `backend._decode(0, q, meta)` measured with CUDA events."""
    from pagedserve.attn.paged_flash import block_tables_nonneg, context_lens_tensor
    block_tables_nonneg(meta)  # prepare the cached device tensors outside the timed region
    context_lens_tensor(meta, q.device)
    with torch.inference_mode():
        for _ in range(warmup):
            backend._decode(0, q, meta)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            backend._decode(0, q, meta)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    return statistics.median(times)


def check_against_torch(backend, q, meta, cache) -> float:
    ref = PagedTorchAttentionBackend(CFG, cache)._decode(0, q, meta)
    out = backend._decode(0, q, meta)
    torch.cuda.synchronize()
    return (out.float() - ref.float()).abs().max().item()


def markdown_table(results: list[dict], batches: list[int], ctxs: list[int]) -> str:
    """Rows: backend/block x ctx; columns: batch. Cell: `ms / GB/s`."""
    lines = ["| backend | block | ctx | " + " | ".join(f"B={b}" for b in batches) + " |",
             "|---|---|---|" + "---|" * len(batches)]
    for name, block in ROWS:
        for c in ctxs:
            cells = []
            for b in batches:
                r = next((r for r in results if r["backend"] == name and r["block_size"] == block
                          and r["batch"] == b and r["ctx"] == c), None)
                cells.append("n/a" if r is None else f"{r['ms']:.3f} ms / {r['gbps']:.0f} GB/s")
            lines.append(f"| {name} | {block} | {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ratio_table(results: list[dict], batches: list[int], ctxs: list[int]) -> str:
    """triton_ms / flash_ms per (block, ctx, B): 1.0 = parity, >1 = Triton slower."""
    by = {(r["backend"], r["block_size"], r["batch"], r["ctx"]): r["ms"] for r in results}
    if not any(k[0] == "paged_flash" for k in by):
        return ""
    lines = ["", "triton / flash time ratio (lower is better; 1.0 = parity)", "",
             "| triton block | ctx | " + " | ".join(f"B={b}" for b in batches) + " |",
             "|---|---|" + "---|" * len(batches)]
    for block in sorted({k[1] for k in by if k[0] == "paged_triton"}):
        for c in ctxs:
            cells = []
            for b in batches:
                t, f = by.get(("paged_triton", block, b, c)), by.get(("paged_flash", 256, b, c))
                cells.append(f"{t / f:.2f}x" if t and f else "-")
            lines.append(f"| {block} | {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--ctxs", default="128,512,2048")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="results/kernels.json")
    ap.add_argument("--splits", type=int, default=None,
                    help="force the Triton split-K factor (default: shape heuristic)")
    ap.add_argument("--variant", choices=["sum", "dot"], default=None,
                    help="Triton kernel variant (default: PAGEDSERVE_TRITON_VARIANT or sum)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA: skipping")
        return 0
    batches = [int(x) for x in args.batches.split(",")]
    ctxs = [int(x) for x in args.ctxs.split(",")]

    available = {"paged_torch": True}
    from pagedserve.attn import paged_flash, paged_triton
    available["paged_flash"] = paged_flash.is_available()
    available["paged_triton"] = paged_triton.is_available()
    for name, ok in available.items():
        if not ok:
            print(f"{name}: not available on this machine, skipped", file=sys.stderr)

    results: list[dict] = []
    for b in batches:
        for c in ctxs:
            for name, block in ROWS:
                if not available[name]:
                    continue
                cache, meta, q = make_case(b, c, block, seed=b * 31 + c)
                backend = make_backend(name, cache, args.splits, args.variant)
                max_err = 0.0 if name == "paged_torch" else check_against_torch(backend, q, meta, cache)
                ms = time_decode(backend, q, meta, args.iters, args.warmup)
                gbps = kv_bytes(b, c) / (ms * 1e-3) / 1e9
                results.append({"backend": name, "block_size": block, "batch": b, "ctx": c,
                                "ms": ms, "gbps": gbps, "kv_bytes": kv_bytes(b, c),
                                "max_abs_err_vs_paged_torch": max_err,
                                "splits": args.splits, "variant": args.variant})
                print(f"  {name:12s} block={block:3d} B={b:3d} ctx={c:4d}: {ms:8.3f} ms "
                      f"{gbps:7.0f} GB/s  err={max_err:.2e}", file=sys.stderr)
                del cache, backend, q, meta
                torch.cuda.empty_cache()

    table = markdown_table(results, batches, ctxs)
    print(f"\ndecode attention, H={H} Hkv={HKV} D={D} {DTYPE}, median of {args.iters} "
          f"(K+V bytes = B*ctx*Hkv*D*2*2)"
          f"{'' if args.variant is None else f', triton variant={args.variant}'}"
          f"{'' if args.splits is None else f', splits={args.splits}'}\n")
    print(table)
    print(ratio_table(results, batches, ctxs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "python": platform.python_version(),
        "geometry": {"H": H, "Hkv": HKV, "D": D, "dtype": str(DTYPE)},
        "iters": args.iters, "warmup": args.warmup,
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
