"""Mixed-step attention at 7B shapes: our two-call path vs one paged varlen call (GPU only).

    python scripts/bench_mixed_attn.py                      # Qwen2.5-7B heads, block 256
    python scripts/bench_mixed_attn.py --decode 32 --ctx 600 --chunk 270 --iters 100

A mixed step (chunked prefill) holds B decode rows plus a prompt chunk. `paged_flash`
runs it per layer as one `flash_attn_with_kvcache` over the decode rows and a second one
over the chunk rows left-padded to [P, max_q] (`_prefill_kvcache`), with an index_select,
an index_copy and per-chunk slice copies around them. vLLM runs the same step as ONE
varlen call over the paged cache (`flash_attn_varlen_func(..., block_table=...)`, decode
rows being queries of length 1). This times, per layer:

    decode_only   the B decode rows alone (what a decode step pays)
    prefill_only  the chunk alone through the varlen path of a fresh prefill
    ours          `PagedFlashAttentionBackend._prefill_kvcache` on the mixed batch
    varlen_paged  one `flash_attn_varlen_func` with the block table, same batch

and checks that `varlen_paged` matches `ours` to fp16 tolerance. GPU time from CUDA
events over `--iters` calls, host time per call from the wall clock of the launches;
x layers gives the per-step cost.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--decode", type=int, default=32, help="decode rows in the step")
    ap.add_argument("--ctx", type=int, default=600, help="context length of each decode row")
    ap.add_argument("--chunk", type=int, default=270, help="prompt-chunk tokens (fresh prompt)")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("bench_mixed_attn needs a CUDA GPU with flash-attn")
    from flash_attn import flash_attn_varlen_func

    from pagedserve.attn.base import AttnMetadata
    from pagedserve.attn.paged_flash import PagedFlashAttentionBackend
    from pagedserve.config import ModelConfig
    from pagedserve.kv.cache import PagedKVCache

    dev = torch.device("cuda")
    dt = torch.float16
    H, Hkv, D, bs = args.heads, args.kv_heads, args.head_dim, args.block_size
    cfg = ModelConfig(hidden_size=H * D, num_attention_heads=H, num_key_value_heads=Hkv,
                      num_hidden_layers=1)
    B, C, Q = args.decode, args.ctx, args.chunk
    blocks_per = [math.ceil(C / bs)] * B + [math.ceil(Q / bs)]
    num_blocks = sum(blocks_per) + 1
    cache = PagedKVCache(cfg, num_blocks, bs, dev, dt)
    backend = PagedFlashAttentionBackend(cfg, cache)
    torch.manual_seed(0)
    cache.k_cache[0].normal_()
    cache.v_cache[0].normal_()
    # sequences: B decode rows (query 1, context C) then the chunk (query Q, context Q)
    tables, nxt = [], 0
    for n in blocks_per:
        tables.append(list(range(nxt, nxt + n)))
        nxt += n
    maxb = max(blocks_per)
    bt = torch.tensor([t + [-1] * (maxb - len(t)) for t in tables], dtype=torch.int32, device=dev)
    query_lens = [1] * B + [Q]
    context_lens = [C] * B + [Q]
    ntok = B + Q
    slots = [tables[i][(C - 1) // bs] * bs + (C - 1) % bs for i in range(B)]
    slots += [tables[B][p // bs] * bs + p % bs for p in range(Q)]
    meta = AttnMetadata(is_prefill=True, seq_ids=list(range(B + 1)), query_lens=query_lens,
                        context_lens=context_lens,
                        positions=torch.zeros(ntok, dtype=torch.long, device=dev),
                        slot_mapping=torch.tensor(slots, dtype=torch.long, device=dev),
                        block_tables=bt, block_size=bs,
                        num_cached_tokens=[C - 1] * B + [0])
    q = torch.randn(ntok, H, D, dtype=dt, device=dev)
    k = torch.randn(ntok, Hkv, D, dtype=dt, device=dev)
    v = torch.randn(ntok, Hkv, D, dtype=dt, device=dev)
    ctx_t = torch.tensor(context_lens, dtype=torch.int32, device=dev)
    cu_q = torch.tensor([0] + list(torch.tensor(query_lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0] + list(torch.tensor(context_lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    bt_nn = bt.clamp_min(0)
    scale = 1.0 / math.sqrt(D)

    def ours():
        meta.mixed_plan = None  # rebuilt once per step in the engine; here per call
        return backend.forward(0, q, k, v, meta)

    def varlen_paged():
        cache.write(0, k, v, meta.slot_mapping)
        return flash_attn_varlen_func(q, cache.k_cache[0], cache.v_cache[0], cu_q, cu_k, Q, max(C, Q),
                                      softmax_scale=scale, causal=True, block_table=bt_nn)

    dec_meta = AttnMetadata(is_prefill=False, seq_ids=list(range(B)), query_lens=[1] * B,
                            context_lens=[C] * B, positions=torch.zeros(B, dtype=torch.long, device=dev),
                            slot_mapping=meta.slot_mapping[:B], block_tables=bt[:B], block_size=bs)
    dec_meta.context_lens_t = ctx_t[:B]
    pre_meta = AttnMetadata(is_prefill=True, seq_ids=[0], query_lens=[Q], context_lens=[Q],
                            positions=torch.zeros(Q, dtype=torch.long, device=dev),
                            slot_mapping=meta.slot_mapping[B:], block_tables=bt[B:], block_size=bs,
                            num_cached_tokens=[0])

    def decode_only():
        return backend.forward(0, q[:B], k[:B], v[:B], dec_meta)

    def prefill_only():
        return backend.forward(0, q[B:], k[B:], v[B:], pre_meta)

    print(f"B={B} decode rows at ctx {C} + one {Q}-token chunk, heads {H}/{Hkv}x{D}, block {bs}")
    cases = [("decode_only", decode_only), ("prefill_only", prefill_only), ("ours", ours)]
    a = ours()
    try:
        b = varlen_paged()
        torch.cuda.synchronize()
        err = (a.float() - b.float()).abs().max().item()
        print(f"varlen_paged vs ours: max abs diff {err:.2e} ({'OK' if err < 2e-2 else 'MISMATCH'})")
        cases.append(("varlen_paged", varlen_paged))
    except (TypeError, RuntimeError) as exc:  # a flash-attn without paged varlen
        print(f"varlen_paged unavailable in this flash-attn: {type(exc).__name__}: {exc}")
    for name, fn in cases:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        e0.record()
        for _ in range(args.iters):
            fn()
        e1.record()
        host_ms = (time.perf_counter() - t0) * 1e3 / args.iters
        torch.cuda.synchronize()
        gpu_ms = e0.elapsed_time(e1) / args.iters
        print(f"{name:<14} gpu {gpu_ms * 1e3:8.1f} us/layer  host {host_ms * 1e3:8.1f} us/layer  "
              f"x{args.layers} layers = gpu {gpu_ms * args.layers:6.2f} ms, host {host_ms * args.layers:6.2f} ms")


if __name__ == "__main__":
    main()
