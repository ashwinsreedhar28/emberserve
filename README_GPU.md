# GPU setup (Runpod RTX 4090 / A40, CUDA 12.x)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation   # or a prebuilt wheel:
#   https://github.com/Dao-AILab/flash-attention/releases  ->  flash_attn-2.8.3+cu12torch2.x cxx11abiTRUE-cp311-...whl
pip install -e '.[hf,server,bench]'
python scripts/download_model.py
python scripts/gpu_smoke.py           # Wed gate: identical tokens across backends + tok/s table
python -m pytest -m gpu -q            # test_paged_flash_gpu, test_cuda_graphs_gpu, test_paged_triton_gpu
python scripts/bench_kernels.py       # decode-attention kernel table (ms + GB/s) -> results/kernels.json
```

## vLLM goes in its own venv

`pip install vllm` replaces torch with the version vLLM pins (2.13/cu130 on Sep 27, 2026),
which breaks the flash-attn binary built against the pod's torch (`undefined symbol:
_ZN3c104cuda29c10_cuda_check_implementation...`). Keep them apart:

```bash
python -m venv /opt/vllm && /opt/vllm/bin/pip install vllm
python -m pagedserve.bench.run_vllm_baseline --server vllm --vllm-bin /opt/vllm/bin/vllm ...
```

If it already happened: `pip uninstall -y vllm && pip install torch==2.8.0 --index-url
https://download.pytorch.org/whl/cu128 && pip install --no-cache-dir --force-reinstall
flash-attn --no-build-isolation`.

## Block size 256 with `paged_flash`

Upstream flash-attn (2.6.3 through 2.8.3.post1 and `main`) hard-checks
`page_block_size % 256 == 0` for paged KV (`flash_api.cpp`: "Paged KV cache block size
must be divisible by 256"). Only vLLM's `vllm-flash-attn` fork allows 16. So GPU runs use
`--block-size 256 --attn-backend paged_flash --dtype float16`; the backend refuses
anything else with a clear error. `paged_torch` still supports 16, which is the
memory-utilization ablation: block 16 vs 256 internal fragmentation at the same request
mix. If a future flash-attn release relaxes the check, lower
`FLASH_PAGE_MULTIPLE` in `pagedserve/attn/paged_flash.py` (gate it on
`flash_attn.__version__`).

## Triton decode kernel (`--attn-backend paged_triton`)

`pagedserve/attn/paged_triton.py` is a hand-written Triton PagedAttention kernel for the
decode step (prefill is delegated to `paged_flash` when the block size allows it, else to
`paged_torch`). Grid `(B, Hkv, num_splits)`: one program owns one sequence, one KV head
and ALL of that head's GQA query heads (7 for Qwen2.5-0.5B, padded to 8), so each K/V
element is read from HBM once per KV head instead of once per query head. The program
walks the sequence's block-table pages in 16-position tiles with an online softmax
(running max / sum / fp32 accumulator, rescaled per tile). `num_splits > 1` is
flash-decoding: contiguous tile ranges go to separate programs and a tiny reduce kernel
merges the partials, chosen from shapes only so CUDA-graph replay is stable. It needs
only `block_size % 16 == 0`.

Why that matters: `paged_flash` forces block 256, and in the ablation a **64-token shared
prefix got ZERO prefix-cache hits at block 256** — a prefix shorter than one block never
fills a block, so no full block was ever shared. With block 16 the same 64-token prefix
is 4 complete blocks, all shared. `paged_triton` is the backend that makes prefix
caching and the low-fragmentation block-16 layout usable on the GPU, and it is
graph-capable (`--enable-cuda-graphs` works with it).

```bash
python -m pytest -m gpu -q -k triton     # kernel vs paged_torch/paged_flash, engine greedy parity
python scripts/bench_kernels.py          # ms/call + effective K/V GB/s, B x ctx, block 256 and 16
# engine flags (the CLI's --attn-backend choices need "paged_triton" added in cli.py):
#   --device cuda --dtype float16 --attn-backend paged_triton --block-size 16 \
#   --enable-prefix-caching --enable-cuda-graphs
```

## CUDA graphs

`--enable-cuda-graphs` (needs cuda + paged_flash or paged_triton) captures decode for batch buckets
1,2,4,...,256 (capped at `max_num_seqs`). One KV block (the last id) is reserved as
scratch for padding rows, so the BlockManager sees `num_blocks - 1`.

## Kernel micro-benchmark, RTX 4090 (pre-optimization baseline)

Decode attention only, H=14 Hkv=2 D=64 fp16, median of 20; `results/kernels.json`.

| backend | block | ctx | B=1 | B=8 | B=32 | B=128 |
|---|---|---|---|---|---|---|
| paged_torch | 256 | 2048 | 0.178 ms / 6 GB/s | 0.616 ms / 14 GB/s | 2.666 ms / 13 GB/s | 10.452 ms / 13 GB/s |
| paged_flash | 256 | 2048 | 0.020 ms / 53 GB/s | 0.020 ms / 410 GB/s | 0.035 ms / 967 GB/s | 0.170 ms / 789 GB/s |
| paged_triton | 256 | 2048 | 0.148 ms / 7 GB/s | 0.147 ms / 57 GB/s | 0.214 ms / 157 GB/s | 0.708 ms / 190 GB/s |
| paged_triton | 16 | 2048 | 0.147 ms / 7 GB/s | 0.147 ms / 57 GB/s | 0.212 ms / 158 GB/s | 0.700 ms / 192 GB/s |

Reading: block 16 costs nothing over block 256 in the Triton kernel (same time at every
shape), which is the point. The kernel is correct (max err 5e-4 vs paged_torch) but ~4x
behind flash-attn at large batch and has a ~0.15 ms floor at small batch that came from
splitting the context whenever `B*Hkv < 512` (the reduce launch dominates). The split
heuristic is now occupancy-based (see `default_num_splits`). Knobs for A/B runs:

```bash
python scripts/bench_kernels.py --variant sum            # CUDA-core path (default, validated)
python scripts/bench_kernels.py --variant dot            # tensor-core path: tl.dot for QK^T and PV
python scripts/bench_kernels.py --splits 1               # no split-K
PAGEDSERVE_TRITON_VARIANT=dot PAGEDSERVE_TRITON_SPLITS=1 python scripts/gpu_smoke.py   # same knobs, engine-wide
```

Each run prints a triton/flash ratio table; 1.0x is parity.

## If CUDA-graph capture fails

`CUDAGraphRunner.capture()` now re-runs the failing bucket eagerly, layer by layer, and
reports which one breaks. For the raw kernel name:

```bash
python scripts/gpu_debug_capture.py --backend paged_triton --block-size 16   # CUDA_LAUNCH_BLOCKING=1
```

## Fresh pod

```bash
git clone https://github.com/ashwinsreedhar28/pagedserve && cd pagedserve && bash scripts/pod_setup.sh
```
