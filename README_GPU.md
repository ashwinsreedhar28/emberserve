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

## Kernel micro-benchmark

Decode attention only, H=14 Hkv=2 D=64 fp16, median of 20; `results/kernels*.json`.
Ratio = triton time / flash-attn time (1.0 = parity).

**A100 SXM 80 GB, tl.dot variant (now the default), split-K auto:**

| triton block | ctx | B=1 | B=8 | B=32 | B=128 |
|---|---|---|---|---|---|
| 256 | 128 | 2.04x | 1.98x | 1.98x | 1.92x |
| 256 | 512 | 2.40x | 2.31x | 2.25x | 1.73x |
| 256 | 2048 | 2.26x | 2.25x | 1.93x | **1.16x** (835 vs 972 GB/s) |
| 16 | 2048 | 2.26x | 2.24x | 1.92x | 1.54x |

Same GPU, the broadcast-multiply `sum` variant (the original kernel): 3.3x–10.8x behind.
Moving QK^T and PV onto tensor cores (`tl.dot`, query heads padded 7 -> 16) is what
closed the gap at large batch; what remains is a flat ~0.04 ms per call that does not
scale with work (Triton launch overhead plus the padded rows), so short contexts and small
batches stay ~2x behind. With split-K forced off, ctx <= 512 improves to 1.6-1.9x and
ctx 2048 at block 16 degrades to 3.4x, which is why the heuristic now splits only past
1k keys. Block 16 costs nothing over block 256 below ctx 2048.

RTX 4090 baseline (sum variant, before any of this): 0.708 ms vs flash 0.170 ms at
B=128/ctx=2048 (190 vs 789 GB/s), `results/kernels.json`.

Knobs for A/B runs:

```bash
python scripts/bench_kernels.py --variant sum            # CUDA-core path
python scripts/bench_kernels.py --variant dot            # tensor-core path (default)
python scripts/bench_kernels.py --splits 1               # no split-K
PAGEDSERVE_TRITON_VARIANT=sum PAGEDSERVE_TRITON_SPLITS=1 python scripts/gpu_smoke.py   # same knobs, engine-wide
```

## The CUDA-graph "capture failure" that wasn't

`pytest -m gpu` failed every Triton engine-level test with `operation failed due to a
previous error during capture` on two different GPUs, while the same engine captured fine
from a standalone script. Root cause: `tests/test_paged_triton.py` sets `TRITON_INTERPRET=1`
at import so the kernel can run on CPU; pytest imported it into the same process as the GPU
tests, the kernel cache was built in interpreter mode, and every "GPU" kernel test silently
ran on the CPU (10-minute runs at 0% GPU). Inside a graph capture the interpreter's host
copies are illegal, hence the error. Fixed by skipping the interpreter module when CUDA is
present and by making the backend refuse `TRITON_INTERPRET=1` with a CUDA cache. Lesson:
process-global env flags in a test module are shared state.

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
