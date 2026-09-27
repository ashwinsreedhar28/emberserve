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
