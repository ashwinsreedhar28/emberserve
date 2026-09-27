# GPU setup (Runpod RTX 4090 / A40, CUDA 12.x)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation   # or a prebuilt wheel:
#   https://github.com/Dao-AILab/flash-attention/releases  ->  flash_attn-2.8.3+cu12torch2.x cxx11abiTRUE-cp311-...whl
pip install -e '.[hf,server,bench]'
python scripts/download_model.py
python scripts/gpu_smoke.py           # Wed gate: identical tokens across backends + tok/s table
python -m pytest -m gpu -q            # tests/test_paged_flash_gpu.py, tests/test_cuda_graphs_gpu.py
```

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

## CUDA graphs

`--enable-cuda-graphs` (needs cuda + paged_flash) captures decode for batch buckets
1,2,4,...,256 (capped at `max_num_seqs`). One KV block (the last id) is reserved as
scratch for padding rows, so the BlockManager sees `num_blocks - 1`.
