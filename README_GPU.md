# GPU notes (Runpod RTX 4090 / A100 SXM, torch 2.8.0+cu128, CUDA 12.x)

## Fresh pod, one shot

```bash
git clone https://github.com/ashwinsreedhar28/pagedserve && cd pagedserve && bash scripts/pod_setup.sh
```

`pod_setup.sh` installs flash-attn (prebuilt wheel for the pod's torch), the package with
`[hf,server,dev]`, downloads the model, dumps the golden reference, and builds vLLM in its
own venv at `/opt/vllm` (`--no-vllm` skips that; it is the slow part). Then:

```bash
python scripts/gpu_smoke.py                    # every backend must print identical tokens; tok/s table
python -m pytest -m gpu -n 4 -v                # 31 GPU tests (xdist so progress is visible)
python scripts/check_golden.py --device cuda --dtype float16 --backends paged_flash,paged_triton --block-size 256
python scripts/bench_kernels.py                # decode-attention table (ms + GB/s + triton/flash ratio) -> results/kernels.json
```

Runpod ops that cost us time: run anything longer than a minute inside `tmux` (the web
terminal's "connection closed" leaves the old process alive and holding GPU memory; the next
run then OOMs). SSH the pod directly (`ssh -i ~/.ssh/runpod -p <port> root@<ip>`, the TCP
port from the pod's Connect panel), not through the `ssh.runpod.io` proxy. Without a network
volume everything under `/root` dies with the pod, so copy `results/` back before stopping it:

```bash
scp -i ~/.ssh/runpod -P <port> -r root@<ip>:/root/pagedserve/results/ .   # from the Mac clone
```

The engine reserves 90% of free GPU memory for the KV cache at startup, so one GPU job at a
time; `nvidia-smi --query-gpu=memory.used --format=csv` should read ~0 MiB before a run.

## Serving defaults on CUDA

`pagedserve serve --device cuda` defaults to the engine in its own process
(`--engine-process`; `--no-engine-process` for the single-process path), async scheduling
(`--no-async-scheduling` to compare), and chunked prefill with a 2048-token per-step cap for
checkpoints of 4 GB and up (`--enable-chunked-prefill` / `--no-chunked-prefill` to force;
`--max-num-batched-tokens 512` for a tighter TPOT tail at ~6% throughput). All measured on
the A100: the process split took Qwen2.5-0.5B from 10.6k to 13.9k tok/s at saturation and
async scheduling to 14.4k; chunked prefill took Qwen2.5-7B from 89% to 97% of vLLM but
costs 11% at 0.5B, where an eager mixed step loses to a graph-replayed decode step.

### Async scheduling A/B

`--async-scheduling` overlaps each step's CPU work with the previous step's GPU work
(README, "Scheduler"); default on. The A/B that made it the default (v6 → v7b: 13,945 →
14,394 tok/s at saturation, TPOT 2.1 → 1.8 ms at 1 req/s):

```bash
python -m pagedserve.bench.run_vllm_baseline --server pagedserve --model models/Qwen2.5-0.5B-Instruct --dtype float16 \
  --max-model-len 4096 --server-args "--device cuda --attn-backend paged_flash --block-size 256 --enable-cuda-graphs" \
  --rates 1,2,4,8,16,inf --trace-n 200 --name pagedserve_flash_v7b
```

(`--no-async-scheduling` for the other arm.) The tiny-model GPU tests
(`tests/test_cuda_graphs_gpu.py::test_async_scheduling_matches_sync_on_cuda`) check token
parity with and without graphs first.

### Piecewise CUDA graphs A/B

`--piecewise-cuda-graphs` (with `--enable-cuda-graphs`) replays prefill and mixed steps
from per-layer graphs with attention eager in between (README, "CUDA graphs"). GPU parity
tests first, then the 0.5B chunked sweep against `pagedserve_flash_v7.json` (12,786 tok/s,
chunked without piecewise) and `_v7b.json` (14,394, prefill-priority):

```bash
python -m pytest tests/test_cuda_graphs_gpu.py tests/test_mla_triton_gpu.py -q -k piecewise
python -m pagedserve.bench.run_vllm_baseline --server pagedserve --model models/Qwen2.5-0.5B-Instruct --dtype float16 \
  --max-model-len 4096 --server-args "--device cuda --attn-backend paged_flash --block-size 256 --enable-cuda-graphs --enable-chunked-prefill --piecewise-cuda-graphs" \
  --rates 1,2,4,8,16,inf --trace-n 200 --name pagedserve_flash_v8
```

## DeepSeek / Moonlight checkpoints

`scripts/download_model.py` fetches the repo's own tokenizer/modeling code and tiktoken
files; the golden dump needs `pip install tiktoken blobfile` and `--trust-remote-code` for
Moonlight's tokenizer (the model itself loads through transformers' native DeepSeek-V3).
The 16B model is built directly on the GPU in bf16 (32 GB); the fp32 HF reference is 64 GB,
so run the dump first, then the check:

```bash
python scripts/dump_golden.py --model models/Moonlight-16B-A3B-Instruct --out golden/moonlight --device cuda --trust-remote-code
python scripts/check_golden.py --model models/Moonlight-16B-A3B-Instruct --golden golden/moonlight --device cuda --dtype bfloat16 --backends mla_torch,mla_triton --block-size 16
```

Serving it: `--attn-backend mla_triton --block-size 16 --enable-cuda-graphs
--enable-chunked-prefill --async-scheduling` (the Triton MLA decode kernel and the fused MoE
grouped GEMM are both captured; `PAGEDSERVE_FUSED_MOE=0` falls back to the per-expert loop,
which was 63.6 ms per batch-1 step against 7.2 ms fused). Where a step's time goes, per
kernel (this is what found the split-K bug under graphs, README "Moonlight"):

```bash
python scripts/profile_step.py --model models/Moonlight-16B-A3B-Instruct --device cuda --dtype bfloat16 \
  --attn-backend mla_triton --block-size 16 --enable-cuda-graphs --batches 1,128 --kernels 1,128 --top 30
python scripts/bench_moe.py      # the MoE layer alone: ms and effective weight GB/s per grouped-GEMM tile config
```

Reading `bench_moe` at M=1: the ~0.28 ms per layer it reports is Python launch overhead
(5 Triton/torch launches), not GPU time; inside a CUDA graph the same layer is ~73 us
(the profile's `_grouped_gemm_kernel` row), 71% of HBM bandwidth for the six experts read.

## vLLM goes in its own venv

`pip install vllm` replaces torch with the version vLLM pins (2.13/cu130 on Sep 27, 2026),
which breaks the flash-attn binary built against the pod's torch (`undefined symbol:
_ZN3c104cuda29c10_cuda_check_implementation...`). Keep them apart:

```bash
python -m venv /opt/vllm && /opt/vllm/bin/pip install vllm
python -m pagedserve.bench.run_vllm_baseline --server vllm --vllm-bin /opt/vllm/bin/vllm ...
```

`run_vllm_baseline` puts the vllm binary's directory on the child's `PATH`; vLLM's
FlashInfer backend JIT-compiles with `ninja` from that venv and fails with
`FileNotFoundError: 'ninja'` otherwise.

If torch already got replaced: `pip uninstall -y vllm && pip install torch==2.8.0
--index-url https://download.pytorch.org/whl/cu128 && pip install --no-deps --no-cache-dir
--force-reinstall flash-attn --no-build-isolation` (`--no-deps`, or flash-attn's resolver
pulls a newer torch right back).

## Block size 256 with `paged_flash`

Upstream flash-attn (2.6.3 through 2.8.3.post1 and `main`) hard-checks
`page_block_size % 256 == 0` for paged KV (`flash_api.cpp`: "Paged KV cache block size
must be divisible by 256"). Only vLLM's `vllm-flash-attn` fork allows 16. So `paged_flash`
runs use `--block-size 256`; the backend refuses anything else with a clear error, and the
ablation runner clamps per backend (`MIN_BLOCK`) so one command can compare backends at
their own minimum block size. `paged_torch` and `paged_triton` support 16, which is the
memory-utilization ablation: 98% slot utilization at block 16 vs 76% at 256 on the same
request mix (A100). If a future flash-attn release relaxes the check, lower
`FLASH_PAGE_MULTIPLE` in `pagedserve/attn/paged_flash.py` (gate it on
`flash_attn.__version__`).

## Triton decode kernel (`--attn-backend paged_triton`)

`pagedserve/attn/paged_triton.py` is a hand-written Triton PagedAttention kernel for the
decode step. Prefill runs flash-attn's packed *varlen* kernel, which has no block-size
constraint: a fresh prompt attends over the step's own packed k/v, and a chunk row that
attends through the cache (chunked prefill, cached prefix) has its context gathered out of
the paged cache into a packed buffer first while the decode rows of the same step take the
Triton kernel (`_prefill_mixed`). Without flash-attn prefill falls back to `paged_torch`.
Grid `(B, Hkv, num_splits)`: one program owns one
sequence, one KV head and ALL of that head's GQA query heads (7 for Qwen2.5-0.5B, padded
to 16 for `tl.dot`), so each K/V element is read from HBM once per KV head instead of once
per query head. The program walks the sequence's block-table pages in tiles with an online
softmax (running max / sum / fp32 accumulator, rescaled per tile). `num_splits > 1` is
flash-decoding: contiguous tile ranges go to separate programs and a tiny reduce kernel
merges the partials. Each split's tile range is computed inside the kernel from the
sequence's real context length (under CUDA graphs the block table is padded to
`max_model_len`; a host-side partition of that width gave split 0 every real tile, which
the Moonlight per-kernel profile caught). Splits are chosen from shapes only (occupancy target 2x SMs, only past
1k keys, max 16) so CUDA-graph replay is stable. It needs only `block_size % 16 == 0`.

Why that matters: `paged_flash` forces block 256, and in the ablation a **64-token shared
prefix got ZERO prefix-cache hits at block 256**: a prefix shorter than one block never
fills a block, so no full block was ever shared. With block 16 the same 64-token prefix is
4 complete blocks, all shared. `paged_triton` is the backend that makes prefix caching and
the low-fragmentation block-16 layout usable on the GPU, and it is graph-capable.

```bash
python -m pytest -m gpu -n 4 -v -k triton   # kernel vs paged_torch/paged_flash at B in {1,8,32,128}, block 16/256; engine greedy parity; graphs
python scripts/bench_kernels.py             # ms/call + effective K/V GB/s, B x ctx, block 256 and 16
python -m pagedserve.cli serve --model models/Qwen2.5-0.5B-Instruct --device cuda --dtype float16 \
  --attn-backend paged_triton --block-size 16 --enable-prefix-caching --enable-cuda-graphs
```

### Kernel micro-benchmark

Decode attention only, H=14 Hkv=2 D=64 fp16, median of 20; `results/kernels*.json`.
Ratio = triton time / flash-attn time (1.0 = parity).

**A100 SXM 80 GB, tl.dot variant (the default), split-K auto:**

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
ctx 2048 at block 16 degrades to 3.4x, which is why the heuristic splits only past 1k keys.
Block 16 costs nothing over block 256 below ctx 2048.

RTX 4090 baseline (sum variant, before any of this): 0.708 ms vs flash 0.170 ms at
B=128/ctx=2048 (190 vs 789 GB/s), `results/kernels.json`.

Knobs for A/B runs:

```bash
python scripts/bench_kernels.py --variant sum            # CUDA-core path
python scripts/bench_kernels.py --variant dot            # tensor-core path (default)
python scripts/bench_kernels.py --splits 1               # no split-K
PAGEDSERVE_TRITON_VARIANT=sum PAGEDSERVE_TRITON_SPLITS=1 python scripts/gpu_smoke.py   # same knobs, engine-wide
```

## CUDA graphs

`--enable-cuda-graphs` (needs cuda + `paged_flash` or `paged_triton`) captures decode for
batch buckets 1,2,4,...,256 (capped at `max_num_seqs`). One KV block (the last id) is
reserved as scratch for padding rows, so the BlockManager sees `num_blocks - 1`. Capture
happens at engine construction, so a server that starts is a server whose graphs work.

### The "capture failure" that wasn't

`pytest -m gpu` failed every Triton engine-level test with `operation failed due to a
previous error during capture` on two different GPUs, while the same engine captured fine
from a standalone script. Root cause: `tests/test_paged_triton.py` sets `TRITON_INTERPRET=1`
at import so the kernel can run on CPU; pytest imported it into the same process as the GPU
tests, the kernel cache was built in interpreter mode, and every "GPU" kernel test silently
ran on the CPU (10-minute runs at 0% GPU). Inside a graph capture the interpreter's host
copies are illegal, hence the error. Fixed by skipping the interpreter module when CUDA is
present (`PAGEDSERVE_FORCE_INTERPRETER=1` overrides) and by making the backend refuse
`TRITON_INTERPRET=1` with a CUDA cache. Lesson: process-global env flags in a test module
are shared state.

### If capture fails for real

`CUDAGraphRunner.capture()` re-runs the failing bucket eagerly, layer by layer, and reports
which one breaks. For the raw kernel name:

```bash
python scripts/gpu_debug_capture.py --backend paged_triton --block-size 16   # CUDA_LAUNCH_BLOCKING=1
```

## fp16 and the golden gate

`check_golden.py` compares every run to the fp32 Hugging Face reference. fp32 on the GPU
matches exactly. fp16 is held to a self-calibrated bar: logits within 1.0, and a token
mismatch counts as a tie-break (not a failure) when the top-2 logit gap at that position is
below 2x the logits error measured on prompt 0. `gpu_smoke.py --tie-margin` reports a
divergence whose top-2 gap is below the margin as a tie-break, so a real bug (a large
margin) is distinguishable from numerics.
`dump_golden.py --device cuda` regenerates the reference on the GPU; pod CPUs can take
10+ minutes for the same job.
