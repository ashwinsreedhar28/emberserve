# pagedserve

A from-scratch LLM inference engine in PyTorch: continuous batching, a paged KV cache with
block tables, automatic prefix caching, recompute preemption, flash-attn paged decode,
CUDA-graph decode, and an OpenAI-compatible streaming server. Benchmarked against vLLM on
the same GPU and model.

The point is not to beat vLLM. It is to land within a measurable fraction of it and be able
to explain every part of the gap.

**Model:** Qwen/Qwen2.5-0.5B-Instruct (24 layers, 14 query heads / 2 KV heads, head_dim 64).
`transformers` is used only for the tokenizer and as the golden reference; the model code,
attention, cache, scheduler and sampler are all written here.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Qwen2 from scratch (RMSNorm, RoPE, GQA, SwiGLU), safetensors loader, naive per-sequence KV cache | done, tested |
| 2 | Sampler (temp / top-k / top-p / repetition penalty / seeds / stop), continuous-batching scheduler, `LLMEngine` step loop | done, tested |
| 3 | `BlockManager` + paged KV cache + PyTorch paged attention (gather) | done, tested |
| 4 | flash-attn paged decode + varlen prefill (`paged_flash`) | written, **needs GPU validation** |
| 5 | Recompute preemption, hash-chained prefix caching with LRU eviction | done, tested |
| 6 | CUDA-graph decode at batch buckets | written, **needs GPU validation** |
| 7 | FastAPI `/v1/completions`, `/v1/chat/completions`, SSE streaming, disconnect cancellation, `/metrics` | done, tested |
| 8 | Bench harness: ShareGPT-like traces, Poisson load generator, in-process ablation, vLLM baseline runner, plots | done, tested on CPU |
| gate | Token-for-token match vs HF greedy on 7 prompts × 64 tokens + logits atol 1e-3 | script ready (`make golden`) |

146 CPU tests, all on a 2-layer random-weight model, no download or GPU needed: `make test`.

## Layout

```
pagedserve/
  config.py            ModelConfig (mirrors HF config.json) / EngineConfig
  model/               qwen2.py (from scratch), rope.py, weights.py (safetensors -> our modules)
  attn/                base.py (AttnMetadata + backend contract, packed token layout)
                       naive.py | paged_torch.py | paged_flash.py | cuda_graphs.py
  kv/                  block_manager.py, cache.py (paged K/V tensors), prefix_cache.py
  sched/               request.py, scheduler.py (prefill-priority, preemption)
  sampling.py          per-request sampling + stop checks
  engine.py            LLMEngine.step(): schedule -> build inputs -> forward -> sample -> postprocess
  llm.py               offline LLM.generate()
  tokenizer.py         HF tokenizer wrapper + incremental detokenizer (stop strings)
  server/              AsyncLLMEngine (worker thread), OpenAI types, FastAPI app
  bench/               trace, load, offline, ablation, run_vllm_baseline, plot
scripts/               download_model, dump_golden, check_golden, gpu_smoke
tests/                 one file per component + test_engine.py (end-to-end gates)
```

## How it works

Every step is either a **prefill batch** (new requests, packed with no padding; token budget
`max_num_batched_tokens`) or a **decode batch** (one token for every running request). The
scheduler owns the `BlockManager`; a request is admitted only when its blocks can be
allocated, and when decode runs out of blocks the youngest running request is preempted
(blocks freed, `num_computed_tokens` reset, back to the front of the queue) and recomputed
later. Outputs are identical either way — that is a test.

With `--enable-chunked-prefill` (Sarathi-Serve / vLLM style) a step is instead **mixed**: one
decode token for every running request whose prefill is done, plus as many prompt tokens as
fit in the remaining `max_num_batched_tokens` (running requests still mid-prefill first,
then new requests FIFO). A long prompt is split across steps, so it no longer stalls every
decoder's TPOT for one long step; `max_num_batched_tokens` becomes the per-step cap (512–2048
is typical) and prompts longer than it are accepted. A partial chunk writes K/V and emits
nothing; the token is sampled only on the step that completes the prompt
(`SchedulerOutput.prefill_complete`). Chunked outputs equal unchunked outputs — also a test.

Attention backends share one contract (`attn/base.py`): tokens are packed
`[num_tokens, heads, head_dim]`, `context_lens[i]` is the KV length after the step, and the
backend writes this step's K/V then attends causally. `paged_torch` gathers blocks via block
tables in plain PyTorch (the reference); `paged_flash` calls `flash_attn_with_kvcache` with
the same block tables. Prefix caching hashes every full block by its content chain; a new
request that hits skips those tokens in prefill and shares the physical blocks (ref counted,
LRU-evicted when unreferenced).

## Run locally (Mac / CPU)

```bash
pip install -e '.[hf,server,dev]'
python scripts/download_model.py                 # ~1 GB into models/
python scripts/dump_golden.py                    # HF reference outputs -> golden/
python scripts/check_golden.py                   # naive + paged_torch must match token-for-token
python -m pagedserve.cli generate --model models/Qwen2.5-0.5B-Instruct --prompt "The capital of France is" --max-tokens 32
python -m pagedserve.cli serve --model models/Qwen2.5-0.5B-Instruct --port 8000
```

Then any OpenAI client works: `openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")`.

## Run on a GPU

See `README_GPU.md` (flash-attn requires paged block size 256; that is itself an ablation).

## Results (RTX 4090, Qwen2.5-0.5B-Instruct fp16, 200 requests, seed 0)

Same trace for every row; `results/*.json`. Prompt median 208 tokens, output median 131.

**Open-loop, 8 req/s** (arrivals span 24.3 s — a config that keeps up finishes in ~25 s):

| config | tok/s | run | TTFT p50/p99 ms | TPOT p50/p99 ms |
|---|---:|---:|---:|---:|
| naive (per-seq torch.cat cache) | 343 | ~108 s | – | – |
| static batching | 595 | ~62 s | – | – |
| paged_torch (gather) | 700 | 52.8 s | 12 / 28 | 86 / 141 |
| paged_flash | 1,329 | 27.8 s | 8.7 / 10.0 | 9.1 / 9.9 |
| paged_flash + CUDA graphs | 1,441 | 25.6 s | 8.7 / 10.6 | **3.8 / 4.8** |

**Saturation, all 200 at t=0:** paged_torch 738 → paged_flash 3,043 → +graphs **5,459 tok/s**.

Two things the numbers taught us. Prefix caching with a 64-token shared prefix did
nothing at block size 256 (no full block is ever shared); at 512 tokens it only moved
TTFT p99 from 10.3 to 9.5 ms because prefill on a 0.5B model is ~8 ms to begin with.
And CUDA graphs are worth more than any kernel at this model size: decode is ~2 ms of
kernels and the rest was launch overhead, so graphs cut TPOT by 2.4x.

Decode-kernel micro-benchmark and the Triton kernel's standing are in `README_GPU.md`.

## Benchmark

```bash
python -m pagedserve.bench.ablation --model models/Qwen2.5-0.5B-Instruct --device cuda --dtype float16 \
  --configs naive,static,paged_torch,paged_torch+prefix,paged_flash,paged_flash+graphs \
  --trace-n 200 --request-rate 8 --shared-prefix-len 64 --out results/ablation.json
python -m pagedserve.bench.run_vllm_baseline --server vllm --model Qwen/Qwen2.5-0.5B-Instruct --rates 1,2,4,8,16,inf --name vllm
python -m pagedserve.bench.run_vllm_baseline --server pagedserve --model models/Qwen2.5-0.5B-Instruct \
  --server-args "--device cuda --attn-backend paged_flash --block-size 256 --enable-prefix-caching --enable-cuda-graphs" \
  --rates 1,2,4,8,16,inf --name pagedserve
python -m pagedserve.bench.plot results/vllm.json results/pagedserve.json --ablation results/ablation.json --out-dir results/plots
```

The vLLM comparison is the remaining run; the gap analysis lands with it.
