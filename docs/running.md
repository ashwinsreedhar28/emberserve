# Running it

Part of [pagedserve](../README.md). Locally, on a GPU, on Runpod Serverless, and the benchmark commands behind the numbers.

## Locally (Mac / CPU, fp32)

```bash
pip install -e '.[hf,server,dev]'
python scripts/download_model.py                 # ~1 GB into models/
make golden                                      # HF reference -> golden/, then check naive + paged_torch
python -m pagedserve.cli generate --model models/Qwen2.5-0.5B-Instruct --prompt "The capital of France is" --max-tokens 32
python -m pagedserve.cli serve --model models/Qwen2.5-0.5B-Instruct --port 8000
```

Then any OpenAI client works:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
for chunk in client.chat.completions.create(model="models/Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": "Explain paged attention in one sentence."}],
        stream=True, max_tokens=64):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## On a GPU

[GPU notes](gpu.md) cover the pod setup (`scripts/pod_setup.sh` does it in one shot), the
flash-attn block-256 constraint, the vLLM venv, the Triton kernel knobs, CUDA-graph debugging,
and the pitfalls we hit. The serving config used for the numbers above:

```bash
python -m pagedserve.cli serve --model models/Qwen2.5-0.5B-Instruct --device cuda --dtype float16 \
  --attn-backend paged_triton --block-size 16 --enable-cuda-graphs --enable-prefix-caching
```

Add `--api-workers 2` for high-concurrency serving of a small model: two API processes in
front of the one engine core ([Two API processes](results.md#two-api-processes---api-workers)).

## On Runpod Serverless

`deploy/runpod/` has a worker and a Dockerfile that bakes a model into the image. The
worker is a proxy in front of `pagedserve serve` (the layout Runpod's own `worker-vllm`
uses): the same server as everywhere else, engine-core process and all, so the endpoint is
OpenAI-compatible and the benchmark client runs against it unchanged. Runpod builds the
image from the repo (Serverless → New endpoint → GitHub repo, Dockerfile path
`deploy/runpod/Dockerfile`); `deploy/runpod/README.md` has the endpoint, `curl` and
benchmark steps, and the job contract is tested against the real app on the CPU engine
(`tests/test_runpod_handler.py`).

Deployed and measured Sep 27 on one RTX 4090 worker, the same 200-request synthetic trace
as the A100 rows, from a laptop across the internet (`results/runpod_serverless_*.json`).
Server-side, the engine was the same in every run: TPOT 1.7–2.0 ms, TTFT 8–11 ms, e2e
~330 ms per request. Everything else in the table is the delivery path, and the first row
is a bug of mine:

| endpoint type | TPOT p50, 1–4 req/s | TTFT p50 @ 4 req/s | saturation | TTFT p50 / p99 at saturation |
|---|---:|---:|---:|---:|
| Queue, one yield per server write (as first built) | 56–231 ms | 4.4 s | 95 tok/s, 100 of 1,200 requests failed | 37 s / 82 s |
| Queue, yields coalesced per 100 ms (`STREAM_FLUSH_MS`) | 1.4–1.5 ms | 2.4 s | 9,162 tok/s | 1.5 s / 2.6 s |
| Load balancer, default concurrency (4 per worker) | 1.7–1.8 ms | 220 ms | 1,013 tok/s | 13 s / 34 s |
| Load balancer, concurrency 200 | 1.9–2.0 ms | 230 ms (@ 8 and 16) | **11,079 tok/s** | 0.82 s / 1.07 s |

On a **Queue** endpoint every chunk a streaming job yields is one call from the worker to
Runpod's job-stream API (~115 ms round trip, rate-limited across the endpoint): with one
stream, one token per write, that pinned a stream at 9 tokens/s while the engine sat
idle, and 200 streams hit the limit and failed. Coalescing yields per 100 ms restores the
throughput (Runpod's old `worker-vllm` had `BATCH_SIZE` knobs for the same reason) but not
the latency: a job still waits seconds in the queue's dispatch at moderate rates. A
**Load balancer** endpoint routes HTTP straight to the container's port and only polls
`/ping`, so the endpoint *is* the server: ~200 ms of gateway round trip on top of the
server's 10 ms TTFT, and the full saturation throughput — once the endpoint's "request
count" is raised, because at its default the balancer admits ~4 concurrent requests per
worker and queues the rest in front of a server that holds 200. The 4090's 11.1k tok/s
against the A100's 16.6k is the memory-bandwidth ratio. Both endpoint types are the same
image, chosen by `RUNPOD_LB=1`.

Cold starts were measured as a series (`scripts/serverless_coldstart.py`, three samples
per row, each taken only after `/health` showed no worker and confirmed by a fresh
`[worker] pagedserve up in X s` line in the worker log, so a parked container never
counts as one). The weights are baked into the image (9.8 GB for 0.5B, ~25 GB for 7B via
`deploy/runpod/Dockerfile.7b`), so nothing downloads at start. Runpod's `delayTime` is
its own queue-to-handler number:

| image | FlashBoot | `delayTime` per cold sample | container start → healthy |
|---|---|---:|---:|
| 0.5B | on | 87.1 s (fresh host) / 25.9 / 22.8 s | 26.1 / 17.9 / 16.3 s |
| 0.5B | off | 29.2 / 25.1 / 29.3 s | 22.1 / 17.4 / 20.1 s |
| 7B | on | 22.6 s / ≈95 s (fresh host) / 0.85 s (resumed) | 12.3 / 14.5 / — s |
| 7B | off | 24.1 / 65.6 s (host GPU taken) / 27.4 s | 15.6 / 14.5 / 15.3 s |

So a 7B cold start on a host that has the image is 22–27 s end to end, of which 12–16 s
is the worker loading 15 GB of weights and capturing graphs; a fresh host adds the pull
(~50 s for 9.8 GB, ~70 s for 25 GB), and a shared host can add a wait for its own GPU:
a stopped container holds no 4090, so one sample sat 49 s while another tenant's job had
it (`/health` calls that worker `throttled`) before Runpod moved the slot to another host. worker-vllm serving Qwen3-8B took
138–144 s from container start to healthy on an A100 and an H100, re-downloading the
weights and running torch.compile in full on every start (the network volume meant to
cache them did not mount while the endpoint's model cache setting was on; with it off, a cache hit took 114 s, compile loaded in ~1 s but the 15 GB checkpoint took ~30 s to read off the FUSE-mounted volume), which is the point of baking the weights in. Two things the series taught: FlashBoot resumed the
paused container once in six tries (0.85 s) and restarted it the other five, so it did
not move the median; and hosts vary — one 0.5B worker took 155.6 s from container start
to healthy on the same image that boots in 16–26 s elsewhere.
`deploy/runpod/README.md` has the full table and the anatomy of a `delayTime`.

The 7B's throughput on the same 4090 (Queue endpoint) came out at **2,176 tok/s** at
saturation — 69% of the A100's 3,166 on a card with half the memory bandwidth — with TPOT
15.7 ms at 1 req/s, which is the card's floor (15.2 GB of weights over 1.0 TB/s). Getting
there found a bug of mine that only a small card could: the first sweep saturated at 1,118
tok/s with ~30 sequences in flight while the worker held all 200 jobs, because the default
KV budget subtracted the weights from a `free`-memory reading taken after they were
loaded. On an 80 GB A100 that just left cache on the table (42 GB instead of 57 for the
7B); on a 24 GB card it drove the budget negative and the engine ran on its 64-block
floor, 16K tokens of cache for a trace that averages 2.17 blocks per request. Fixed
(`kv_blocks_for` in `engine.py`, CPU-tested against the 4090 numbers), and the reserve
outside the cache is now explicit — a hand-set 512 blocks on the same card OOMed at graph
capture with 71 MiB to spare, so the budget keeps 1 GiB plus one prefill chunk's MLP
activations and the largest decode batch's logits free. The queue endpoint's delivery path
then set everything else: the server's own counters put TTFT at 625 ms while the client
saw 4.27 s at 16 req/s, and the gateway's per-endpoint limit on fetching job streams
turned a 200-request burst into `Error fetching the stream: HTTP 429` in some runs and
not others — while the refused jobs still ran to completion on the worker, since
cancellation does not travel through the queue path. `deploy/runpod/README.md` has both
tables.

## Benchmarks

```bash
# in-process ablation: one command, every backend at its own minimum block size
python -m pagedserve.bench.ablation --model models/Qwen2.5-0.5B-Instruct --device cuda --dtype float16 \
  --configs naive,static,paged_torch,paged_flash,paged_flash+graphs,paged_triton,paged_triton+graphs,paged_triton+graphs+prefix \
  --trace-n 200 --request-rate 8 --shared-prefix-len 64 --block-size 16 --out results/ablation.json
python -m pagedserve.bench.ablation ... --request-rate inf --out results/ablation_sat.json      # saturation

# real-text traces (ShareGPT conversations; the synthetic trace draws random token ids)
python scripts/download_sharegpt.py                                  # ~670 MB, once
python -m pagedserve.bench.run_vllm_baseline --server pagedserve --model models/Qwen2.5-0.5B-Instruct --dtype float16 \
  --max-model-len 4096 --tokenizer models/Qwen2.5-0.5B-Instruct --sharegpt data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --server-args "--device cuda --attn-backend paged_flash --block-size 256 --enable-cuda-graphs" --name pagedserve_flash_text

# HTTP rate sweeps, vLLM then pagedserve (a different trace seed per rate; fresh servers for comparisons)
python -m pagedserve.bench.run_vllm_baseline --server vllm --vllm-bin /opt/vllm/bin/vllm \
  --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --max-model-len 4096 --rates 1,2,4,8,16,inf --trace-n 200 --name vllm
python -m pagedserve.bench.run_vllm_baseline --server pagedserve --model models/Qwen2.5-0.5B-Instruct \
  --dtype float16 --max-model-len 4096 \
  --server-args "--device cuda --attn-backend paged_triton --block-size 16 --enable-cuda-graphs" \
  --rates 1,2,4,8,16,inf --trace-n 200 --name pagedserve_triton

# kernel micro-benchmark and figures
python scripts/bench_kernels.py                  # ms/call + effective K/V GB/s, Triton vs flash ratio table
python -m pagedserve.bench.plot results/vllm.json results/pagedserve_triton.json --ablation results/ablation.json --out-dir results/plots
```

`run_vllm_baseline --base-url https://...` points the same load generator at any live
OpenAI-compatible endpoint, which is how a hosted-API row gets added.
