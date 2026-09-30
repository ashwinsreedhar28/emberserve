# Cold start

Part of [pagedserve](../README.md). Numbers come from the files in `results/` named in each section.

### Cold start: process start to first token (`results/coldstart/`)

A server that scales to zero pays its startup on every cold request, so the number that
matters is the time from starting the process to the first streamed token. Measured on one
A100 SXM pod on Sep 29 with `scripts/bench_coldstart.py` (fresh process per run, `/health`
polled every 50 ms, then one streamed 1-token completion), Qwen2.5-7B-Instruct fp16,
weights on local disk and in the page cache for every run, three runs each; vLLM 0.30.0:

| 7B, process start → first token | runs 2–3 (warm caches) | run 1 |
|---|---:|---:|
| **pagedserve** | **10.3 s** (10.0, 10.6) | 10.9 s |
| vLLM, defaults (torch.compile, piecewise + full CUDA graphs) | 67.0 s (71.7, 62.4) | 188.0 s |
| vLLM tuned per its cold-start guides (compile cache, capture sizes 1–64, Run:ai streamer) | 83.0 s (81.9, 84.1) | 101.3 s |
| vLLM `--enforce-eager` (no compile, no graphs) | 49.8 s (49.8, 49.9) | 49.1 s |

pagedserve is 4.8× faster than vLLM's fastest-booting configuration (which then serves
without CUDA graphs) and 6.5× faster than its default once vLLM's compile cache is warm;
first run against first run (vLLM compiling from scratch), 17×. Two things make the difference. pagedserve has no
compile step (its graphs capture in 1.7–1.9 s), and weights now stream: the reference
loader read the 7B's 15.2 GB at 0.68 GB/s (22.5 s) whether or not the file was in the page
cache, and vLLM's own load took 21.8–24.3 s the same way. `pagedserve/model/fastload.py`
reads the safetensors headers, plans where every byte goes, fills a ring of pinned host
buffers from parallel `preadv` threads and copies each buffer to the GPU on a side stream
while the next ones are read, casting bf16 → fp16 on the device: **1.04 s, 14.6 GB/s**
(`results/coldstart/bench_load_7b.txt`; 4–16 readers and 64–256 MB buffers all land at
12–14.7 GB/s once warm), and the 7B golden gate is ALL OK through it. The "tuned" vLLM row
is slower than the default one; the Run:ai streamer ran at its default concurrency, which
was not tuned here.

Where that 10.3 s went: pagedserve's own boot phases were ~4 s (weights 1.8–2.1 s, graph
capture 1.7–1.9 s, KV cache 0.05 s); the rest came before the engine existed — `import
torch` 1.35 s in each of the two processes, CUDA context 0.6 s, and 2.45 s loading the
tokenizer through `transformers`, all in sequence (`results/coldstart/startup_imports.txt`).
So the engine core now starts first: `pagedserve serve` parses its arguments without
importing torch and spawns the core from the raw argv (`pagedserve/server/early.py`), and
the core imports torch, loads the weights and captures its graphs while the API process
imports FastAPI and loads the tokenizer. Two pods, both A100 SXM (`scripts/coldstart_c.sh`;
pod A's JSON was lost with the pod, its console output is `results/coldstart/c_podA_console.txt`;
pod B's is `results/coldstart/c_*.json`):

| process start → first token | runs 2–3 (warm caches) | run 1 |
|---|---:|---:|
| **pagedserve, Qwen2.5-7B, core spawned first** (pod A) | **5.6 s** (5.7, 5.5) | 5.7 s |
| **pagedserve, Qwen3-8B** (pod A) | **5.5 s** (5.5, 5.5) | 5.8 s |
| **pagedserve, Qwen3-8B** (pod B) | **6.7 s** (6.9, 6.6) | 6.7 s |
| vLLM 0.30.0 defaults, Qwen3-8B (pod B) | 69.1 s (69.2, 69.0) | 202.3 s |
| vLLM `--enforce-eager`, Qwen3-8B (pod B) | 55.2 s (54.8, 55.6) | 57.0 s |

Pod A's driver (CUDA 12.8) could not run vLLM 0.30.0's cu130 torch, so the vLLM rows and
the Qwen3-8B comparison come from a second pod (driver 580, CUDA 13.0), where pagedserve
also ran and was ~1 s slower (graph capture 1.9–2.3 s against 1.4–1.6). Same pod, Qwen3-8B:
**10× faster than vLLM's default once its compile cache is warm, 8.2× faster than its
`--enforce-eager`, 30× first run against first run**. vLLM spends 24.3–25.6 s loading the
16.4 GB of weights (~0.65 GB/s) and 17 s in engine init with the compile cache warm (140 s
cold, 36 s of it torch.compile); pagedserve reads the same weights in 1.11–1.17 s
(14.0–14.8 GB/s). The 7B on pod B measured 6.9, 10.7 and 6.1 s: run 2 read its weights at
5.4 GB/s instead of 14, most likely the disk still writing back the model downloaded just
before; all three are in `results/coldstart/c_local_7b.txt`. The Qwen3-8B golden gate is
ALL OK through the streaming loader in fp16 and bf16 (`results/qwen3/golden.txt`).

What remains of pagedserve's ~6 s: the core's `import torch` and CUDA context (~2 s), its
boot (~3.5–4 s: weights 1.1 s plus the fp16 cast and setup, graphs 1.4–2.3 s), then the
first request.

#### On Runpod Serverless: pagedserve vs worker-vllm (`results/serverless_coldstart_*qwen3*`)

The same comparison where it matters to a user: a queue endpoint at zero workers, one
16-token job, Runpod's own `delayTime` (job submitted → a worker picks it up). Both
endpoints on the RTX 4090 tier ("24 GB PRO"), Qwen3-8B, max model length 4096, idle
timeout 5 s, FlashBoot off, the same night (Sep 29), from a laptop over the internet.
pagedserve v0.9.8 with the weights baked into a ~27 GB image (`deploy/runpod/Dockerfile.qwen3`);
worker-vllm v2.28.0 (vLLM 0.30.0) as Runpod's vLLM quick-deploy creates it, which downloads
the weights from Hugging Face at start (model cache off, bf16):

| Qwen3-8B, cold job on a host that has the image | `delayTime` | executionTime (16 tokens) |
|---|---:|---:|
| **pagedserve** (3 samples) | **16.2, 17.2, 17.7 s** — median 17.2 s | 1.8–2.1 s |
| worker-vllm (2 samples) | 154.3, 140.7 s — median 147.5 s | 0.5–0.6 s |
| worker-vllm on a fresh host (image pull) | 210.4 s | 0.5 s |
| pagedserve on a fresh host (first job to a new endpoint) | **328.4 s** — 317.5 s of it scheduling + pulling the 27 GB image, 7.1 s engine boot | 1.3 s |

**8.6× sooner to a working worker on a host that has the image — and 1.56× slower on
one that doesn't.** The fresh-host sample (`..._fresh_host.json`, one job sent the moment
a new endpoint began rolling out its image) spent 317.5 s before our container's first
process started: Runpod pulled the 27 GB image at ~85 MB/s, while worker-vllm fetched the
same 16 GB of weights from Hugging Face at ~760 MB/s. The engine booted in 7.1 s either
way. Baking the weights in wins on warm hosts and loses on fresh ones, which is where a
scale-out lands; the fix is a small image that fetches the weights at start (next section). One
sample, taken right after the image upload, when the registry may also have been cold. The pagedserve worker returns its own wall-clock
marks with the job (`deploy/runpod/timeline.py`, `serverless_coldstart.py --timeline`), so
its `delayTime` splits into phases (medians of three):

| phase | seconds | whose |
|---|---:|---|
| job submitted → container's first process | 3.8 (3.7–6.5) | Runpod: scheduling, container create |
| container → Python → worker `main()` | 0.8 | worker |
| `pagedserve serve` → engine `[boot]` line | 8.6 | engine: weights 1.9 s at 8.0–8.8 GB/s off the host's disk (the first cold-page-cache number; 14 GB/s from page cache on the A100 pod), graph capture 3.8–4.5 s, torch import + CUDA context ~2 s |
| boot → `/health` | 0.3 | server |
| healthy → SDK ready → first job | 3.0 | Runpod SDK: its seven fitness checks (~2 s) and the first job poll |

worker-vllm's side comes from one worker's log (`results/serverless_coldstart_vllm_qwen3_8b_worker_log.txt`,
the 154.3 s sample): 148 s inside the container, so ~6 s of scheduling and container
create. Of the 148 s, **52 s is Python startup** (worker pre-flight, then `vllm serve`,
the engine-core process and the GPU worker process each importing vLLM's dependency tree,
one after another), 21.6 s the weight download, 8.4 s loading them, **32.6 s torch.compile
from an empty cache** (the cache directory is inside the container, so every cold start
compiles from scratch; Runpod's own guide persists it to a network volume), ~20 s of graph
capture and profiling, and ~9 s of API server start, fitness checks and hand-off. Without
the download it would still be ~125 s. The gap is what vLLM does at startup, not where
the weights come from.

**The small image** (`deploy/runpod/Dockerfile.slim`: CUDA runtime base, no weights) fetches
the checkpoint at start the way worker-vllm does, but without waiting for it: config,
tokenizer and the shard index come first (2.2 s), the five shards download in the
background, and `pagedserve serve` starts at once with `PAGEDSERVE_WAIT_WEIGHTS_S`, its
streaming loader loading each shard the moment it is renamed into place
(`results/serverless_coldstart_pagedserve_slim_qwen3_8b_4090_off_v2.json`, same endpoint
settings, 40 GB container disk):

| Qwen3-8B, warm host, both fetching 16 GB from Hugging Face at start | `delayTime` | median |
|---|---|---:|
| **pagedserve, small image** | **47.4, 37.9, 32.9 s** | **37.9 s** |
| worker-vllm | 154.3, 140.7 s | 147.5 s |
| **pagedserve, small image, fresh host** (first job to a new endpoint) | **91.7 s** | — |
| worker-vllm, fresh host | 210.4 s | — |

3.9× sooner with the same starting point on a warm host, 2.3× on a fresh one. The fresh
host's 91.7 s is 68.7 s of scheduling and image pull (against 317.5 s for the 27 GB baked
image), 1.7 s for config and tokenizer, 13.9 s of download at 1.18 GB/s with the engine
starting inside it, 4.9 s after the last shard and 2 s of SDK hand-off: what is left is
almost all Runpod's. So the choice is by traffic: weights baked in when the same hosts are
reused (17 s warm, 328 s fresh), the small image when workers scale out onto new hosts
(38 s warm, 92 s fresh), and either beats worker-vllm (148 s warm, 210 s fresh). The download ran at 0.56, 0.75 and 1.02 GB/s
(29.2, 21.9 and 16.1 s; unauthenticated — an `HF_TOKEN` may steady it) and the engine's
whole startup overlapped it: after the last shard landed, 5.6–7.0 s remained (loading that
shard, then graph capture, 4.1 s). The first attempt ran out of container disk (the
default is too small for 16 GB), and the first successful one captured piecewise graphs
for 15.5 s because the engine sized the checkpoint by the files present — none yet — and
took it for a small model; it now reads the index's `total_size`. One fresh-host sample per image so far.

What is left in pagedserve's 17 s, cheapest first: the first request's executionTime is
1.8–2.1 s against vLLM's 0.5 s (most likely Triton kernels compiling on first use into an
empty cache; warm them into the image), graph capture could run in the background while
the first requests run eagerly, and 3 s is the Runpod SDK's fitness checks. FlashBoot was
accidentally on for a first worker-vllm series (`..._flashboot_on.json`): it resumed once
in three tries (0.5 s), and its misses were the full boots above — so FlashBoot's value
depends on how cheap a miss is. Not measured: more than three warm samples per engine,
more than one fresh-host sample.
