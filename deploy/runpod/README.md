# pagedserve on Runpod Serverless

The worker is a proxy in front of the real server, the layout Runpod's own `worker-vllm`
uses: `main.py` starts `pagedserve serve` on localhost (engine-core process, chunked prefill,
async scheduling, CUDA graphs — the CLI's CUDA defaults, i.e. exactly the server the
benchmarks ran against) and forwards every job to it. Runpod wraps a request to
`https://api.runpod.ai/v2/<endpoint>/openai/v1/...` as a job with `openai_route` and
`openai_input`, so the endpoint is OpenAI-compatible and the benchmark client works
against it unchanged. Jobs per worker share one continuous batch (`concurrency_modifier`
= `MAX_CONCURRENCY`, default 64); workers per endpoint are Runpod's scaling.

## Build

Either let Runpod build it (Serverless → New endpoint → GitHub repo, branch `main`,
Dockerfile path `deploy/runpod/Dockerfile`), or locally:

```bash
docker build --platform linux/amd64 -f deploy/runpod/Dockerfile -t <dockerhub-user>/pagedserve-worker:0.5b .
docker push <dockerhub-user>/pagedserve-worker:0.5b
```

Qwen2.5-0.5B-Instruct is baked in by default; `--build-arg MODEL_REPO=Qwen/Qwen2.5-7B-Instruct`
for the 7B (15 GB in the image), `--build-arg MODEL_REPO=` for a model-free image that
downloads the `MODEL_REPO` env var at cold start.

## Endpoint

Image above, any 24 GB GPU (A100 80 GB for Moonlight), min workers 0, max workers as many
as the budget allows. Environment overrides, all optional: `DTYPE` (float16), `ATTN_BACKEND`
(paged_flash), `BLOCK_SIZE` (256), `MAX_MODEL_LEN` (4096), `MAX_NUM_SEQS` (256), `CUDA_GRAPHS`
(1), `PREFIX_CACHING` (0), `QUANTIZATION` (int8), `TENSOR_PARALLEL_SIZE`, `SERVED_MODEL_NAME`,
`EXTRA_SERVE_ARGS` (anything `pagedserve serve` takes), `MAX_CONCURRENCY` (64).

```bash
E=<endpoint id>; K=$RUNPOD_API_KEY
# OpenAI-compatible, through Runpod's proxy (stream or not)
curl -s https://api.runpod.ai/v2/$E/openai/v1/chat/completions -H "Authorization: Bearer $K" \
  -H 'Content-Type: application/json' \
  -d '{"model":"pagedserve","messages":[{"role":"user","content":"Explain paged attention in one paragraph."}],"max_tokens":128}'
# the job API, shorthand input
curl -s https://api.runpod.ai/v2/$E/runsync -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
  -d '{"input": {"prompt": "Once upon a time", "sampling_params": {"max_tokens": 32}}}'
```

Job input shapes: `{"openai_route", "openai_input"}` (what the proxy sends), `{"route",
"body", "method"}` for any server route (`/v1/models`, `/metrics`, `/health`), or the
shorthand `{"prompt" | "messages", "sampling_params", "stream"}`. With `"stream": true`
the server's SSE bytes are relayed as they arrive; otherwise the JSON response comes back
whole.

## Measured (Sep 27, one RTX 4090 worker, 0.5B, 200-request trace)

| endpoint | TPOT p50 @1–4 req/s | TTFT p50 @4 req/s | saturation tok/s | TTFT p50/p99 @inf |
|---|---:|---:|---:|---:|
| Queue, per-yield (first build) | 56–231 ms | 4.4 s | 95 (100 failures) | 37 s / 82 s |
| Queue, `STREAM_FLUSH_MS=100` | 1.4–1.5 ms | 2.4 s | 9,162 | 1.5 s / 2.6 s |
| Load balancer, request count 4 (default) | 1.7–1.8 ms | 220 ms | 1,013 | 13 s / 34 s |
| Load balancer, request count 200 | 1.9–2.0 ms | 230 ms | 11,079 | 0.82 s / 1.07 s |

Server-side (from `/metrics`, load-balancer runs) the engine was identical throughout:
TTFT 8–11 ms, TPOT 1.7–2.0 ms. For a load-balancer endpoint set the endpoint's
**request count** to the concurrency the server should hold (200 here): at the default
the balancer admits about four requests per worker and queues the rest.

## 7B image and the cold-start series

`deploy/runpod/Dockerfile.7b` bakes Qwen2.5-7B-Instruct instead (~25 GB image); point a
new endpoint's Dockerfile path at it. A 24 GB GPU holds the fp16 weights (15 GB) plus a
~6 GB KV cache. `scripts/serverless_coldstart.py` measures cold starts the way
worker-vllm's were measured: on a **queue** endpoint, Runpod's own `delayTime` and
`executionTime` per `/runsync`, and `/health` polled to zero workers before every sample
(a parked worker is not a cold start). Set the endpoint's idle timeout to 5 s, then

```bash
python scripts/serverless_coldstart.py --mode queue --endpoint $E7 --api-key "$RUNPOD_API_KEY" \
  --repeats 3 --idle-s 60 --label 7b_4090_flashboot_on --image "Qwen2.5-7B, ~25 GB image, weights baked in" \
  --out results/serverless_coldstart_7b_4090_flashboot_on.json
```

then toggle FlashBoot on the endpoint and run it again with the other label. Both worker
modes log `[worker] pagedserve up in X s` (container start to healthy); read it off the
worker log and pass it as `--note`. `--mode lb` does the same series against a
load-balancing endpoint by wall clock to the first byte (no delayTime there). A `/runsync`
answers `IN_QUEUE` after 90 s whatever the job is doing, so the script keeps polling
`/status/<id>` until the job completes and records the job's final `delayTime`.

### Measured (Sep 27, one RTX 4090 worker per endpoint, idle timeout 5 s)

Three cold samples per row (`results/serverless_coldstart_*.json`). A sample counts only
when `/health` reported no running worker beforehand and the worker log has a fresh
`[worker] pagedserve up in X s` line for it — the container really restarted. Nothing is
downloaded at start: both images carry the weights (`Dockerfile`: Qwen2.5-0.5B-Instruct,
9.8 GB; `Dockerfile.7b`: Qwen2.5-7B-Instruct, ~25 GB).

| image | FlashBoot | Runpod `delayTime` per cold sample | container start → healthy (worker log) |
|---|---|---:|---:|
| 0.5B, 9.8 GB | on | 87.1 s (fresh host) / 25.9 / 22.8 s | 26.1 / 17.9 / 16.3 s |
| 0.5B, 9.8 GB | off | 29.2 / 25.1 / 29.3 s | 22.1 / 17.4 / 20.1 s |
| 7B, ~25 GB | on | 22.6 s / ≈95 s (fresh host) / 0.85 s (container resumed, not restarted) | 12.3 / 14.5 / — s |
| 7B, ~25 GB | off | 24.1 s / 65.6 s (host GPU taken, `throttled`) / 27.4 s (new worker, other host) | 15.6 / 14.5 / 15.3 s |

`delayTime` is Runpod's own number: scheduling, the image pull when the host does not
have it, container start, the boot above, and ~2 s of the SDK's fitness checks. On a host
that already holds the image it runs 8–12 s over the boot line; a fresh host adds the pull,
roughly 50 s for the 9.8 GB image and 70 s for the 25 GB one (that 7B sample is the one
that outlived the 90 s `/runsync` cap; its wall time is read off the worker log). The
third kind of wait is the host itself: a stopped container holds no GPU, and hosts are
shared with pods, so `/health` reported the 7B worker as `throttled` (its host's 4090 in
use by someone else) before two of the FlashBoot-off samples. The first cleared at once
(8.5 s over the boot); the second waited 49 s for the GPU before the container could even
start; by the third Runpod had re-homed the slot to another host (a new worker id, a
different driver) and, since it did so during the wait between samples, that host already
had the image and the sample cost 12 s over its boot. The 7B image
boots *faster* than the 0.5B one: below 4 GB of weights the CLI turns piecewise CUDA
graphs on and captures a graph per token bucket on top of the full-step ones, while the
7B worker captures only the full-step graphs and spends the time on 15 GB of weights.
Host variance is real: one 0.5B worker earlier in the day took 155.6 s from container
start to healthy (108 s inside a model load and graph capture that takes ~16 s
elsewhere), on the same image. FlashBoot's paused container came back once in six
FlashBoot-on samples (0.85 s, same worker, three minutes after its previous job); the
other five restarted the container, so with FlashBoot on the median cold start was no
better than with it off — it is a "may resume", and worth leaving on because it costs
nothing when it doesn't.

### 7B throughput (Queue endpoint, one RTX 4090, `STREAM_FLUSH_MS=100`, 200-request trace)

| rate | tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
|---|---:|---:|---:|
| 1 req/s | 171 | 749 ms / 5.95 s | 15.8 / 18.3 ms |
| 2 | 331 | 1.19 / 6.26 s | 16.3 / 19.0 ms |
| 4 | 606 | 1.39 / 6.65 s | 18.5 / 25.0 ms |
| 8 (5.3 served) | 919 | 5.38 / 8.56 s | 21.7 / 27.2 ms |
| 16 (6.3 served) | 1,085 | 6.58 / 13.8 s | 22.4 / 30.0 ms |
| inf | 1,118 | 13.4 / 24.8 s | 22.7 / 26.7 ms |

`results/runpod_serverless_7b_pagedserve.json`, measured with a KV-cache sizing bug
(below). TPOT 15.8 ms at 1 req/s is the card: 15.2 GB of fp16 weights over the 4090's
1.0 TB/s is 15 ms per decode step. TTFT is the queue endpoint's job dispatch (prefill is
~50 ms), as with the 0.5B. The saturation number is wrong by about 3×: 1,118 tok/s at
22.7 ms per token is ~30 sequences in flight, and the worker log shows the SDK holding
all 200 jobs, so the engine was admitting 30. The default KV budget subtracted the
weights from a `free` memory reading taken *after* they were loaded; on an 80 GB card
that only cost some cache (7B: 42 GB instead of 57), on a 24 GB card it drove the budget
negative and the engine fell to its 64-block floor — 16K tokens, and this trace averages
2.17 blocks per request, so 29.5 of them at a time. Fixed in `pagedserve/engine.py`
(`kv_blocks_for`, with a CPU test). The other edge of the same card: a hand-set
`EXTRA_SERVE_ARGS=--num-blocks 512` (7.0 GiB of cache) OOMed at the bucket-256 graph
capture with 71 MiB left — the container sees the 4090 as 22.04 GiB, and 14.18 GiB of
weights plus the cache left nothing for the graph mempool. So the reserve is now explicit
(`activation_reserve_bytes`: 1 GiB plus a prefill chunk's MLP activations and the largest
decode batch's logits, 1.36 GiB for the 7B), which sizes the 7B's cache on this card at
380 blocks (97K tokens, ~175 of this trace's requests in flight). The corrected row is
pending a rebuild.

## Benchmark it

```bash
python -m pagedserve.bench.run_vllm_baseline --base-url https://api.runpod.ai/v2/$E/openai/v1 \
  --completions-path /completions --no-health --api-key "$RUNPOD_API_KEY" \
  --model pagedserve --tokenizer models/Qwen2.5-0.5B-Instruct --max-model-len 4096 \
  --rates 1,2,4,8,inf --trace-n 100 --name runpod_serverless_pagedserve
```

`tests/test_runpod_handler.py` runs the proxy handler against the real app on the tiny CPU
engine, so the job contract is tested without a deployment.

Runpod builds the image from a GitHub *release*; a push alone does not rebuild, and a release
on a commit that already has a build record is skipped, so a failed registry push (seen once:
`Manifest upload failed: 500`) needs a new commit plus a new release to rebuild.
