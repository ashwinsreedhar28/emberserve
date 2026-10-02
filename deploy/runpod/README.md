# emberserve on Runpod Serverless

The worker is a proxy in front of the real server, the layout Runpod's own `worker-vllm`
uses: `main.py` starts `emberserve serve` on localhost (engine-core process, chunked prefill,
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
docker build --platform linux/amd64 -f deploy/runpod/Dockerfile -t <dockerhub-user>/emberserve-worker:0.5b .
docker push <dockerhub-user>/emberserve-worker:0.5b
```

Qwen2.5-0.5B-Instruct is baked in by default; `--build-arg MODEL_REPO=Qwen/Qwen2.5-7B-Instruct`
for the 7B (15 GB in the image), `--build-arg MODEL_REPO=` for a model-free image that
downloads the `MODEL_REPO` env var at cold start.

## Endpoint

Image above, any 24 GB GPU (A100 80 GB for Moonlight), min workers 0, max workers as many
as the budget allows. Environment overrides, all optional: `DTYPE` (float16), `ATTN_BACKEND`
(paged_flash), `BLOCK_SIZE` (256), `MAX_MODEL_LEN` (4096), `MAX_NUM_SEQS` (256), `CUDA_GRAPHS`
(1), `PREFIX_CACHING` (0), `QUANTIZATION` (int8), `TENSOR_PARALLEL_SIZE`, `SERVED_MODEL_NAME`,
`EXTRA_SERVE_ARGS` (anything `emberserve serve` takes), `MAX_CONCURRENCY` (64).

```bash
E=<endpoint id>; K=$RUNPOD_API_KEY
# OpenAI-compatible, through Runpod's proxy (stream or not)
curl -s https://api.runpod.ai/v2/$E/openai/v1/chat/completions -H "Authorization: Bearer $K" \
  -H 'Content-Type: application/json' \
  -d '{"model":"emberserve","messages":[{"role":"user","content":"Explain paged attention in one paragraph."}],"max_tokens":128}'
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

## Small image: weights fetched at start (`Dockerfile.slim`)

Baking the weights in makes a warm host fast and a fresh host slow: with Qwen3-8B inside
(`Dockerfile.qwen3`, ~27 GB) a host that has the image starts a worker in ~17 s, but a
fresh host spent 317 s pulling it (~85 MB/s), while Hugging Face served the same 16 GB to
worker-vllm at ~760 MB/s. `Dockerfile.slim` carries no weights (CUDA runtime base, ~5–6 GB
expected) and the worker fetches `MODEL_REPO` at start (`fetch.py`): config and tokenizer
first, then the shards in the background, 8 at a time, each renamed into place when
complete, while `emberserve serve` is already starting with `EMBERSERVE_WAIT_WEIGHTS_S` set,
so the streaming loader loads each shard the moment it lands. `--timeline` then also
reports `fetch_small_files`, `download_after_spawn` and `boot_after_download`, and the
download rate.

Endpoint: GitHub repo, Dockerfile path `deploy/runpod/Dockerfile.slim` (or
`Dockerfile.slim-devel`, the same on the devel base, if the runtime base fails to build or
Triton cannot compile its launcher), env `MODEL_REPO=Qwen/Qwen3-8B` (the default), an
optional `HF_TOKEN` for Hugging Face's rate limits, and **container disk 40 GB** (the
default is too small for 16.4 GB of weights: the first try failed with "No space left on
device"). A failed download now stops the worker at once instead of leaving the engine
waiting. Measured on a warm host: delayTime 47.4 / 37.9 / 32.9 s (worker-vllm 154.3 /
140.7 s); fresh host 91.7 s (worker-vllm 210.4 s), 68.7 s of it the image pull.

Since Sep 30, with `EMBERSERVE_WAIT_WEIGHTS_S` set the engine is built *before* the
weights: the KV cache and the CUDA graphs are done while the download is still running
(`LLMEngine._from_pretrained_graphs_first`). The `[boot]` line then starts with
`build_model`, has `load_weights` after the capture, and notes "engine built before the
weights". Measured on the same endpoint: the time from the last shard to the boot line
fell from a median 5.7 s to 2.2 s (six samples), and the warm-host `delayTime` median to
32.8 s (26.5 / 29.1 / 36.4 / 36.5 s). `EMBERSERVE_GRAPHS_BEFORE_WEIGHTS=0` on the endpoint
restores the old order without a rebuild.

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

`deploy/runpod/Dockerfile.qwen3` is the same with Qwen3-8B (16.4 GB of weights, ~27 GB
image). Add `--timeline` to the command and the cold job asks the worker for its own
wall-clock marks (`deploy/runpod/timeline.py`: container start, Python start, server
spawn, the engine's `[boot]` line, healthy, SDK ready, first job), so `delayTime` comes back
split into phases: scheduling + image pull + container create, container to Python, the
engine's boot, and the SDK hand-off. The split across the client/worker boundary carries
the two clocks' skew (NTP, well under 100 ms).

then toggle FlashBoot on the endpoint and run it again with the other label. Both worker
modes log `[worker] emberserve up in X s` (container start to healthy); read it off the
worker log and pass it as `--note`. `--mode lb` does the same series against a
load-balancing endpoint by wall clock to the first byte (no delayTime there). A `/runsync`
answers `IN_QUEUE` after 90 s whatever the job is doing, so the script keeps polling
`/status/<id>` until the job completes and records the job's final `delayTime`.

### Measured (Sep 27, one RTX 4090 worker per endpoint, idle timeout 5 s)

Three cold samples per row (`results/serverless_coldstart_*.json`). A sample counts only
when `/health` reported no running worker beforehand and the worker log has a fresh
`[worker] emberserve up in X s` line for it — the container really restarted. Nothing is
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

As first measured (`results/runpod_serverless_7b_pagedserve.json`), with a KV-cache sizing
bug that is explained below:

| rate | tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
|---|---:|---:|---:|
| 1 req/s | 171 | 749 ms / 5.95 s | 15.8 / 18.3 ms |
| 4 | 606 | 1.39 / 6.65 s | 18.5 / 25.0 ms |
| 16 (6.3 served) | 1,085 | 6.58 / 13.8 s | 22.4 / 30.0 ms |
| inf | 1,118 | 13.4 / 24.8 s | 22.7 / 26.7 ms |

TPOT 15.8 ms at 1 req/s is the card: 15.2 GB of fp16 weights over the 4090's 1.0 TB/s is
15 ms per decode step. TTFT is the queue endpoint's job dispatch (prefill is ~50 ms), as
with the 0.5B. The saturation number was wrong by 2×: 1,118 tok/s at 22.7 ms per token is
~30 sequences in flight, and the worker log shows the SDK holding all 200 jobs, so the
engine was admitting 30. The default KV budget subtracted the weights from a `free`
memory reading taken *after* they were loaded; on an 80 GB card that only cost some cache
(7B: 42 GB instead of 57), on a 24 GB card it drove the budget negative and the engine
fell to its 64-block floor — 16K tokens, and this trace averages 2.17 blocks per request,
so 29.5 of them at a time. Fixed in `emberserve/engine.py` (`kv_blocks_for`, with a CPU
test). The other edge of the same card: a hand-set `EXTRA_SERVE_ARGS=--num-blocks 512`
(7.0 GiB of cache) OOMed at the bucket-256 graph capture with 71 MiB left — the container
sees the 4090 as 22.04 GiB, and 14.18 GiB of weights plus the cache left nothing for the
graph mempool. So the reserve is now explicit (`activation_reserve_bytes`: 1 GiB plus a
prefill chunk's MLP activations and the largest decode batch's logits, 1.36 GiB for the
7B), which sizes the 7B's cache on this card at ~380 blocks (97K tokens). Released in v0.9.4: with no override the 7B endpoint booted at 410 blocks (105K tokens) on the host it landed on, through graph capture — the budget follows each host's free memory.

Corrected, same endpoint with `--num-blocks 380` (what the fixed default computes for this
card; `/metrics` reported 379 after the graph scratch block) —
`results/runpod_serverless_7b_pagedserve_kv380*.json`:

| rate | tok/s | TTFT p50 / p99 | TPOT p50 / p99 | ok |
|---|---:|---:|---:|---:|
| 1 req/s | 172 | 474 ms / 5.56 s | 15.7 / 17.2 ms | 200/200 |
| 2 | 317 | 1.26 / 5.73 s | 16.4 / 18.6 ms | 200/200 |
| 4 | 546 | 2.41 / 5.98 s | 18.1 / 21.5 ms | 198/200 |
| 8 (5.7 served) | 941 | 3.38 / 6.42 s | 20.3 / 32.7 ms | 190/200 |
| 16 (10.8–11.2 served) | 1,179 · 1,599 (rerun) | 4.27 / 7.79 s | 26.0 / 47.9 ms | 200/200 |
| inf, 64 in flight | 1,088 | 3.83 / 6.52 s | 21.1 / 44.0 ms | 200/200 |
| inf, 128 in flight | 787 | 3.14 / 6.66 s | 28.5 / 56.2 ms | 173/200 |
| inf, unbounded (200) | **2,176** | 3.10 / 9.86 s | 33.2 / 51.1 ms | 200/200 |

Rows with failed requests are over the run's whole duration, failures included (547, 947
and 791 tok/s before the harness counted a failure's time; the saved runs' `wall_s`).

2,176 tok/s is 1.95× the bugged run and 69% of the A100's 3,166 on a card with half the
memory bandwidth. The rest of the table is the queue endpoint's delivery path, and three
things in it are worth knowing before serving a 7B this way:

- **TTFT is the platform's.** The server's own counters over the rate-16 run put TTFT at
  a mean of 625 ms from arrival; the client saw a p50 of 4.27 s. The 3.6 s between them is
  job dispatch plus the gateway. The engine's TPOT over the same window, with 100–200
  sequences in the batch, averaged 31 ms — the 4090 running the 7B at a real batch, which
  the client only sees at the unbounded burst.
- **The stream fetch is rate-limited.** The OpenAI route on a queue endpoint fetches each
  job's stream on the client's behalf, and that fetch has a per-endpoint limit: the
  client gets a `200` whose body is `Error fetching the stream: HTTP 429`, not SSE (the
  load generator now records that text instead of "empty stream"). It hit 27 of 200
  streams at 128 in flight, all 200 within a second in two earlier unbounded runs, and
  none in the run above — the gateway's bucket, not the concurrency, decides. The 2 and
  10 failures at 4 and 8 req/s were not recorded and are most likely the same.
- **A refused stream is still a job.** After the client had written off a burst, the
  worker reported `requests_running: 98`: the jobs were queued, dispatched and run to their
  full length (`ignore_eos`) with nobody listening, because cancellation does not travel
  through the queue path. Bounding the client's concurrency (64 here) avoids the 429 but
  halves the throughput, since every slot then spends most of its life in dispatch.

A load-balancing endpoint has none of this (the 0.5B measurements above); the 7B was not
built as one.

## Benchmark it

```bash
python -m emberserve.bench.run_vllm_baseline --base-url https://api.runpod.ai/v2/$E/openai/v1 \
  --completions-path /completions --no-health --api-key "$RUNPOD_API_KEY" \
  --model emberserve --tokenizer models/Qwen2.5-0.5B-Instruct --max-model-len 4096 \
  --rates 1,2,4,8,inf --trace-n 100 --name runpod_serverless_pagedserve
```

`tests/test_runpod_handler.py` runs the proxy handler against the real app on the tiny CPU
engine, so the job contract is tested without a deployment.

Runpod builds the image from a GitHub *release*; a push alone does not rebuild, and a release
on a commit that already has a build record is skipped, so a failed registry push (seen once:
`Manifest upload failed: 500`) needs a new commit plus a new release to rebuild.
