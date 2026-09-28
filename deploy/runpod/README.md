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

## Benchmark it

```bash
python -m pagedserve.bench.run_vllm_baseline --base-url https://api.runpod.ai/v2/$E/openai/v1 \
  --completions-path /completions --no-health --api-key "$RUNPOD_API_KEY" \
  --model pagedserve --tokenizer models/Qwen2.5-0.5B-Instruct --max-model-len 4096 \
  --rates 1,2,4,8,inf --trace-n 100 --name runpod_serverless_pagedserve
```

`tests/test_runpod_handler.py` runs the proxy handler against the real app on the tiny CPU
engine, so the job contract is tested without a deployment.
