# pagedserve on Runpod Serverless

The worker (`handler.py`) builds one engine per worker at cold start and serves every job
the worker holds through the same continuous batch (`concurrency_modifier` lets a worker
take up to `MAX_CONCURRENCY` jobs at once, so scaling happens in two layers: jobs per
worker inside the engine's batch, workers per endpoint by Runpod).

```bash
# image with the model baked in (fast cold starts)
docker build -f deploy/runpod/Dockerfile --build-arg MODEL_REPO=Qwen/Qwen2.5-0.5B-Instruct \
  -t <dockerhub-user>/pagedserve-worker:0.5b .
docker push <dockerhub-user>/pagedserve-worker:0.5b
```

Endpoint settings: the image above, a 24 GB GPU (A100 80 GB for Moonlight), min workers 0,
max workers as many as the budget allows, and environment overrides if needed
(`DTYPE`, `ATTN_BACKEND`, `BLOCK_SIZE`, `MAX_MODEL_LEN`, `ASYNC_SCHEDULING=1`, ...).

```bash
# one request, aggregated
curl -s https://api.runpod.ai/v2/$ENDPOINT/runsync -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input": {"messages": [{"role": "user", "content": "Explain paged attention in one paragraph."}], "max_tokens": 128}}'
# streamed
curl -s https://api.runpod.ai/v2/$ENDPOINT/run ... | jq -r .id   # then GET /stream/<id>
```

Input fields: `prompt` | `messages` | `prompt_token_ids`, `max_tokens`, `temperature`,
`top_p`, `top_k`, `seed`, `stop`, `stop_token_ids`, `ignore_eos`, `repetition_penalty`.
Output chunks: `{"text": delta}`; the last chunk adds `finish_reason` and `usage`.

`tests/test_runpod_handler.py` runs the handler against the tiny CPU engine, so the job
format is tested without a deployment.
