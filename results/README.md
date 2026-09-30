# results

Part of [pagedserve](../README.md). Every number in the docs comes from a file here; this
page maps the files to the sections that use them. JSON files from the sweep runner carry
the full arguments, the per-rate summaries and, for newer runs, the per-request records.

| files | what they are | used in |
|---|---|---|
| `serverless_coldstart_*` | Runpod Serverless cold starts: `delayTime`, per-phase timelines, worker-vllm's worker log | [Cold start on Serverless](../docs/cold-start.md#on-runpod-serverless-pagedserve-vs-worker-vllm-resultsserverless_coldstart_qwen3) |
| `coldstart/` | process start → first token on A100 pods, pagedserve and vLLM; loader benchmarks | [Cold start](../docs/cold-start.md#cold-start-process-start-to-first-token-resultscoldstart) |
| `sweep_fresh/`, `qwen3/` | fresh-server sweeps with a trace per rate: 7B and Qwen3-8B against vLLM; Qwen3-8B golden | [The correction](../docs/results.md#a-correction-the-sweeps-replayed-one-trace-and-vllm-cached-it) |
| `tail/` | the 7B at 16 req/s: step traces, sync hunt, mixed-step attention | [The 7B tail](../docs/results.md#the-7b-tail-re-measured) |
| `apiw/` | one, two and four API processes against vLLM with one and two API servers | [Two API processes](../docs/results.md#two-api-processes---api-workers) |
| `vllm*.json`, `pagedserve_*.json`, `*_sweep.log` | the rate sweeps over HTTP, every engine version (`pagedserve_flash_v2`…`v9`), 7B, R1-8B, Moonlight, ShareGPT text, int8, TP2, speculation | [Every sweep](../docs/results.md#a100-pagedserve-over-http-vs-vllm-resultsvllmjson-resultspagedserve_json), [Models](../docs/models.md#models) |
| `profile_*.json` | per-phase and per-kernel step profiles | [The gap against vLLM](../docs/results.md#the-gap-against-vllm) |
| `steps_*.tsv*` | per-step logs at saturation (engine-core stalls, GC) | [The gap against vLLM](../docs/results.md#the-gap-against-vllm) |
| `ablation*.json`, `ablation_a100.log` | in-process ablations: every backend, graphs, prefix caching | [RTX 4090 ablation](../docs/results.md#rtx-4090-ablation-resultsablationjson-block-size-256-torch-280cu128) |
| `kernels*.json` | decode-attention kernel micro-benchmarks | [Decode-attention kernel](../docs/results.md#a100-sxm-80-gb-decode-attention-kernel-resultskernels_a100_json) |
| `runpod_serverless_*` | throughput on Runpod Serverless (queue and load-balancer endpoints) | [Running on Serverless](../docs/running.md#on-runpod-serverless) |
| `hosted_*`, `openrouter_*` | the same load generator against hosted APIs | [Hosted APIs](../docs/models.md#hosted-apis-for-scale-a-footnote) |
| `plots/` | figures made by `python -m pagedserve.bench.plot` from the files above | throughout |
| `debug_capture.log`, `pytest_triton.log` | debugging records kept for reference | [GPU notes](../docs/gpu.md) |

vLLM sweep files written before Sep 29 replayed one trace at every rate on one server, so
vLLM's prefix cache served its later rates in part; newer runs record a `trace_seed` per
rate. See [the correction](../docs/results.md#a-correction-the-sweeps-replayed-one-trace-and-vllm-cached-it).
