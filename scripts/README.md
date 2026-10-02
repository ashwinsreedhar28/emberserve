# scripts

Part of [emberserve](../README.md). Every script has a docstring or header with its usage;
this page says which one to reach for. Run them from the repo root.

## Setup

| script | what it does |
|---|---|
| `pod_setup.sh` | one-shot setup of a fresh GPU pod: flash-attn, the package, a model, the golden files, vLLM in its own venv ([GPU notes](../docs/gpu.md)) |
| `download_model.py` | an HF snapshot into `models/` (weights, config, tokenizer) |
| `download_sharegpt.py` | the ShareGPT dump the real-text traces sample from |
| `gpu_smoke.py` | every backend must print the same greedy tokens on the GPU; then a tok/s table |

## Correctness

| script | what it does |
|---|---|
| `dump_golden.py` | Hugging Face reference outputs and logits → `golden/<model>/` |
| `check_golden.py` | the gate: emberserve against the golden files on every backend, with the fp16/bf16 tie-break rule |

## Benchmarks

| script | what it does |
|---|---|
| `python -m emberserve.bench.run_vllm_baseline` | rate sweeps over HTTP against vLLM, emberserve or any OpenAI endpoint (a module, not a script) |
| `bench_coldstart.py` | process start → first token, emberserve against vLLM on one machine |
| `serverless_coldstart.py` | Runpod Serverless cold starts; `--timeline` splits `delayTime` into phases |
| `bench_load.py` | weight loading, reference path against the streaming loader |
| `bench_api_layer.py` | tokens/s the API process can deliver with no model behind it |
| `bench_kernels.py` | decode-attention kernels: paged_torch vs paged_flash vs paged_triton |
| `bench_mixed_attn.py` | mixed-step attention at 7B shapes: two calls vs one paged varlen call |
| `bench_moe.py` | the fused MoE layer at Moonlight's geometry |

## Profiling and tracing

| script | what it does |
|---|---|
| `profile_step.py` | where a decode step's wall time goes, per phase (`--kernels` for per-kernel) |
| `profile_mixed_step.py` | a mixed prefill + decode step, kernel by kernel |
| `step_trace_report.py` | reads `EMBERSERVE_STEP_TRACE`: per-step GPU time, idle gaps, syncs |
| `stall_report.py` | reads `EMBERSERVE_STEP_LOG`: where the time went at saturation |
| `pyspy_summary.py` | summarizes a `py-spy record --format raw` file |
| `gpu_debug_capture.py` | reproduces a CUDA-graph capture failure with `CUDA_LAUNCH_BLOCKING=1` |

## Pod sessions

Each of these is one GPU session, recorded so the numbers it produced can be re-run as-is.
They start with `cd` to the repo root and write into `results/`.

| script | session | results |
|---|---|---|
| `tail_b1.sh` | the 7B tail at 16 req/s: step traces, sync hunt, vLLM baseline | `results/tail/` |
| `api_workers_b.sh` | a second API process in front of the engine core | `results/apiw/` |
| `coldstart_b.sh` | cold start phase B: streaming loader, vLLM configs | `results/coldstart/` |
| `coldstart_c.sh` | cold start phase C + Qwen3-8B golden, cold start and sweep | `results/coldstart/c_*`, `results/qwen3/` |
| `qwen3_16.sh` | Qwen3-8B at 16 req/s, repeated, with a step trace | `results/qwen3/` |
| `qwen3_sat.sh` | Qwen3-8B at 4 req/s and saturation, fresh servers | `results/qwen3/` |
| `sweep_7b_fresh.sh` | the 7B sweep re-measured with a trace per rate | `results/sweep_fresh/` |

## Results housekeeping

| script | what it does |
|---|---|
| `merge_sweeps.py` | merge two sweeps of one system, split at a rate |
| `redact_results.py` | blank an `api_key` an old sweep JSON recorded |
