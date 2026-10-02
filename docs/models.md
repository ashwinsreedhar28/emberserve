# Models

Part of [emberserve](../README.md). Numbers come from the files in `results/` named in each section.

Four dense families run through the same decoder block (`model/qwen2.py`), with
`ModelConfig` carrying the differences: `qwen2` (attention bias, rope_theta 1e6), `qwen3`
(no bias, per-head RMSNorm on q and k before RoPE, an explicit `head_dim`), `llama`
and `mistral` (no bias, list-valued eos ids, Llama 3's RoPE frequency scaling). The
DeepSeek-V2/V3 family (`model/deepseek.py`: multi-head latent attention + mixture of
experts) is its own block; see [Latent attention and MoE](design.md#latent-attention-and-moe). The
golden gate is run per model on the A100 (`golden/<model>/`), the profile is
`scripts/profile_step.py` at batch 1, and the sweeps are the same 200-request trace with
prompt ids drawn from each model's own vocabulary.

| model | arch | golden vs HF | batch-1 forward | weight-read floor | saturation tok/s, emberserve / vLLM | TPOT p50 @ 8 req/s |
|---|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | qwen2, 24L, GQA 14/2, D=64 | exact (fp32), all tokens (fp16) | 1.9 ms | ~0.9 ms | **16,635 / 16,269 (102%)** ⁰ | **2.0** / 2.1 ms |
| Qwen2.5-7B-Instruct | qwen2, 28L, GQA 28/4, D=128 | all tokens (fp16; TP2: 6/7 exact, one tie-break) | 10.1 ms (TP2: 7.65) | ~10 ms | **3,412 / 3,415 (100%)** ⁴; TP2 on 2× A100: 4,485 / 4,949 (91%) ⁵ | 12.35 / 11.98 ms ⁴ (TP2: 9.8 / 7.3 ⁵) |
| Qwen3-8B | qwen3, 36L, GQA 32/8, D=128, per-head q/k RMSNorm | all tokens (fp16: one tie-break, logits 4.0e-2; bf16: two, 4.4e-1) | – | ~11 ms | **2,839 / 2,972 (96%)** ⁶ | 12.71 / 11.98 ms @ 4 req/s; 20.90 / 20.98 @ 16 ⁶ |
| DeepSeek-R1-Distill-Llama-8B | llama (3.1), 32L, GQA 32/8, D=128, llama3 rope | all tokens (fp16) | 10.8 ms | ~11 ms | **2,797 / 2,823 (99%)** ¹ ⁵ | 14.9 / 12.2 ms ⁵ |
| Moonshot Moonlight-16B-A3B-Instruct | deepseek_v3, 27L, MLA (kv_lora 512 + rope 64), 64 experts top-6 + 2 shared, 3B active | all tokens (bf16; 2–3 tie-breaks inside noise, both paths) | 6.2 ms ² | ~3 ms (3B active + 0.7 GB lm_head) | **2,697 / 3,223 (84%)** ³ ⁵ | 18.8 / 13.4 ms ³ ⁵ |

⁰ v9, mean of three saturation repeats against vLLM's single run; on real text (six repeats) 22,964 vs 23,111 (99%). Both servers are limited by their API process at this size.
¹ `results/pagedserve_r1_8b_flash_v7.json` (CUDA defaults: engine process, chunked prefill, async scheduling); the first measurement, prefill-priority and in-process, was 2,519 (89%) with 19.0 ms TPOT at 8 req/s. TPOT at 1 req/s 11.1 vs 10.8 ms, 20.6 vs 14.0 at 16 req/s: the same chunked-prefill tail as at 7B.
² whole decode step at batch 1 (`mla_triton` + fused MoE + CUDA graphs): 63.6 ms with the per-expert loop, 9.4 with the grouped GEMM, 7.2 after the routing/alignment kernels, 6.2 after the split-K fix; vLLM's TPOT at 1 req/s is 7.1 ms, ours over HTTP 7.6. See [Moonlight](results.md#moonlight-mla--moe-on-the-a100).
³ chunked prefill (2048-token cap) + async scheduling (`results/pagedserve_moonlight_v3.json`); the first run, prefill-priority and synchronous, was 2,457 (76%) and 26.3 ms.
⁴ fresh servers, a trace per rate, mean of two, Sep 29 (`results/sweep_fresh/`). The old sweep (`results/pagedserve_7b_flash_v7.json` vs `results/vllm_7b.json`) had 3,166 / 3,188 and 12.6 / 10.6 ms, vLLM's side partly served from its prefix cache ([correction](results.md#a-correction-the-sweeps-replayed-one-trace-and-vllm-cached-it)).
⁵ vLLM's number comes from a sweep that replayed one trace into its prefix cache; not yet re-measured, so the gap is overstated by an unknown amount.
⁶ mean of three fresh servers each, same pod, vLLM 0.30.0 (`results/qwen3/`). emberserve's saturation repeats spread more (2,709–2,909 vs 2,958–2,985).

## Hosted APIs, for scale (a footnote)

The same load generator, the same ShareGPT prompts and the same client, pointed at
OpenAI-compatible endpoints instead of a local server (`--hosted --base-url ...`;
`results/openrouter_*.json`, `results/hosted_*.json`). Hosted runs stop at the model's own
EOS, so their tok/s is how much the model chose to say, not capacity, and is left out.
Where two days disagree the range is shown. List prices are OpenRouter's on Sep 27, 2026;
the self-hosted rows convert the A100's $1.59/hr into a per-output-token price at the
measured throughput (prompt tokens ride free inside the hour).

| endpoint | TTFT p50 | TPOT p50 | $/M output tokens | what it is |
|---|---:|---:|---:|---|
| emberserve, Qwen2.5-0.5B, one A100 | 9 ms | 1.8–2.0 ms | $0.03 at saturation (16.6k tok/s) | this repo |
| emberserve, Qwen2.5-7B, one A100 | 39 ms | 10.2 ms | $2.50 at 1 req/s · $0.22 at 16 req/s · $0.14 at saturation (3,166 tok/s) | this repo |
| vLLM, Qwen3-8B, Runpod Serverless | 770–930 ms | 8.2–8.7 ms | per-second GPU billing | Runpod's `worker-vllm`, through their proxy; a 100-stream burst at one H100 worker (`MAX_CONCURRENCY` 64) measured TTFT p50 **63.6 s** while TPOT stayed at 8 ms (queueing before the first token) |
| DeepSeek V4.1 Flash, OpenRouter | 530–1,070 ms | 4.1–6.9 ms | $0.29 (in: $0.035) | 50/50 at 1, 2 and 4 req/s |
| Claude Haiku 4.5, OpenRouter | 940 ms | 7.8 ms | $5 (in: $1) | 20 requests at 0.3 req/s |
| Kimi K3, OpenRouter | 880–1,010 ms | 5.8–14.0 ms | $9 (in: $1) | **rate-limited**: `new-account-rpm` 429s above ~0.5 req/s (36/50 at 1 req/s, 0/50 at 2, 26/30 at 0.5) |

Two things the row for a rented A100 says. Per token, the cheapest frontier-class API
($0.29/M) sits between the 7B at 16 req/s and the 7B at saturation, so self-hosting a 7B
beats it on price only when the card stays above roughly half load, and what you get for
it is a 7B. And the APIs' first token arrives 15–25× later than ours (they queue,
batch and route at a scale where 500 ms is fine) while their per-token time is *lower*
than our fp16 7B's 10.2 ms: 4–7 ms is what a large deployment gets from bigger batches
on bigger hardware, and what this repo reaches only with int8 (6.6 ms) or two GPUs
(7.4 ms). A hosted API also brings a rate limit you do not control, which is the only
reason the Kimi column has three numbers of completed requests in it.

At 7–8B the decode step is the weight read: 15–16 GB of fp16 at ~1.5 TB/s is 10–11 ms, and
both engines land there at batch 1. The CPU-side costs that decide the 0.5B result are
~10% of the step at this size, and moving the engine to its own process changed nothing
measurable at 7B (2,832 → 2,859 tok/s). What did matter at 7B was *scheduling*: a 7B
prefill of a 270-token prompt is ~30 ms of compute-bound work, and prefill-priority runs
one for every arrival while every decoder waits, so TPOT at 16 req/s was 24.8 ms against
vLLM's 11.5 (a vLLM number that turned out to be partly its prefix cache; fresh, it is
16.05 ms). Chunked prefill (decode rows and a prompt chunk in one step) brings that to
17.6 ms and saturation throughput to 97% of vLLM with a 2048-token cap (3,092 vs 3,188
tok/s, the same 112 ms p99 tail vLLM shows); a 512-token cap trades 6% of that throughput
for a 42 ms tail. The first chunked run measured 2x *slower*: the mixed-step attention path
padded every sequence's queries to the chunk length (100k padded queries per layer at 200
decodes plus one chunk); the fix batches the decode rows and pads only the chunk rows.
Async scheduling then took it to 99% (3,166 tok/s) and 16.5 ms at 16 req/s
(`results/pagedserve_7b_flash_v7.json`) — against the Sep 27 vLLM run's 11.5 ms, a gap
that did not survive re-measurement ([below](results.md#the-7b-tail-re-measured)): vLLM's number was
served partly from its prefix cache
([correction](results.md#a-correction-the-sweeps-replayed-one-trace-and-vllm-cached-it)). Piecewise CUDA graphs, the fix for the same
eager mixed steps at 0.5B, *lose* 1% here and lengthen the tail (19.9 ms at 16 req/s): a
7B chunk is compute-bound, so padding it up to a token bucket costs real FLOPs, whereas at
0.5B the launches it removes were the whole cost. Hence the default is by size (piecewise
below 4 GB). Finer token buckets (`--piecewise-bucket-step 256`) recover the saturation
loss (3,168 tok/s) but not the tail (18.3 ms at 16 req/s), so the padding was only part
of the cost. Any `model_type: qwen2 | qwen3 | llama | mistral | deepseek_v2 | deepseek_v3` snapshot loads with
`scripts/download_model.py --repo <hf repo>`; DeepSeek-R1-Distill-Qwen, Mistral-7B and
DeepSeek-V2-Lite are the same code paths as the rows above.

![7B TPOT vs offered load](../results/plots/7b/tpot_vs_rate.png)
