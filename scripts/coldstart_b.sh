#!/usr/bin/env bash
# Cold start, phase B (one A100 SXM pod, after `bash scripts/pod_setup.sh`). ~30 min.
# Everything lands in results/coldstart/. Commit that directory afterwards.
#   1. golden gate through the streaming loader (0.5B and 7B, fp16: the 7B's bf16 weights
#      take the cast-on-GPU path)
#   2. weight loading alone, 7B: reference loader vs streaming loader, threads x buffer sweep
#   3. process start -> first token, 7B, 3 runs each: emberserve, vLLM default, vLLM tuned
#      (compile cache + trimmed graph sizes [+ Run:ai streamer]), vLLM --enforce-eager
# Weights are on local disk and in the page cache for every run (the pod downloads them
# first); a Serverless container's first read may be colder — that is measured separately.
set -euo pipefail
cd "$(dirname "$0")/.."
M=models/Qwen2.5-7B-Instruct
OUT=results/coldstart
mkdir -p "$OUT"
[[ -f $M/config.json ]] || python scripts/download_model.py --repo Qwen/Qwen2.5-7B-Instruct
nproc > "$OUT/nproc.txt"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> "$OUT/nproc.txt"
python scripts/check_golden.py --model models/Qwen2.5-0.5B-Instruct --device cuda --dtype float16 \
  --backends paged_flash --block-size 256 2>&1 | tail -5 | tee "$OUT/golden_stream_loader.txt"
[[ -d golden/Qwen2.5-7B-Instruct ]] || python scripts/dump_golden.py --model $M --out golden/Qwen2.5-7B-Instruct --device cuda
python scripts/check_golden.py --model $M --golden golden/Qwen2.5-7B-Instruct --device cuda --dtype float16 \
  --backends paged_flash --block-size 256 2>&1 | tail -5 | tee -a "$OUT/golden_stream_loader.txt"
python scripts/bench_load.py --model $M --device cuda --dtype float16 --repeats 2 \
  --threads 4,8,16 --buffer-mb 64,256 | tee "$OUT/bench_load_7b.txt"
RUNAI=""
if /opt/vllm/bin/pip install -q runai-model-streamer >/dev/null 2>&1 \
   && /opt/vllm/bin/python -c "import runai_model_streamer" 2>/dev/null; then RUNAI="--runai"; fi
echo "runai streamer: ${RUNAI:-not available}" | tee "$OUT/runai.txt"
/opt/vllm/bin/pip show vllm 2>/dev/null | head -2 > "$OUT/vllm_version.txt" || true
python scripts/bench_coldstart.py --model $M --repeats 3 --system emberserve --system vllm \
  --system vllm_tuned --system vllm_eager --vllm-bin /opt/vllm/bin/vllm $RUNAI \
  --out "$OUT/local_7b.json" | tee "$OUT/local_7b.txt"
echo "done: git add results/coldstart && git commit -m 'results: cold start B'"
