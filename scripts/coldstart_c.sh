#!/usr/bin/env bash
# Cold start, phase C + Qwen3-8B (one A100 SXM pod, after `bash scripts/pod_setup.sh`). ~40 min.
# Everything lands in results/coldstart/ (c_*) and results/qwen3/. Commit both afterwards.
#   1. Qwen3-8B golden gate (fp16 and bf16) through the streaming loader
#   2. process start -> first token: emberserve 7B with the early core (vs 10.3 s before),
#      emberserve Qwen3-8B, vLLM Qwen3-8B default and --enforce-eager
#   3. Qwen3-8B sweep, emberserve vs vLLM, 1/4/16 req/s + saturation
set -euo pipefail
cd "$(dirname "$0")/.."
Q=models/Qwen3-8B
M7=models/Qwen2.5-7B-Instruct
V=/opt/vllm/bin/vllm
mkdir -p results/coldstart results/qwen3
# vLLM's torch (cu130) needs a CUDA 13 driver (>= 580); a 12.8 pod fails every vLLM run.
/opt/vllm/bin/python -c "import torch; torch.cuda.init(); print('vllm torch', torch.__version__, 'sees', torch.cuda.get_device_name())" \
  || { nvidia-smi | head -4; echo "vLLM's torch can't use this driver: deploy a pod with CUDA 13.0"; exit 1; }
[[ -f $Q/config.json ]] || python scripts/download_model.py --repo Qwen/Qwen3-8B
[[ -f $M7/config.json ]] || python scripts/download_model.py --repo Qwen/Qwen2.5-7B-Instruct
/opt/vllm/bin/pip show vllm 2>/dev/null | head -2 > results/qwen3/vllm_version.txt || true
[[ -d golden/Qwen3-8B ]] || python scripts/dump_golden.py --model $Q --out golden/Qwen3-8B --device cuda
for dt in float16 bfloat16; do
  echo "== $dt"
  python scripts/check_golden.py --model $Q --golden golden/Qwen3-8B --device cuda --dtype $dt \
    --backends paged_flash --block-size 256 2>&1 | tail -9
done | tee results/qwen3/golden.txt
python scripts/bench_coldstart.py --model $M7 --repeats 3 --system emberserve \
  --out results/coldstart/c_local_7b.json | tee results/coldstart/c_local_7b.txt
python scripts/bench_coldstart.py --model $Q --repeats 3 --system emberserve --system vllm \
  --system vllm_eager --vllm-bin $V --out results/coldstart/c_local_qwen3_8b.json | tee results/coldstart/c_local_qwen3_8b.txt
R="--rates 1,4,16,inf --trace-n 200 --max-model-len 4096 --out-dir results/qwen3"
python -m emberserve.bench.run_vllm_baseline --server emberserve --model $Q --dtype float16 $R \
  --server-args "--device cuda --attn-backend paged_flash --block-size 256 --enable-cuda-graphs" --name pagedserve_qwen3_8b
python -m emberserve.bench.run_vllm_baseline --server vllm --vllm-bin $V --model $Q --dtype float16 $R --name vllm_qwen3_8b
python - <<'PY' | tee results/qwen3/summary.txt
import json
for name in ("pagedserve_qwen3_8b", "vllm_qwen3_8b"):
    d = json.load(open(f"results/qwen3/{name}.json"))
    for r in d["runs"]:
        s = r["summary"]
        print(f"{name:22s} rate {str(r['request_rate']):>4}  {s['throughput_tok_s']:8,.0f} tok/s  "
              f"TPOT p50 {s['tpot_ms']['p50']:6.2f} ms  TTFT p50 {s['ttft_ms']['p50']:7.1f} ms  ok {s['completed']}/{s['num_requests']}")
PY
echo "done: git add results/coldstart results/qwen3 && git commit -m 'results: cold start C + Qwen3-8B'"
