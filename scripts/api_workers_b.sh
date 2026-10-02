#!/usr/bin/env bash
# Item 2, phase B (one A100 SXM pod, after `bash scripts/pod_setup.sh`): does a second API
# process lift the 0.5B saturation point, for emberserve and (if it has the knob) vLLM?
# ~30 min of pod time; everything lands in results/apiw/. Commit that directory afterwards.
#   1. the API layer alone on this pod's CPU (fake core, 400k tok/s offered), 1/2/4 workers
#   2. emberserve 0.5B saturation, 3 repeats per server lifetime, synthetic trace and
#      ShareGPT text, --api-workers 1 / 2 / 4
#   3. vLLM, same traces, one API server and (if `--api-server-count` exists) two
set -euo pipefail
cd "$(dirname "$0")/.."
M=models/Qwen2.5-0.5B-Instruct
OUT=results/apiw
mkdir -p "$OUT"
[[ -f data/ShareGPT_V3_unfiltered_cleaned_split.json ]] || python scripts/download_sharegpt.py
nproc | tee "$OUT/nproc.txt"
CP=6   # load-generator processes, the same for every run
SAT="--rates inf,inf,inf --trace-n 200 --client-procs $CP --max-model-len 4096"
TXT="--tokenizer $M --sharegpt data/ShareGPT_V3_unfiltered_cleaned_split.json"
S="--device cuda --attn-backend paged_flash --block-size 256 --enable-cuda-graphs"
for n in 1 2 4; do
  python scripts/bench_api_layer.py --api-workers $n --repeats 3 --client-procs 4 --step-ms 0.5 --max-tokens 512
done 2>&1 | grep -E "fake core|run " | tee "$OUT/api_layer.txt"
for n in 1 2 4; do
  python -m emberserve.bench.run_vllm_baseline --server emberserve --model $M --dtype float16 $SAT \
    --server-args "$S --api-workers $n" --out-dir "$OUT" --name ps_sat_w$n
  python -m emberserve.bench.run_vllm_baseline --server emberserve --model $M --dtype float16 $SAT $TXT \
    --server-args "$S --api-workers $n" --out-dir "$OUT" --name ps_text_w$n
done
V=/opt/vllm/bin/vllm
python -m emberserve.bench.run_vllm_baseline --server vllm --vllm-bin $V --model $M --dtype float16 $SAT \
  --out-dir "$OUT" --name vllm_sat_a1
python -m emberserve.bench.run_vllm_baseline --server vllm --vllm-bin $V --model $M --dtype float16 $SAT $TXT \
  --out-dir "$OUT" --name vllm_text_a1
if $V serve --help=all 2>/dev/null | grep -q -- "--api-server-count"; then
  echo "vLLM has --api-server-count" | tee "$OUT/vllm_api_server_count.txt"
  python -m emberserve.bench.run_vllm_baseline --server vllm --vllm-bin $V --model $M --dtype float16 $SAT \
    --server-args "--api-server-count 2" --out-dir "$OUT" --name vllm_sat_a2
  python -m emberserve.bench.run_vllm_baseline --server vllm --vllm-bin $V --model $M --dtype float16 $SAT $TXT \
    --server-args "--api-server-count 2" --out-dir "$OUT" --name vllm_text_a2
else
  echo "vLLM has no --api-server-count" | tee "$OUT/vllm_api_server_count.txt"
fi
/opt/vllm/bin/pip show vllm 2>/dev/null | head -2 > "$OUT/vllm_version.txt" || true
python - <<'PY'
import json, glob, statistics as st
for f in sorted(glob.glob("results/apiw/*.json")):
    d = json.load(open(f)); runs = d["runs"]
    tps = [r["summary"]["throughput_tok_s"] for r in runs]
    srv = [r.get("server_latency", {}).get("ttft_ms_mean") for r in runs]
    print(f"{f.split('/')[-1]:24s} tok/s {' '.join(f'{t:8,.0f}' for t in tps)}  mean {st.mean(tps):8,.0f}  "
          f"server TTFT mean {' '.join('-' if s is None else f'{s:.0f}' for s in srv)} ms")
PY
