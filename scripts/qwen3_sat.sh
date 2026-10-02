#!/usr/bin/env bash
# Qwen3-8B at 4 req/s and saturation, 3 fresh servers per system, a different trace per rate
# (the default since the harness fix: vLLM's prefix cache otherwise serves every rate after the
# first from cached prompts). ~12 min. Lands in results/qwen3/.
set -euo pipefail
cd "$(dirname "$0")/.."
Q=models/Qwen3-8B
V=/opt/vllm/bin/vllm
OUT=results/qwen3
SA="--device cuda --attn-backend paged_flash --block-size 256 --enable-cuda-graphs"
C="--dtype float16 --max-model-len 4096 --rates 4,inf --trace-n 200 --out-dir $OUT"
echo "== vLLM prefix-cache hit rate in the coldstart_c sweep (same trace at every rate):"
grep -o "Prefix cache hit rate: [0-9.]*%" $OUT/vllm_qwen3_8b.server.log | sort -u -t: -k2 -n | tail -3 || true
for r in 1 2 3; do
  python -m emberserve.bench.run_vllm_baseline --server emberserve --model $Q $C --server-args "$SA" --name ps_q3_sat_r$r
  python -m emberserve.bench.run_vllm_baseline --server vllm --vllm-bin $V --model $Q $C --name vllm_q3_sat_r$r
done
echo "== vLLM prefix-cache hit rate with a trace per rate (should be ~0):"
grep -o "Prefix cache hit rate: [0-9.]*%" $OUT/vllm_q3_sat_r1.server.log | sort -u -t: -k2 -n | tail -3 || true
python - <<'PY' | tee $OUT/summary_sat.txt
import glob, json, statistics as st
for sysname in ("ps_q3_sat", "vllm_q3_sat"):
    files = sorted(glob.glob(f"results/qwen3/{sysname}_r*.json"))
    by_rate = {}
    for f in files:
        for r in json.load(open(f))["runs"]:
            by_rate.setdefault(str(r["request_rate"]), []).append(r["summary"])
    for rate, xs in by_rate.items():
        print(f"{sysname:12s} rate {rate:>4} n={len(xs)}  tok/s {st.mean(x['throughput_tok_s'] for x in xs):7,.0f}  "
              f"TPOT p50 {st.mean(x['tpot_ms']['p50'] for x in xs):6.2f}  TTFT p50 {st.mean(x['ttft_ms']['p50'] for x in xs):7.1f} ms  "
              "(tok/s runs " + ", ".join("%.0f" % x["throughput_tok_s"] for x in xs) + ")")
PY
echo "done: git add results && git commit -qm 'results: cold start C + Qwen3-8B' && git log --oneline -1"
