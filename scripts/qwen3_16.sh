#!/usr/bin/env bash
# Qwen3-8B at 16 req/s, repeated (same pod as coldstart_c.sh). ~10 min. Lands in results/qwen3/.
# The coldstart_c sweep had pagedserve TPOT p50 21.0 ms vs vLLM 13.8 ms at 16 req/s (one run
# each). This repeats that point 3x per system, traces pagedserve's steps on the first run, and
# runs the 7B at 16 req/s once per system on the same pod to see whether the gap is Qwen3-only.
set -euo pipefail
cd "$(dirname "$0")/.."
Q=models/Qwen3-8B
M7=models/Qwen2.5-7B-Instruct
V=/opt/vllm/bin/vllm
OUT=results/qwen3
SA="--device cuda --attn-backend paged_flash --block-size 256 --enable-cuda-graphs"
C="--dtype float16 --max-model-len 4096 --rates 16 --trace-n 200 --out-dir $OUT"
for r in 1 2 3; do
  if [[ $r == 1 ]]; then T="PAGEDSERVE_STEP_TRACE=$OUT/trace_q3_16.jsonl"; else T="PAGEDSERVE_X=0"; fi
  env $T python -m pagedserve.bench.run_vllm_baseline --server pagedserve --model $Q $C \
    --server-args "$SA" --name ps_q3_16_r$r
  python -m pagedserve.bench.run_vllm_baseline --server vllm --vllm-bin $V --model $Q $C --name vllm_q3_16_r$r
done
PAGEDSERVE_STEP_TRACE=$OUT/trace_7b_16.jsonl python -m pagedserve.bench.run_vllm_baseline --server pagedserve \
  --model $M7 $C --server-args "$SA" --name ps_7b_16
python -m pagedserve.bench.run_vllm_baseline --server vllm --vllm-bin $V --model $M7 $C --name vllm_7b_16
for t in $OUT/trace_q3_16.jsonl $OUT/trace_7b_16.jsonl; do
  echo "== $t"
  python scripts/step_trace_report.py "$t" --skip 20 --json "${t%.jsonl}.analysis.json"
done | tee $OUT/report_16.txt
python - <<'PY' | tee $OUT/summary_16.txt
import glob, json, statistics as st
def row(pat):
    xs = [json.load(open(f))["runs"][0]["summary"] for f in sorted(glob.glob(f"results/qwen3/{pat}.json"))]
    m = lambda k, q: st.mean(x[k][q] for x in xs)
    print(f"{pat:16s} n={len(xs)}  tok/s {st.mean(x['throughput_tok_s'] for x in xs):7,.0f}  "
          f"TPOT p50 {m('tpot_ms','p50'):6.2f} p99 {m('tpot_ms','p99'):6.2f}  "
          f"TTFT p50 {m('ttft_ms','p50'):6.1f} p99 {m('ttft_ms','p99'):6.1f} ms  "
          "(TPOT p50 runs " + ", ".join("%.2f" % x["tpot_ms"]["p50"] for x in xs) + ")")
for p in ("ps_q3_16_r*", "vllm_q3_16_r*", "ps_7b_16", "vllm_7b_16"):
    row(p)
PY
echo "done: git add results && git commit -qm 'results: cold start C + Qwen3-8B' && git log --oneline -1"
