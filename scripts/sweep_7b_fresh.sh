#!/usr/bin/env bash
# Re-measure the README's 7B sweep rows with a different trace per rate (the committed vLLM
# sweeps replayed one trace, so its prefix cache served rates 2+ from cached prompts). Two
# fresh servers per system, rates 2/4/8/inf (16 req/s already has 4+ fresh repeats). ~15 min.
set -euo pipefail
cd "$(dirname "$0")/.."
M=models/Qwen2.5-7B-Instruct
V=/opt/vllm/bin/vllm
OUT=results/sweep_fresh
mkdir -p $OUT
echo "== old coldstart_c Qwen3 sweep, vLLM prefix-cache hit rate (same trace every rate):"
grep -o "Prefix cache hit rate: [0-9.]*%" results/qwen3/vllm_qwen3_8b.server.log | sort -u -t: -k2 -n | tail -3 || true
SA="--device cuda --attn-backend paged_flash --block-size 256 --enable-cuda-graphs"
C="--dtype float16 --max-model-len 4096 --rates 2,4,8,inf --trace-n 200 --out-dir $OUT"
for r in 1 2; do
  python -m emberserve.bench.run_vllm_baseline --server emberserve --model $M $C --server-args "$SA" --name ps_7b_r$r
  python -m emberserve.bench.run_vllm_baseline --server vllm --vllm-bin $V --model $M $C --name vllm_7b_r$r
done
grep -o "Prefix cache hit rate: [0-9.]*%" $OUT/vllm_7b_r1.server.log | sort -u -t: -k2 -n | tail -1 || true
python - <<'PY' | tee $OUT/summary.txt
import glob, json, statistics as st
for sysname in ("ps_7b", "vllm_7b"):
    by_rate = {}
    for f in sorted(glob.glob(f"results/sweep_fresh/{sysname}_r*.json")):
        for r in json.load(open(f))["runs"]:
            by_rate.setdefault(str(r["request_rate"]), []).append(r["summary"])
    for rate, xs in by_rate.items():
        print(f"{sysname:8s} rate {rate:>4} n={len(xs)}  tok/s {st.mean(x['throughput_tok_s'] for x in xs):7,.0f}  "
              f"TPOT p50 {st.mean(x['tpot_ms']['p50'] for x in xs):6.2f}  TTFT p50 {st.mean(x['ttft_ms']['p50'] for x in xs):7.1f} ms  "
              "(TPOT p50 runs " + ", ".join("%.2f" % x["tpot_ms"]["p50"] for x in xs) + ")")
PY
echo "done: git add results && git commit -qm 'results: cold start C, Qwen3-8B, fresh-trace sweeps' && git log --oneline -1"
