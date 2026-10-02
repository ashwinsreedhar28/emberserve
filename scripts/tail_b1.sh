#!/usr/bin/env bash
# Phase B1 of the 7B-tail investigation (one A100 SXM pod, after `bash scripts/pod_setup.sh`).
# ~25 min of GPU time; everything lands in results/tail/. Commit that directory afterwards.
#   1. mixed-step attention microbenchmark (ours vs one paged varlen call)
#   2. emberserve at 16 and 8 req/s with the per-step trace (non-syncing index copies, the default)
#   3. the same 16 req/s run with EMBERSERVE_LEGACY_SYNC_COPIES=1 (the code as of v0.9.4)
#   4. a short 16 req/s run under torch's sync debug mode: every synchronizing call, by site
#   5. vLLM at 16 req/s once on this pod (baseline check; its server log shows its token budget)
#   6. step_trace_report on every trace
set -euo pipefail
cd "$(dirname "$0")/.."
M=models/Qwen2.5-7B-Instruct
[[ -f $M/config.json ]] || python scripts/download_model.py --repo Qwen/Qwen2.5-7B-Instruct
OUT=results/tail
mkdir -p "$OUT"
SA="--device cuda --attn-backend paged_flash --block-size 256 --enable-cuda-graphs"
run_ps() {  # name rate trace-n [ENV=...]
  local name=$1 rate=$2 n=$3; shift 3
  env "$@" EMBERSERVE_STEP_TRACE="$OUT/trace_$name.jsonl" \
    python -m emberserve.bench.run_vllm_baseline --server emberserve --model $M --dtype float16 \
      --max-model-len 4096 --server-args "$SA" --rates "$rate" --trace-n "$n" --out-dir "$OUT" --name "ps_$name"
}
python scripts/bench_mixed_attn.py --decode 32 --ctx 600 --chunk 270 | tee "$OUT/bench_mixed_attn.txt"
python scripts/bench_mixed_attn.py --decode 64 --ctx 800 --chunk 512 | tee -a "$OUT/bench_mixed_attn.txt"
run_ps 16 16 200
run_ps 8 8 200
run_ps 16_legacy 16 200 EMBERSERVE_LEGACY_SYNC_COPIES=1
run_ps 16_syncdebug 16 60 EMBERSERVE_SYNC_DEBUG=1
run_ps 16_legacy_syncdebug 16 60 EMBERSERVE_SYNC_DEBUG=1 EMBERSERVE_LEGACY_SYNC_COPIES=1
python -m emberserve.bench.run_vllm_baseline --server vllm --vllm-bin /opt/vllm/bin/vllm --model $M \
  --dtype float16 --max-model-len 4096 --rates 16 --trace-n 200 --out-dir "$OUT" --name vllm_16
grep -iE "max_num_batched_tokens|chunked|max_num_seqs|cuda graph|compil" "$OUT/vllm_16.server.log" | head -20 \
  > "$OUT/vllm_16_config.txt" || true
for t in "$OUT"/trace_*.jsonl; do
  echo "== $t"
  python scripts/step_trace_report.py "$t" --skip 20 --json "${t%.jsonl}.analysis.json"
done | tee "$OUT/report.txt"
echo "done: git add results/tail && git commit -m 'results: 7B tail B1 traces' && git push (via the Mac fetch)"
