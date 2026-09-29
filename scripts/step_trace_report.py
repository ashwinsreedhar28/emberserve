"""Read a PAGEDSERVE_STEP_TRACE file and say where the per-token time goes.

    python scripts/step_trace_report.py results/trace_7b_16rps.jsonl [--skip 50] [--json out.json]

Prints the boot phases, then one row per step kind (count, share, mean batch shape, host
phases, GPU time, the GPU idle gap before the step), then the decomposition the 7B-tail
question needs:

    engine TPOT   mean over emitted decode tokens of the step they rode in, (GPU time +
                  the idle gap before it): what a decoding request waits per token inside
                  the engine, before the API process
    = decode floor (the same average if every step were a decode-only step of its batch)
    + prompt excess (what steps carrying a prompt chunk added), split into GPU work and
      GPU idle gaps
    + other gaps (idle GPU before decode-only steps)

and the excess per prompt chunk admitted. With PAGEDSERVE_SYNC_DEBUG=1 it also lists the
synchronizing calls per step kind. On a CPU trace (no GPU fields) the host launch +
resolve time stands in for the step time.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> tuple[dict, list[dict]]:
    boot: dict = {}
    steps: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # a killed writer's last partial line
            if "boot" in d:
                boot = d["boot"]
            else:
                steps.append(d)
    return boot, steps


def step_time(d: dict) -> float:
    """GPU busy + idle-before, or the host time on a CPU trace."""
    if d.get("gpu_ms") is not None:
        return d["gpu_ms"] + (d.get("gpu_gap_ms") or 0.0)
    return d.get("host_launch_ms", 0.0) + d.get("host_resolve_ms", 0.0)


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def decode_floor_by_batch(steps: list[dict]) -> dict[int, float]:
    """Median decode-only step time per batch size, to price a mixed step's decode rows."""
    by_n: dict[int, list[float]] = defaultdict(list)
    for d in steps:
        if d["kind"].startswith("decode"):
            by_n[d["n_decode"]].append(d["gpu_ms"] if d.get("gpu_ms") is not None else step_time(d))
    return {n: st.median(v) for n, v in by_n.items() if v}


def floor_for(n: int, table: dict[int, float]) -> float | None:
    if not table:
        return None
    if n in table:
        return table[n]
    keys = sorted(table)
    lo = max((k for k in keys if k <= n), default=None)
    hi = min((k for k in keys if k >= n), default=None)
    if lo is None:
        return table[hi]
    if hi is None:
        return table[lo]
    if hi == lo:
        return table[lo]
    w = (n - lo) / (hi - lo)
    return table[lo] * (1 - w) + table[hi] * w


def analyse(steps: list[dict]) -> dict:
    kinds: dict[str, list[dict]] = defaultdict(list)
    for d in steps:
        kinds[d["kind"]].append(d)
    rows = {}
    for k, ds in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        def m(key: str) -> float | None:
            vals = [d[key] for d in ds if d.get(key) is not None]
            return st.mean(vals) if vals else None
        gaps = [d["gpu_gap_ms"] for d in ds if d.get("gpu_gap_ms") is not None]
        rows[k] = {
            "count": len(ds), "share": len(ds) / len(steps),
            "n_decode": m("n_decode"), "n_prefill_tokens": m("n_prefill_tokens"),
            "host_sched_ms": m("host_sched_ms"), "host_build_ms": m("host_build_ms"),
            "host_launch_ms": m("host_launch_ms"), "host_resolve_ms": m("host_resolve_ms"),
            "gpu_ms": m("gpu_ms"), "gpu_ms_p90": pct([d["gpu_ms"] for d in ds if d.get("gpu_ms") is not None], 0.9),
            "gpu_gap_ms": st.mean(gaps) if gaps else None, "gpu_gap_ms_p90": pct(gaps, 0.9),
        }
    floors = decode_floor_by_batch(steps)
    tok = sum(d["n_decode"] for d in steps)
    engine_tpot = sum(d["n_decode"] * step_time(d) for d in steps) / tok if tok else None
    floor_sum = excess_gpu = excess_gap = other_gap = 0.0
    prompt_steps = prompt_chunks = 0
    for d in steps:
        f = floor_for(d["n_decode"], floors)
        if f is None:
            f = step_time(d)
        n = d["n_decode"]
        floor_sum += n * f
        gap = d.get("gpu_gap_ms") or 0.0
        busy = d["gpu_ms"] if d.get("gpu_ms") is not None else step_time(d)
        if d["n_prefill_tokens"]:
            prompt_steps += 1
            prompt_chunks += d["n_prefill_seqs"]
            excess_gpu += n * max(0.0, busy - f)
            excess_gap += n * gap
        else:
            other_gap += n * gap
    out = {"steps": len(steps), "kinds": rows, "decode_tokens": tok, "engine_tpot_ms": engine_tpot}
    if tok:
        out["decomposition_ms_per_token"] = {
            "decode_floor": floor_sum / tok,
            "prompt_steps_gpu_excess": excess_gpu / tok,
            "prompt_steps_gpu_gap": excess_gap / tok,
            "decode_steps_gpu_gap": other_gap / tok,
        }
    mixed = [d for d in steps if d["n_prefill_tokens"] and d["n_decode"]]
    if mixed:
        per = []
        for d in mixed:
            f = floor_for(d["n_decode"], floors)
            if f is not None:
                per.append(step_time(d) - f)
        if per:
            out["mixed_step_excess_ms"] = {"mean": st.mean(per), "p50": pct(per, 0.5), "p90": pct(per, 0.9)}
            out["excess_per_prompt_chunk_ms"] = sum(per) / max(1, sum(d["n_prefill_seqs"] for d in mixed))
    syncs: dict[str, Counter] = defaultdict(Counter)
    for d in steps:
        for s in d.get("syncs", []):
            syncs[d["kind"]][s] += 1
    if syncs:
        out["syncs"] = {k: dict(c.most_common(8)) for k, c in syncs.items()}
        out["syncs_per_step"] = {k: sum(c.values()) / len(kinds[k]) for k, c in syncs.items()}
    out["prompt_steps"] = prompt_steps
    out["prompt_chunks"] = prompt_chunks
    return out


def fmt(v: float | None, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--skip", type=int, default=0, help="drop the first N steps (warm-up)")
    ap.add_argument("--json", default=None, help="also write the analysis here")
    args = ap.parse_args()
    boot, steps = load(Path(args.trace))
    steps = steps[args.skip:]
    if boot:
        print("boot: " + " · ".join(f"{k.removesuffix('_s')} {v:.2f} s" for k, v in boot.items()))
    if not steps:
        print("no steps")
        return
    a = analyse(steps)
    print(f"{a['steps']} steps, {a['decode_tokens']} decode tokens, {a['prompt_chunks']} prompt chunks "
          f"in {a['prompt_steps']} steps")
    hdr = ("kind", "count", "share", "dec", "pre_tok", "sched", "build", "launch", "resolve",
           "gpu", "gpu_p90", "gap", "gap_p90")
    print("{:<24}{:>7}{:>7}{:>6}{:>8}{:>7}{:>7}{:>8}{:>8}{:>7}{:>8}{:>6}{:>8}".format(*hdr))
    for k, r in a["kinds"].items():
        print("{:<24}{:>7}{:>7}{:>6}{:>8}{:>7}{:>7}{:>8}{:>8}{:>7}{:>8}{:>6}{:>8}".format(
            k, r["count"], f"{r['share']:.0%}", fmt(r["n_decode"], 0), fmt(r["n_prefill_tokens"], 0),
            fmt(r["host_sched_ms"]), fmt(r["host_build_ms"]), fmt(r["host_launch_ms"]),
            fmt(r["host_resolve_ms"]), fmt(r["gpu_ms"]), fmt(r["gpu_ms_p90"]),
            fmt(r["gpu_gap_ms"]), fmt(r["gpu_gap_ms_p90"])))
    if a.get("engine_tpot_ms") is not None:
        dcmp = a["decomposition_ms_per_token"]
        print(f"engine TPOT {a['engine_tpot_ms']:.2f} ms = decode floor {dcmp['decode_floor']:.2f}"
              f" + prompt-step GPU excess {dcmp['prompt_steps_gpu_excess']:.2f}"
              f" + prompt-step idle gaps {dcmp['prompt_steps_gpu_gap']:.2f}"
              f" + decode-step idle gaps {dcmp['decode_steps_gpu_gap']:.2f}")
    if "mixed_step_excess_ms" in a:
        m = a["mixed_step_excess_ms"]
        print(f"mixed step over a decode-only step of its batch: mean {m['mean']:.2f} ms, p50 "
              f"{m['p50']:.2f}, p90 {m['p90']:.2f}; per prompt chunk {a['excess_per_prompt_chunk_ms']:.2f} ms")
    for k, c in a.get("syncs", {}).items():
        print(f"syncs in {k} ({a['syncs_per_step'][k]:.2f}/step): " + ", ".join(f"{s} x{n}" for s, n in c.items()))
    if args.json:
        Path(args.json).write_text(json.dumps({"boot": boot, **a}, indent=1))


if __name__ == "__main__":
    main()
