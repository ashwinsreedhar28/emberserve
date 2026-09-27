"""Read a PAGEDSERVE_STEP_LOG and say where the time went at saturation.

    python scripts/stall_report.py results/steps_sat.tsv [--threshold-ms 15]

Prints step-duration and inter-step-gap percentiles, the worst stalls (a step, or the
gap before it, longer than the threshold) with the GC pauses of the core and API
processes that overlap them, and each process's GC pause totals by generation.
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path


def read_tsv(path: Path) -> list[list[float]]:
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            rows.append([float(x) for x in line.split("\t")])
    return rows


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--threshold-ms", type=float, default=15.0)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    path = Path(args.log)
    steps = read_tsv(path)
    if not steps:
        raise SystemExit("empty step log")
    gcs = {}
    for role in ("core", "api"):
        p = Path(f"{path}.gc-{role}")
        if p.exists():
            gcs[role] = read_tsv(p)
    t0 = steps[0][0]
    span = steps[-1][0] + steps[-1][1] - t0
    durs = [s[1] * 1e3 for s in steps]
    gaps = [(steps[i][0] - (steps[i - 1][0] + steps[i - 1][1])) * 1e3 for i in range(1, len(steps))]
    busy = sum(s[1] for s in steps)
    tokens = sum(int(s[3]) for s in steps)
    print(f"{len(steps)} steps over {span:.2f} s: stepping {busy / span:.1%} of the time, "
          f"{tokens} tokens ({tokens / span:.0f} tok/s inside the core)")
    print(f"step duration ms: p50 {pct(durs, .5):.2f}  p90 {pct(durs, .9):.2f}  p99 {pct(durs, .99):.2f}  "
          f"max {max(durs):.2f}  mean {st.mean(durs):.2f}")
    print(f"gap between steps ms: p50 {pct(gaps, .5):.2f}  p90 {pct(gaps, .9):.2f}  p99 {pct(gaps, .99):.2f}  "
          f"max {max(gaps):.2f}  total {sum(gaps) / 1e3:.2f} s")
    by_n: dict[int, list[float]] = {}
    for s, d in zip(steps, durs):
        by_n.setdefault(int(s[2]) // 25 * 25, []).append(d)
    print("step ms by running sequences (bucket of 25): " + "  ".join(
        f"{k}-{k + 24}: {st.median(v):.2f} (n={len(v)})" for k, v in sorted(by_n.items())))
    for role, rows in gcs.items():
        tot = {}
        for _, d, g in rows:
            tot.setdefault(int(g), [0, 0.0])
            tot[int(g)][0] += 1
            tot[int(g)][1] += d * 1e3
        print(f"GC pauses in the {role} process: " + "  ".join(
            f"gen{g}: {c} runs, {ms:.0f} ms total, max {max(d * 1e3 for _, d, gg in rows if int(gg) == g):.1f} ms"
            for g, (c, ms) in sorted(tot.items())))

    stalls = []
    for i, s in enumerate(steps):
        if durs[i] > args.threshold_ms:
            stalls.append((durs[i], s[0], s[0] + s[1], "step", int(s[2])))
        if i > 0 and gaps[i - 1] > args.threshold_ms:
            prev_end = steps[i - 1][0] + steps[i - 1][1]
            stalls.append((gaps[i - 1], prev_end, s[0], "gap", int(s[2])))
    stalls.sort(reverse=True)
    print(f"\n{len(stalls)} stalls over {args.threshold_ms:g} ms; worst {args.top}:")
    for ms, a, b, kind, n in stalls[:args.top]:
        overl = []
        for role, rows in gcs.items():
            small = 0
            for gt, gd, gg in rows:
                if gt < b and gt + gd > a - 0.002:
                    if gd * 1e3 >= 1.0:
                        overl.append(f"{role} gen{int(gg)} {gd * 1e3:.1f} ms")
                    else:
                        small += 1
            if small:
                overl.append(f"{role}: {small} sub-ms gen0 runs")
        print(f"  t+{a - t0:7.3f}s  {kind:4s} {ms:7.1f} ms  running={n:3d}  "
              + (", ".join(overl) if overl else "no GC pause overlapping"))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:  # `| head`
        pass
