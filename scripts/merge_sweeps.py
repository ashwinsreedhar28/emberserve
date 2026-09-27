"""Merge two rate sweeps of the same system into one results file, split at a rate.

    python scripts/merge_sweeps.py --low results/pagedserve_flash_v6_low.json \
        --high results/pagedserve_flash_v6.json --split 8 --out results/pagedserve_flash_final.json \
        --note "v6_low ran rates 1-4 on 94bdca3, v6 ran 8-inf on the same commit"

Runs with `request_rate < split` come from `--low`, the rest (including the unbounded
`inf` rate, stored as null) from `--high`. The merged file records where each half came
from under `derived_from`, so the README's figures stay traceable to the raw sweeps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", required=True, help="sweep JSON providing rates below --split")
    ap.add_argument("--high", required=True, help="sweep JSON providing rates at/above --split and inf")
    ap.add_argument("--split", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="`system` field of the merged file (default: <out stem>)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    low = json.loads(Path(args.low).read_text())
    high = json.loads(Path(args.high).read_text())
    runs = ([r for r in low["runs"] if r["request_rate"] is not None and r["request_rate"] < args.split]
            + [r for r in high["runs"] if r["request_rate"] is None or r["request_rate"] >= args.split])
    runs.sort(key=lambda r: (r["request_rate"] is None, r["request_rate"] or 0.0))
    merged = dict(high)
    merged["system"] = args.name or Path(args.out).stem
    merged["runs"] = runs
    merged["derived_from"] = {f"rates<{args.split:g}": Path(args.low).name,
                              f"rates>={args.split:g}": Path(args.high).name, "note": args.note}
    Path(args.out).write_text(json.dumps(merged, indent=1))
    rates = ["inf" if r["request_rate"] is None else f"{r['request_rate']:g}" for r in runs]
    print(f"wrote {args.out}: rates {', '.join(rates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
