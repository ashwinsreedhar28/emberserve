"""Summarize a `py-spy record --format raw` file: where a process spends its samples.

    py-spy record --pid <api pid> --duration 15 --rate 250 --format raw --nonblocking -o results/pyspy_api.txt
    python scripts/pyspy_summary.py results/pyspy_api.txt [--top 25]

Raw lines are `frame;frame;...;frame count` (root first). Prints self time (the leaf
frame) and inclusive time (any frame) per function, as a share of all samples, plus the
share of samples whose stack passes through a few landmarks (the SSE route, the
detokenizer, the pipe reader, JSON encoding).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

LANDMARKS = {
    "sse route (stream)": "stream",
    "sse_starlette": "sse_starlette",
    "json.dumps": "dumps",
    "detokenizer": "detok",
    "pipe recv / unpickle": "recv",
    "_deliver": "_deliver",
    "uvicorn write": "httptools",
    "pydantic": "pydantic",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    self_t: Counter = Counter()
    incl: Counter = Counter()
    marks: Counter = Counter()
    total = 0
    for line in Path(args.raw).read_text().splitlines():
        line = line.strip()
        if not line or " " not in line:
            continue
        stack, _, count = line.rpartition(" ")
        try:
            n = int(count)
        except ValueError:
            continue
        frames = stack.split(";")
        total += n
        self_t[frames[-1]] += n
        for f in set(frames):
            incl[f] += n
        for label, needle in LANDMARKS.items():
            if any(needle in f for f in frames):
                marks[label] += n
    if not total:
        raise SystemExit("no samples")
    print(f"{total} samples")
    print("\nself time (leaf frame):")
    for f, n in self_t.most_common(args.top):
        print(f"  {100 * n / total:5.1f}%  {f}")
    print("\ninclusive (function anywhere on the stack):")
    for f, n in incl.most_common(args.top):
        print(f"  {100 * n / total:5.1f}%  {f}")
    print("\nlandmarks (share of samples whose stack passes through):")
    for label, n in marks.most_common():
        print(f"  {100 * n / total:5.1f}%  {label}")


if __name__ == "__main__":
    main()
