"""Cold-start series against a Runpod Serverless endpoint (load-balancing or the queue's
/openai route): let the endpoint scale to zero, send one small request and time it to the
first byte and to completion, then one warm request right after; repeat.

    python scripts/serverless_coldstart.py --base-url https://<id>.api.runpod.ai \\
        --api-key "$RUNPOD_API_KEY" --repeats 5 --idle-s 90 --label flashboot_on \\
        --out results/serverless_coldstart_7b_4090_flashboot_on.json

`--idle-s` must exceed the endpoint's idle timeout (set the endpoint to 5 s for this).
Cold time = wall clock from the request being sent to the first response byte; it includes
worker scheduling, image start, model load and graph capture, and the gateway's own wait.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import httpx


def one_request(client: httpx.Client, path: str, body: dict, timeout_s: float) -> dict:
    t0 = time.perf_counter()
    first = None
    text = b""
    with client.stream("POST", path, json=body, timeout=timeout_s) as r:
        for chunk in r.iter_bytes():
            if first is None and chunk:
                first = time.perf_counter()
            text += chunk
        status = r.status_code
    t1 = time.perf_counter()
    return {"status": status, "ttfb_s": (first or t1) - t0, "total_s": t1 - t0,
            "bytes": len(text), "ok": status == 200}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--path", default="/v1/completions")
    ap.add_argument("--model", default="pagedserve")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--idle-s", type=float, default=90.0, help="wait before each cold request")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    body = {"model": args.model, "prompt": "The capital of France is", "max_tokens": args.max_tokens,
            "ignore_eos": True, "stream": True}
    runs = []
    with httpx.Client(base_url=args.base_url, headers={"Authorization": f"Bearer {args.api_key}"}) as c:
        for i in range(args.repeats):
            print(f"[coldstart] waiting {args.idle_s:.0f} s for the endpoint to scale to zero", file=sys.stderr)
            time.sleep(args.idle_s)
            cold = one_request(c, args.path, body, args.timeout_s)
            warm = one_request(c, args.path, body, args.timeout_s)
            runs.append({"cold": cold, "warm": warm})
            print(f"[coldstart] {i + 1}/{args.repeats}: cold ttfb {cold['ttfb_s']:.1f} s "
                  f"(total {cold['total_s']:.1f} s, status {cold['status']}); "
                  f"warm ttfb {warm['ttfb_s'] * 1e3:.0f} ms total {warm['total_s'] * 1e3:.0f} ms",
                  file=sys.stderr)
    cold_ok = [r["cold"]["ttfb_s"] for r in runs if r["cold"]["ok"]]
    warm_ok = [r["warm"]["total_s"] for r in runs if r["warm"]["ok"]]
    summary = {
        "cold_ttfb_s": {"n": len(cold_ok), "min": min(cold_ok, default=None),
                        "median": st.median(cold_ok) if cold_ok else None, "max": max(cold_ok, default=None)},
        "warm_total_s": {"n": len(warm_ok), "median": st.median(warm_ok) if warm_ok else None},
        "failures": sum(1 for r in runs if not r["cold"]["ok"]),
    }
    out = {"kind": "serverless_coldstart", "label": args.label, "base_url": args.base_url,
           "path": args.path, "model": args.model, "max_tokens": args.max_tokens,
           "idle_s": args.idle_s, "runs": runs, "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[coldstart] {args.label or args.base_url}: cold ttfb median "
          f"{summary['cold_ttfb_s']['median'] and round(summary['cold_ttfb_s']['median'], 1)} s "
          f"(min {summary['cold_ttfb_s']['min'] and round(summary['cold_ttfb_s']['min'], 1)}, "
          f"max {summary['cold_ttfb_s']['max'] and round(summary['cold_ttfb_s']['max'], 1)}), "
          f"warm {summary['warm_total_s']['median'] and round(summary['warm_total_s']['median'] * 1e3)} ms, "
          f"failures {summary['failures']}; wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
