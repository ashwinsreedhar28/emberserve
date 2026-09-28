"""Cold-start series against a Runpod Serverless endpoint.

Queue endpoint (comparable to worker-vllm's numbers: Runpod's own `delayTime` and
`executionTime` come back with every `/runsync` response, and `/health` says how many
workers exist, so a sample is only taken once the endpoint is really at zero):

    python scripts/serverless_coldstart.py --mode queue --endpoint <id> --api-key "$RUNPOD_API_KEY" \\
        --repeats 3 --idle-s 60 --label 7b_4090_flashboot_on --image "pagedserve 7B, 25 GB, weights baked in" \\
        --out results/serverless_coldstart_7b_4090_flashboot_on.json

Load-balancing endpoint (no delayTime; wall clock to the first byte through the gateway):

    python scripts/serverless_coldstart.py --mode lb --endpoint <id> --api-key "$RUNPOD_API_KEY" ...

Set the endpoint's idle timeout to 5 s for a series. Each sample: wait `--idle-s`, in queue
mode also wait for `/health` to report zero workers (a parked worker is not a cold start),
send one 16-token request, time it, then one warm request right after. Container start to
healthy is the worker log's `[worker] pagedserve up in X s` line; note it by hand in `--note`.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import httpx

PROMPT = "The capital of France is"


def lb_request(client: httpx.Client, model: str, max_tokens: int, timeout_s: float) -> dict:
    body = {"model": model, "prompt": PROMPT, "max_tokens": max_tokens, "ignore_eos": True, "stream": True}
    t0 = time.perf_counter()
    first = None
    n = 0
    with client.stream("POST", "/v1/completions", json=body, timeout=timeout_s) as r:
        for chunk in r.iter_bytes():
            if first is None and chunk:
                first = time.perf_counter()
            n += len(chunk)
        status = r.status_code
    t1 = time.perf_counter()
    return {"status": status, "ok": status == 200, "ttfb_s": (first or t1) - t0, "total_s": t1 - t0, "bytes": n}


def queue_request(client: httpx.Client, max_tokens: int, timeout_s: float) -> dict:
    """One job through /runsync; when Runpod's 90 s runsync cap returns IN_QUEUE /
    IN_PROGRESS (a long cold start), keep polling /status/<id> so the record carries the
    job's final delayTime and the true wall time."""
    body = {"input": {"prompt": PROMPT, "sampling_params": {"max_tokens": max_tokens, "ignore_eos": True}}}
    t0 = time.perf_counter()
    r = client.post("/runsync", json=body, timeout=timeout_s)
    try:
        j = r.json()
    except ValueError:
        j = {}
    job_id = j.get("id")
    deadline = t0 + timeout_s
    while r.status_code == 200 and j.get("status") in ("IN_QUEUE", "IN_PROGRESS") and job_id \
            and time.perf_counter() < deadline:
        time.sleep(2.0)
        r = client.get(f"/status/{job_id}", timeout=30.0)
        try:
            j = r.json()
        except ValueError:
            j = {}
    t1 = time.perf_counter()
    out = {"status": r.status_code, "total_s": t1 - t0}
    out["ok"] = r.status_code == 200 and j.get("status") == "COMPLETED"
    out["job_status"] = j.get("status")
    out["delay_ms"] = j.get("delayTime")          # Runpod: queue wait incl. worker start
    out["execution_ms"] = j.get("executionTime")  # Runpod: handler time
    out["worker_id"] = j.get("workerId")
    if not out["ok"]:
        out["error"] = (r.text or "")[:300]
    return out


def queue_health(client: httpx.Client) -> dict | None:
    try:
        r = client.get("/health", timeout=10.0)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def wait_for_zero_workers(client: httpx.Client, max_wait_s: float) -> dict | None:
    """Poll /health until no worker is idle or running (or give up); the last snapshot."""
    deadline = time.monotonic() + max_wait_s
    snap = None
    while time.monotonic() < deadline:
        snap = queue_health(client)
        w = (snap or {}).get("workers") or {}
        if snap is not None and (w.get("idle", 0) + w.get("running", 0) + w.get("initializing", 0)) == 0:
            return snap
        time.sleep(5.0)
    return snap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["queue", "lb"], required=True)
    ap.add_argument("--endpoint", required=True, help="endpoint id")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", default="pagedserve", help="lb mode: the model field")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--idle-s", type=float, default=60.0, help="wait before each cold sample")
    ap.add_argument("--zero-wait-s", type=float, default=300.0, help="queue: max extra wait for /health to show zero workers")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--image", default="", help="what the image carries, e.g. '0.5B, 9.8 GB, weights baked in'")
    ap.add_argument("--note", default="", help="e.g. worker-log start-to-healthy seconds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.api_key.strip():
        raise SystemExit("--api-key is empty (export RUNPOD_API_KEY in this shell)")

    base = (f"https://api.runpod.ai/v2/{args.endpoint}" if args.mode == "queue"
            else f"https://{args.endpoint}.api.runpod.ai")
    headers = {"Authorization": f"Bearer {args.api_key}"}
    runs = []
    with httpx.Client(base_url=base, headers=headers) as c:
        for i in range(args.repeats):
            print(f"[coldstart] waiting {args.idle_s:.0f} s for the endpoint to scale to zero", file=sys.stderr)
            time.sleep(args.idle_s)
            health = None
            if args.mode == "queue":
                health = wait_for_zero_workers(c, args.zero_wait_s)
                print(f"[coldstart] /health before sample: {json.dumps((health or {}).get('workers'))}", file=sys.stderr)
            if args.mode == "queue":
                cold = queue_request(c, args.max_tokens, args.timeout_s)
                warm = queue_request(c, args.max_tokens, args.timeout_s)
                print(f"[coldstart] {i + 1}/{args.repeats}: cold total {cold['total_s']:.1f} s, delayTime "
                      f"{cold.get('delay_ms')} ms, executionTime {cold.get('execution_ms')} ms, {cold.get('job_status')}; "
                      f"warm total {warm['total_s'] * 1e3:.0f} ms, delayTime {warm.get('delay_ms')} ms", file=sys.stderr)
            else:
                cold = lb_request(c, args.model, args.max_tokens, args.timeout_s)
                warm = lb_request(c, args.model, args.max_tokens, args.timeout_s)
                print(f"[coldstart] {i + 1}/{args.repeats}: cold ttfb {cold['ttfb_s']:.1f} s (total {cold['total_s']:.1f} s, "
                      f"status {cold['status']}); warm total {warm['total_s'] * 1e3:.0f} ms", file=sys.stderr)
            runs.append({"health_before": health, "cold": cold, "warm": warm})

    key = "ttfb_s" if args.mode == "lb" else "total_s"
    cold_ok = [r["cold"][key] for r in runs if r["cold"]["ok"]]
    delays = [r["cold"]["delay_ms"] for r in runs if r["cold"]["ok"] and r["cold"].get("delay_ms") is not None]
    warm_ok = [r["warm"]["total_s"] for r in runs if r["warm"]["ok"]]
    summary = {
        "cold_s": {"n": len(cold_ok), "min": min(cold_ok, default=None),
                   "median": st.median(cold_ok) if cold_ok else None, "max": max(cold_ok, default=None)},
        "cold_delay_ms": {"median": st.median(delays) if delays else None, "values": delays},
        "warm_total_s": {"median": st.median(warm_ok) if warm_ok else None},
        "failures": sum(1 for r in runs if not r["cold"]["ok"]),
    }
    out = {"kind": "serverless_coldstart", "mode": args.mode, "endpoint": args.endpoint, "label": args.label,
           "image": args.image, "note": args.note, "max_tokens": args.max_tokens, "idle_s": args.idle_s,
           "runs": runs, "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    cs = summary["cold_s"]
    fmt = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    print(f"[coldstart] {args.label or args.endpoint}: cold median {fmt(cs['median'])} s "
          f"(min {fmt(cs['min'])}, max {fmt(cs['max'])}), delayTime median {summary['cold_delay_ms']['median']} ms, "
          f"warm {summary['warm_total_s']['median'] and round(summary['warm_total_s']['median'] * 1e3)} ms, "
          f"failures {summary['failures']}; wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
