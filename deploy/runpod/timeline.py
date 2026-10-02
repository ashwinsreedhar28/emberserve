"""Wall-clock marks of a Serverless worker's cold start, returned with the first job.

Runpod's `delayTime` is one number: from the job's submission to a worker picking it up. To
say where it goes, the worker records epoch timestamps of its own phases and hands them back
when a job asks (`"timeline": true` in the job input); the client subtracts its own submit
time. Marks:

    container_start   PID 1's start time (/proc/1/stat), i.e. the container's init process
    worker_start      this Python process's start time (/proc/self/stat)
    worker_main       main() entered (after imports)
    serve_spawned     `emberserve serve` launched
    engine_boot       the engine core printed its `[boot]` line (weights, KV cache, graphs done)
    serve_healthy     /health answered 200
    sdk_ready         the Runpod SDK imported and its job loop about to start
    first_job         the handler received its first job

Client and worker clocks are both NTP-synced but not the same clock; differences across
that boundary (submit -> container_start) carry the skew, typically well under 100 ms.
Stdlib only.
"""

from __future__ import annotations

import os
import time

_marks: dict[str, float] = {}
_notes: dict[str, str] = {}


def _boot_time() -> float | None:
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except OSError:
        return None
    return None


def proc_start_wall(pid: int | str) -> float | None:
    """Epoch seconds at which process `pid` started (Linux /proc), or None."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return None
    # the command name (field 2) may contain spaces; fields after the last ')' are fixed
    fields = stat[stat.rfind(")") + 2:].split()
    try:
        start_ticks = int(fields[19])  # field 22 overall: starttime, in clock ticks since boot
    except (IndexError, ValueError):
        return None
    btime = _boot_time()
    if btime is None:
        return None
    return btime + start_ticks / os.sysconf("SC_CLK_TCK")


def mark(name: str, when: float | None = None) -> None:
    """Record `name` once (the first call wins)."""
    if name not in _marks:
        _marks[name] = time.time() if when is None else when


def note(name: str, text: str) -> None:
    _notes.setdefault(name, text)


def record_process_starts() -> None:
    for name, pid in (("container_start", 1), ("worker_start", "self")):
        t = proc_start_wall(pid)
        if t is not None:
            mark(name, t)


def snapshot() -> dict:
    return {"marks": dict(sorted(_marks.items(), key=lambda kv: kv[1])), "notes": dict(_notes)}


def reset() -> None:  # tests
    _marks.clear()
    _notes.clear()
