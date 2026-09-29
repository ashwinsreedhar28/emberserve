"""Per-step trace: where a step's time goes, on the host and on the GPU, one JSON line each.

    PAGEDSERVE_STEP_TRACE=results/trace_7b_16rps.jsonl pagedserve serve ...
    python scripts/step_trace_report.py results/trace_7b_16rps.jsonl

Off unless the variable is set (the engine then holds `None` and pays nothing). Each line
is one step:

    kind            decode_graph | decode_eager | prefill | mixed (+ "_piecewise" when the
                    piecewise runner ran it): mixed = decode rows and prompt tokens together
    n_seqs, n_decode, n_prefill_tokens, n_prefill_seqs, max_chunk
    host_sched_ms   Scheduler.schedule()
    host_build_ms   _build_inputs (plan + pinned host->device block)
    host_launch_ms  forward + sampling launched (returns before the GPU finishes, except
                    where something in the launch synchronizes; compare with gpu_ms)
    host_resolve_ms waiting for the previous step's tokens + postprocess (async), or the
                    whole sample/postprocess (sync)
    gpu_ms          CUDA-event time from the forward's first kernel to sampling's last
    gpu_gap_ms      GPU time between the previous step's end event and this step's start
                    event: > 0 means the GPU sat idle waiting for the host to enqueue
    syncs           (PAGEDSERVE_SYNC_DEBUG=1 only) "file:line" of every synchronizing CUDA
                    call made during the step, from torch.cuda.set_sync_debug_mode("warn")

GPU fields are null on CPU. The first line is `{"boot": {...}}`: the engine's startup
phases in seconds (see `LLMEngine.boot_phases`).
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch


def trace_path() -> str | None:
    return os.environ.get("PAGEDSERVE_STEP_TRACE") or None


def sync_debug() -> bool:
    return os.environ.get("PAGEDSERVE_SYNC_DEBUG", "").strip() == "1"


@dataclass
class StepRecord:
    step: int
    kind: str
    n_seqs: int
    n_decode: int
    n_prefill_tokens: int
    n_prefill_seqs: int
    max_chunk: int
    t_start: float
    host_sched_ms: float
    host_build_ms: float = 0.0
    host_launch_ms: float = 0.0
    host_resolve_ms: float = 0.0
    gpu_ms: float | None = None
    gpu_gap_ms: float | None = None
    syncs: list[str] = field(default_factory=list)
    ev_start: Any = None
    ev_end: Any = None

    def to_json(self) -> str:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("ev_")}
        for k in ("host_sched_ms", "host_build_ms", "host_launch_ms", "host_resolve_ms",
                  "gpu_ms", "gpu_gap_ms"):
            if d[k] is not None:
                d[k] = round(d[k], 4)
        if not d["syncs"]:
            del d["syncs"]
        return json.dumps(d)


def classify(query_lens: list[int], graph: bool, piecewise: bool) -> tuple[str, int, int, int, int]:
    """(kind, n_decode, n_prefill_tokens, n_prefill_seqs, max_chunk) of a scheduled step."""
    n_decode = sum(1 for q in query_lens if q == 1)
    chunks = [q for q in query_lens if q > 1]
    if not chunks:
        kind = "decode_graph" if graph else "decode_eager"
    elif n_decode:
        kind = "mixed"
    else:
        kind = "prefill"
    if piecewise and kind != "decode_graph":
        kind += "_piecewise"
    return kind, n_decode, sum(chunks), len(chunks), max(chunks, default=0)


class StepTracer:
    """Collects `StepRecord`s and writes each one when its GPU time is known."""

    def __init__(self, path: str, cuda: bool, boot: dict[str, float] | None = None) -> None:
        self.cuda = cuda
        self._f = open(path, "w", buffering=1)  # a killed process loses one line at most
        self._f.write(json.dumps({"boot": {k: round(v, 3) for k, v in (boot or {}).items()}}) + "\n")
        self._prev_end: Any = None
        self._syncs_on = cuda and sync_debug()
        if self._syncs_on:
            torch.cuda.set_sync_debug_mode("warn")
        self._catcher: warnings.catch_warnings | None = None
        self._caught: list[warnings.WarningMessage] = []

    # -- sync capture around one step() call ---------------------------------------
    def begin_step(self) -> None:
        if self._syncs_on:
            self._catcher = warnings.catch_warnings(record=True)
            self._caught = self._catcher.__enter__()
            warnings.simplefilter("always")

    def end_step(self, rec: StepRecord | None) -> None:
        if self._catcher is None:
            return
        self._catcher.__exit__(None, None, None)
        self._catcher = None
        if rec is not None:
            rec.syncs = [f"{os.path.basename(w.filename)}:{w.lineno}" for w in self._caught
                         if "synchroniz" in str(w.message)]

    # -- events ----------------------------------------------------------------------
    def event(self) -> Any:
        if not self.cuda:
            return None
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def idle(self) -> None:
        """The engine ran out of work: the GPU gap before the next step is an empty queue,
        not the host falling behind, so it is not measured."""
        self._prev_end = None

    def finish(self, rec: StepRecord) -> None:
        """Called once the step's GPU work is known complete (its tokens were read)."""
        if rec.ev_start is not None and rec.ev_end is not None:
            rec.ev_end.synchronize()  # already complete: the tokens were read after it
            rec.gpu_ms = rec.ev_start.elapsed_time(rec.ev_end)
            if self._prev_end is not None:
                rec.gpu_gap_ms = max(0.0, self._prev_end.elapsed_time(rec.ev_start))
            self._prev_end = rec.ev_end
        self._f.write(rec.to_json() + "\n")
        rec.ev_start = rec.ev_end = None

    def close(self) -> None:
        try:
            self._f.close()
        except OSError:
            pass


def now() -> float:
    return time.perf_counter()
