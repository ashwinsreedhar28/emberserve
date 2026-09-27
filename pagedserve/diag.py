"""Stall diagnostics and the GC knob, both off unless asked for by environment variable.

Why: the 0.5B saturation point measured 17.0-20.8k tok/s over eight repeats (vLLM's
three repeats: within 1%), with TPOT p99 doubling in the slow runs. Something stalls the
pipeline intermittently. The two processes that could stall are the engine core (one
`step()` per token round) and the API process (detokenize, SSE); in both, the first
suspect is Python's cyclic garbage collector: at 20k output tokens per second every
process allocates hundreds of thousands of small objects per second, generation-0
collections run constantly and, every few thousand of them, a generation-2 pass walks
every live object (200 streams' state, the model's module tree, the tokenizer) for tens of
milliseconds while the loop is stopped.

`PAGEDSERVE_STEP_LOG=<path>` makes the engine core record every step (start time,
duration, sequences, tokens) and every GC pause in either process (`<path>.gc-<pid>`),
flushed on exit; `scripts/stall_report.py` reads them. `PAGEDSERVE_GC=tune` applies the
standard remedy after startup: `gc.freeze()` moves everything alive at that point (weights,
modules, tokenizer, caches) out of the collector's reach, and the thresholds are raised so
generation-0 runs less often and the older generations far less.
"""

from __future__ import annotations

import atexit
import gc
import os
import time
from typing import Any

_STEP_LOG: list[tuple[float, float, int, int]] = []
_GC_LOG: list[tuple[float, float, int]] = []
_GC_START: list[float] = [0.0]
_INSTALLED = False


def step_log_path() -> str | None:
    return os.environ.get("PAGEDSERVE_STEP_LOG") or None


def gc_mode() -> str:
    return os.environ.get("PAGEDSERVE_GC", "").strip().lower()


def record_step(t_start: float, duration_s: float, num_seqs: int, num_tokens: int) -> None:
    """Called by the engine core after every step when the step log is on."""
    _STEP_LOG.append((t_start, duration_s, num_seqs, num_tokens))


def _gc_callback(phase: str, info: dict[str, Any]) -> None:
    if phase == "start":
        _GC_START[0] = time.perf_counter()
    else:
        now = time.perf_counter()
        _GC_LOG.append((_GC_START[0], now - _GC_START[0], int(info.get("generation", -1))))


def install(role: str) -> None:
    """Enable what the environment asks for in this process (`role`: "core" or "api")."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    path = step_log_path()
    if path:
        gc.callbacks.append(_gc_callback)
        atexit.register(_flush, path, role)
    if gc_mode() == "tune":
        tune_gc()


def tune_gc() -> None:
    """Freeze what is alive now and make collections rarer. Call once startup is done
    (model loaded, graphs captured), so the frozen set is the long-lived state."""
    gc.collect()
    gc.freeze()
    # (allocations before gen0 runs, gen0 runs before gen1, gen1 runs before gen2):
    # default (700, 10, 10) means a gen2 pass every 70k allocations, i.e. several per
    # second here; 50k allocations per gen0 run and a gen2 pass every 25M allocations
    # keeps the young-object sweeps cheap and the full walks rare.
    gc.set_threshold(50_000, 20, 25)


def _flush(path: str, role: str) -> None:
    try:
        if role == "core" and _STEP_LOG:
            with open(path, "w") as f:
                f.write("t_start\tduration_s\tnum_seqs\tnum_tokens\n")
                for t, d, n, k in _STEP_LOG:
                    f.write(f"{t:.6f}\t{d:.6f}\t{n}\t{k}\n")
        if _GC_LOG:
            with open(f"{path}.gc-{role}", "w") as f:
                f.write("t_start\tduration_s\tgeneration\n")
                for t, d, g in _GC_LOG:
                    f.write(f"{t:.6f}\t{d:.6f}\t{g}\n")
    except OSError:
        pass
