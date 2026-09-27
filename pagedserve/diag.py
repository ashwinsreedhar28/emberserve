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
flushed on exit; `scripts/stall_report.py` reads them. The GC remedy is on by default
(`PAGEDSERVE_GC=off` disables it) and is the standard one, applied after startup: `gc.freeze()` moves everything alive at that point (weights,
modules, tokenizer, caches) out of the collector's reach, and the thresholds are raised so
generation-0 runs less often and the older generations far less.
"""

from __future__ import annotations

import atexit
import gc
import os
import signal
import threading
import time
from typing import Any

_STEP_LOG: list[tuple[float, float, int, int]] = []
_GC_LOG: list[tuple[float, float, int]] = []
_GC_START: list[float] = [0.0]
_INSTALLED = False


def step_log_path() -> str | None:
    return os.environ.get("PAGEDSERVE_STEP_LOG") or None


def gc_mode() -> str:
    """`tune` (the default since v9: it halved TPOT p99 at saturation and costs nothing),
    or `off` (`PAGEDSERVE_GC=off`) to leave the collector at Python's defaults."""
    return os.environ.get("PAGEDSERVE_GC", "tune").strip().lower()


def record_step(t_start: float, duration_s: float, num_seqs: int, num_tokens: int) -> None:
    """Called by the engine core after every step when the step log is on."""
    _STEP_LOG.append((t_start, duration_s, num_seqs, num_tokens))
    if len(_STEP_LOG) % 5000 == 0:  # a hard kill still leaves most of the data
        path = step_log_path()
        if path:
            _flush(path, "core")


def _gc_callback(phase: str, info: dict[str, Any]) -> None:
    if phase == "start":
        _GC_START[0] = time.perf_counter()
    else:
        now = time.perf_counter()
        _GC_LOG.append((_GC_START[0], now - _GC_START[0], int(info.get("generation", -1))))


# ---- in-process sampling profiler (py-spy cannot ptrace inside the Runpod container) ------
_SAMPLES: dict[str, int] = {}
_SAMPLER_STOP = threading.Event()


def _sampler(interval_s: float, flush) -> None:
    import sys

    my_ident = threading.get_ident()  # (the sampler's own thread, not the caller's)
    n = 0
    while not _SAMPLER_STOP.wait(interval_s):
        n += 1
        if n % int(5.0 / interval_s) == 0:
            flush()  # every ~5 s: a process that is SIGKILLed still leaves its profile
        for ident, frame in sys._current_frames().items():
            if ident == my_ident:
                continue
            parts = []
            f = frame
            while f is not None:
                code = f.f_code
                parts.append(f"{code.co_filename.rsplit('/', 1)[-1]}:{code.co_name}")
                f = f.f_back
            key = ";".join(reversed(parts))
            _SAMPLES[key] = _SAMPLES.get(key, 0) + 1


def start_sampler(path: str, role: str, interval_s: float = 0.004) -> None:
    """Sample every thread's Python stack `1/interval_s` times a second and write the
    collapsed stacks (`frame;frame;... count`, the py-spy raw format) to
    `<path>.<role>` at exit; `scripts/pyspy_summary.py` reads it. The sampler needs the
    GIL to run, so C code that holds it is under-counted; Python-side cost is what it shows."""
    def write() -> None:
        try:
            with open(f"{path}.{role}", "w") as f:
                for k, n in sorted(list(_SAMPLES.items()), key=lambda kv: -kv[1]):
                    f.write(f"{k} {n}\n")
        except OSError:
            pass

    def flush() -> None:
        _SAMPLER_STOP.set()
        write()

    threading.Thread(target=_sampler, args=(interval_s, write), daemon=True,
                     name="pagedserve-sampler").start()
    atexit.register(flush)
    _flushers.append(flush)


_flushers: list = []


def install(role: str) -> None:
    """Enable what the environment asks for in this process (`role`: "core" or "api")."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    prof = os.environ.get("PAGEDSERVE_SAMPLE_PROFILE")
    if prof:
        start_sampler(prof, role)
    path = step_log_path()
    if prof and role == "core" and not path:
        def _on_term_prof(signum, frame):  # noqa: ARG001
            for fl in _flushers:
                fl()
            os._exit(0)

        signal.signal(signal.SIGTERM, _on_term_prof)
    if path:
        gc.callbacks.append(_gc_callback)
        atexit.register(_flush, path, role)
        if role == "core":
            # The benchmark runner stops the server with a process-group SIGTERM, which
            # would end the core before its exit hooks: flush, then leave.
            def _on_term(signum, frame):  # noqa: ARG001
                _flush(path, role)
                for fl in _flushers:
                    fl()
                os._exit(0)

            signal.signal(signal.SIGTERM, _on_term)
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
