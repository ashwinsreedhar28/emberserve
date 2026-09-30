"""The engine core in its own process (the vLLM v1 layout).

`_run_engine_core` owns the `LLMEngine` (model, scheduler, KV cache, sampler) in a spawned
subprocess and talks to the API process over two pipes: commands in (`add`, `abort`,
`stop`), one message per step out (token ids only, no text). The API process keeps the
tokenizer: it encodes prompts, runs the incremental detokenizer, checks stop strings and
writes SSE. The two interpreters no longer share a GIL, so the server's per-token work
overlaps the engine's per-step work instead of serializing with it.

Measured motivation (A100, Qwen2.5-0.5B, 200 streams): the in-process step is 6.1 ms but
the same step seen through the server is 10.2 ms; the difference is the server thread
holding the GIL against the engine thread.

Stop strings are the one thing the core cannot see. The client detects them on decoded
text, marks the request finished for its consumer, and sends an abort; the core may have
produced a token or two more in the meantime, which the client drops.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any

from pagedserve import diag
from pagedserve.config import EngineConfig
from pagedserve.sched.request import SamplingParams


@dataclass(frozen=True)
class EngineSpec:
    """How the core process builds its engine. `model_dir` for a real snapshot, or
    `tiny=True` for the 2-layer random model tests use (seeded)."""

    engine_config: EngineConfig
    model_dir: str | None = None
    tiny: bool = False
    tiny_seed: int = 0
    tiny_overrides: dict = field(default_factory=dict)
    # > 0: no model at all, a clock that hands every running request one token per step
    # (server/fake_engine.py) - the API layer's own throughput ceiling.
    fake_step_ms: float = 0.0


@dataclass(frozen=True)
class CoreRequest:
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams
    arrival_time: float


# Per-step output row: (request_id, token_id, finished, finish_reason value or None).
# Plain tuples keep the pickle small; the client tracks token counts itself.
StepRow = tuple[str, list[int], bool, str | None]  # request id, new token ids, finished, reason


def build_engine(spec: EngineSpec):
    from pagedserve.engine import LLMEngine

    if spec.fake_step_ms > 0:
        from pagedserve.config import ModelConfig
        from pagedserve.server.fake_engine import FakeEngine

        return FakeEngine(ModelConfig.tiny(**spec.tiny_overrides), spec.engine_config,
                          step_ms=spec.fake_step_ms, seed=spec.tiny_seed)
    if spec.engine_config.tensor_parallel_size > 1:
        from pagedserve.dist import WorkerSpec

        if spec.tiny and spec.engine_config.num_gpu_blocks is None:
            spec.engine_config.num_gpu_blocks = 512
        return LLMEngine.launch_tp(WorkerSpec(spec.engine_config, spec.model_dir, spec.tiny,
                                              spec.tiny_seed, dict(spec.tiny_overrides)),
                                   load_tokenizer=False)
    if spec.tiny:
        from pagedserve.config import ModelConfig
        from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic

        mcfg = ModelConfig.tiny(**spec.tiny_overrides)
        model = Qwen2ForCausalLM(mcfg)
        reset_parameters_deterministic(model, spec.tiny_seed)
        ecfg = spec.engine_config
        if ecfg.num_gpu_blocks is None:
            ecfg.num_gpu_blocks = 512
        return LLMEngine(model, mcfg, ecfg, tokenizer=None)
    if spec.model_dir is None:
        raise ValueError("EngineSpec needs model_dir or tiny=True")
    return LLMEngine.from_pretrained(spec.model_dir, spec.engine_config, load_tokenizer=False)


def _run_engine_core(spec: EngineSpec, cmd_conns: "Connection | list[Connection]",
                     out_conns: "Connection | list[Connection]") -> None:
    """Subprocess main. One (commands in, outputs out) pipe pair per API worker; with a
    single pair this is the one-API-process server. Protocol out, per worker: ("ready",
    info) once; then ("step", rows, snapshot) per step carrying only that worker's rows,
    ("idle", snapshot) when the queue drains, ("failed", [request_ids], message) for its
    requests the engine could not serve, ("fatal", message) if the loop dies.

    With several API workers (`serve --api-workers N`, `serve_multi`) every request is
    owned by the worker that added it: its rows, failures and stuck-queue aborts go back on
    that worker's pipe only. A worker whose pipe closes has its requests aborted; the core
    exits when every worker is gone (or on "stop", which only the single-worker client
    sends)."""
    from multiprocessing.connection import wait as mp_wait

    if not isinstance(cmd_conns, list):
        cmd_conns, out_conns = [cmd_conns], [out_conns]
    n_workers = len(cmd_conns)
    parent = os.getppid()
    engine = None
    live = list(range(n_workers))
    owner: dict[str, int] = {}  # request id -> the worker that added it

    def send(w: int, msg: tuple) -> None:
        if w not in live:
            return
        try:
            out_conns[w].send(msg)
        except (BrokenPipeError, EOFError, OSError):
            drop_worker(w)

    def drop_worker(w: int) -> None:
        if w in live:
            live.remove(w)
        if engine is None:
            return
        for rid in [r for r, o in owner.items() if o == w]:
            owner.pop(rid, None)
            engine.abort_request(rid)

    def by_owner(request_ids: list[str]) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for rid in request_ids:
            w = owner.get(rid)
            if w is not None:
                out.setdefault(w, []).append(rid)
        return out

    try:
        engine = build_engine(spec)
        engine.keep_stats = False
        diag.install("core")  # PAGEDSERVE_STEP_LOG / PAGEDSERVE_GC, after the model is built
        step_log = diag.step_log_path() is not None
        boot = getattr(engine, "boot_phases", {}) or {}
        if boot:  # where startup went (the cold-start budget), one line in the server log
            notes = getattr(engine, "boot_notes", "")
            print("[boot] " + " · ".join(f"{k.removesuffix('_s')} {v:.2f} s" for k, v in boot.items())
                  + (f" ({notes})" if notes else ""), file=sys.stderr, flush=True)
        info = {"eos_token_ids": sorted(engine.eos_token_ids), "boot_phases": boot,
                "max_model_len": engine.config.max_model_len,
                "vocab_size": engine.model_config.vocab_size, "pid": os.getpid(),
                "api_workers": n_workers}
        for w in range(n_workers):
            send(w, ("ready", info))
        sched = engine.scheduler

        def snapshot() -> dict[str, int | float]:
            st = engine.block_manager.stats()
            return {"num_running": sched.num_running, "num_waiting": sched.num_waiting,
                    "kv_blocks_total": st.num_blocks, "kv_blocks_free": st.num_free,
                    "kv_blocks_used": st.num_used, "kv_block_utilization": st.utilization,
                    "spec_drafted": engine.spec_drafted, "spec_accepted": engine.spec_accepted}

        def handle(w: int, cmd: tuple) -> bool:
            """Apply one command from worker `w`; True means stop."""
            kind = cmd[0]
            if kind == "stop":
                return True
            if kind == "add":
                req: CoreRequest = cmd[1]
                if req.request_id in owner:
                    send(w, ("failed", [req.request_id], f"duplicate request id {req.request_id!r}"))
                    return False
                try:
                    engine.add_request(req.request_id, req.prompt_token_ids, req.params,
                                       arrival_time=req.arrival_time)
                    owner[req.request_id] = w
                except Exception as exc:  # noqa: BLE001 - reported per request
                    send(w, ("failed", [req.request_id], f"{type(exc).__name__}: {exc}"))
            elif kind == "abort":
                if owner.get(cmd[1]) == w:
                    owner.pop(cmd[1], None)
                    engine.abort_request(cmd[1])
            return False

        def drain(timeout: float) -> bool:
            """Apply every command waiting on any worker's pipe; True means stop."""
            conns = {cmd_conns[w]: w for w in live}
            for conn in mp_wait(list(conns), timeout):
                w = conns[conn]
                try:
                    while conn.poll():
                        if handle(w, conn.recv()):
                            return True
                except (EOFError, OSError):
                    drop_worker(w)
            return False

        idle_reported = False
        while live:
            if not engine.has_unfinished_requests():
                if not idle_reported:  # metrics stay current while nothing is stepping
                    snap = snapshot()
                    for w in list(live):
                        send(w, ("idle", snap))
                    idle_reported = True
                # Idle: block for a command (zero CPU), checking that the parent is alive.
                while live and not mp_wait([cmd_conns[w] for w in live], 0.5):
                    if os.getppid() != parent:
                        return
            if drain(0):
                return
            if not engine.has_unfinished_requests():
                continue
            idle_reported = False
            waiting_before = {r.request_id for r in sched.waiting}
            idle_before = not sched.running
            try:
                if step_log:
                    t_step = time.perf_counter()
                    n_seqs = sched.num_running
                outputs = engine.step()
                if step_log:
                    diag.record_step(t_step, time.perf_counter() - t_step, n_seqs,
                                     sum(len(o.new_token_ids) for o in outputs))
            except Exception as exc:  # noqa: BLE001 - every in-flight request is told
                if engine.tp.size > 1:
                    # A failed step may have left a worker inside a collective the driver
                    # never completed; there is no resynchronizing that, so the core dies
                    # (the API process reports "fatal") rather than serving garbage.
                    raise
                in_flight = [r.request_id for r in list(sched.running) + list(sched.waiting)]
                groups = by_owner(in_flight)
                engine.reset()
                owner.clear()
                for w, rids in groups.items():
                    send(w, ("failed", rids, f"engine step failed: {type(exc).__name__}: {exc}"))
                continue
            if not outputs:
                if idle_before and waiting_before and not engine.last_step_scheduled:
                    stuck = sched.waiting[0].request_id
                    engine.abort_request(stuck)
                    w = owner.pop(stuck, None)
                    if w is not None:
                        send(w, ("failed", [stuck],
                                 "prompt is too long for the KV cache; it can never be scheduled"))
                continue
            snap = snapshot()
            if n_workers == 1:
                rows: list[StepRow] = [
                    (o.request_id, list(o.new_token_ids), o.finished,
                     o.finish_reason.value if o.finish_reason else None)
                    for o in outputs]
                for o in outputs:
                    if o.finished:
                        owner.pop(o.request_id, None)
                send(0, ("step", rows, snap))
                continue
            per: dict[int, list[StepRow]] = {}
            for o in outputs:
                w = owner.get(o.request_id)
                if w is None:
                    continue  # its worker is gone
                per.setdefault(w, []).append(
                    (o.request_id, list(o.new_token_ids), o.finished,
                     o.finish_reason.value if o.finish_reason else None))
                if o.finished:
                    owner.pop(o.request_id, None)
            for w, rows in per.items():
                send(w, ("step", rows, snap))
    except Exception:  # noqa: BLE001
        msg = ("fatal", traceback.format_exc())
        for w in list(live):
            try:
                out_conns[w].send(msg)
            except Exception:  # noqa: BLE001
                pass
        raise
    finally:
        if engine is not None:
            engine.shutdown()  # tensor-parallel workers, if any


def spawn_core(spec: EngineSpec, n_workers: int = 1) -> tuple[Any, list[tuple[Connection, Connection]]]:
    """Start the engine core with one (commands, outputs) pipe pair per API worker. Returns
    the process and, per worker, (command sender, output receiver) for its client."""
    ctx = mp.get_context("spawn")
    cmd_recvs, out_sends, chans = [], [], []
    for _ in range(n_workers):
        cmd_recv, cmd_send = ctx.Pipe(duplex=False)
        out_recv, out_send = ctx.Pipe(duplex=False)
        cmd_recvs.append(cmd_recv)
        out_sends.append(out_send)
        chans.append((cmd_send, out_recv))
    proc = ctx.Process(target=_run_engine_core, args=(spec, cmd_recvs, out_sends),
                       name="pagedserve-engine-core", daemon=True)
    proc.start()
    for c in cmd_recvs + out_sends:
        c.close()
    return proc, chans


class EngineCoreProcess:
    """Owns the subprocess and its pipes. Thread-safe `send`; `recv` from one reader."""

    def __init__(self, spec: EngineSpec) -> None:
        self.spec = spec
        self._ctx = mp.get_context("spawn")
        self._proc: Any = None
        self._cmd_send: Connection | None = None
        self._out_recv: Connection | None = None
        self.info: dict[str, Any] = {}
        import threading

        self._send_lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(self, ready_timeout_s: float = 900.0) -> dict[str, Any]:
        self._proc, [(cmd_send, out_recv)] = spawn_core(self.spec, 1)
        self._cmd_send, self._out_recv = cmd_send, out_recv
        deadline = time.monotonic() + ready_timeout_s
        while True:
            if out_recv.poll(1.0):
                msg = out_recv.recv()
                if msg[0] == "ready":
                    self.info = msg[1]
                    return self.info
                if msg[0] == "fatal":
                    raise RuntimeError(f"engine core failed to start:\n{msg[1]}")
            if not self._proc.is_alive():
                raise RuntimeError("engine core process exited during startup")
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError("engine core did not become ready")

    def send(self, msg: tuple) -> None:
        assert self._cmd_send is not None
        with self._send_lock:
            self._cmd_send.send(msg)

    def recv(self, timeout: float | None = None) -> tuple | None:
        """Next message from the core, or None on timeout."""
        assert self._out_recv is not None
        if timeout is not None and not self._out_recv.poll(timeout):
            return None
        return self._out_recv.recv()

    def stop(self, timeout_s: float = 10.0) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.is_alive():
                self.send(("stop",))
        except Exception:  # noqa: BLE001
            pass
        self._proc.join(timeout_s)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(5.0)
        for c in (self._cmd_send, self._out_recv):
            if c is not None:
                c.close()
        self._proc = None
        self._cmd_send = self._out_recv = None


class AttachedCore:
    """An API worker's handle on a core it did not start (`serve_multi`): the same
    interface as `EngineCoreProcess` over one pipe pair of a shared core. `stop()` only
    closes this worker's pipes; the core's lifetime belongs to the supervisor, and the core
    aborts whatever this worker still had in flight when its pipe closes."""

    def __init__(self, cmd_send: Connection, out_recv: Connection, core_pid: int) -> None:
        import threading

        self._cmd_send: Connection | None = cmd_send
        self._out_recv: Connection | None = out_recv
        self.core_pid = core_pid
        self.info: dict[str, Any] = {}
        self._dead = False
        self._send_lock = threading.Lock()

    @property
    def alive(self) -> bool:
        if self._dead or self._out_recv is None:
            return False
        try:
            os.kill(self.core_pid, 0)
        except OSError:
            return False
        return True

    def start(self, ready_timeout_s: float = 900.0) -> dict[str, Any]:
        deadline = time.monotonic() + ready_timeout_s
        while True:
            msg = self.recv(timeout=1.0)
            if msg is not None:
                if msg[0] == "ready":
                    self.info = msg[1]
                    return self.info
                if msg[0] == "fatal":
                    raise RuntimeError(f"engine core failed to start:\n{msg[1]}")
            elif not self.alive:
                raise RuntimeError("engine core process exited during startup")
            if time.monotonic() > deadline:
                raise TimeoutError("engine core did not become ready")

    def send(self, msg: tuple) -> None:
        if msg[0] == "stop":
            return  # a worker does not stop the shared core
        assert self._cmd_send is not None
        with self._send_lock:
            self._cmd_send.send(msg)

    def recv(self, timeout: float | None = None) -> tuple | None:
        if self._out_recv is None:
            return None
        try:
            if timeout is not None and not self._out_recv.poll(timeout):
                return None
            return self._out_recv.recv()
        except (EOFError, OSError):
            self._dead = True
            return None

    def stop(self, timeout_s: float = 10.0) -> None:  # noqa: ARG002 - same signature
        for c in (self._cmd_send, self._out_recv):
            if c is not None:
                try:
                    c.close()
                except OSError:
                    pass
        self._cmd_send = self._out_recv = None
        self._dead = True
