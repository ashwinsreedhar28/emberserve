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
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any

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


def _run_engine_core(spec: EngineSpec, cmd_conn: Connection, out_conn: Connection) -> None:
    """Subprocess main. Protocol out: ("ready", info) once; then ("step", rows, snapshot)
    per step, ("idle", snapshot) when the queue drains, ("failed", [request_ids], message)
    for requests the engine could not serve, ("fatal", message) if the loop dies."""
    parent = os.getppid()
    engine = None
    try:
        engine = build_engine(spec)
        engine.keep_stats = False
        out_conn.send(("ready", {"eos_token_ids": sorted(engine.eos_token_ids),
                                 "max_model_len": engine.config.max_model_len,
                                 "vocab_size": engine.model_config.vocab_size,
                                 "pid": os.getpid()}))
        sched = engine.scheduler

        def snapshot() -> dict[str, int | float]:
            st = engine.block_manager.stats()
            return {"num_running": sched.num_running, "num_waiting": sched.num_waiting,
                    "kv_blocks_total": st.num_blocks, "kv_blocks_free": st.num_free,
                    "kv_blocks_used": st.num_used, "kv_block_utilization": st.utilization,
                    "spec_drafted": engine.spec_drafted, "spec_accepted": engine.spec_accepted}

        def handle(cmd: tuple) -> bool:
            """Apply one command; True means stop."""
            kind = cmd[0]
            if kind == "stop":
                return True
            if kind == "add":
                req: CoreRequest = cmd[1]
                try:
                    engine.add_request(req.request_id, req.prompt_token_ids, req.params,
                                       arrival_time=req.arrival_time)
                except Exception as exc:  # noqa: BLE001 - reported per request
                    out_conn.send(("failed", [req.request_id], f"{type(exc).__name__}: {exc}"))
            elif kind == "abort":
                engine.abort_request(cmd[1])
            return False

        idle_reported = False
        while True:
            if not engine.has_unfinished_requests():
                if not idle_reported:  # metrics stay current while nothing is stepping
                    out_conn.send(("idle", snapshot()))
                    idle_reported = True
                # Idle: block for a command (zero CPU), checking that the parent is alive.
                while not cmd_conn.poll(0.5):
                    if os.getppid() != parent:
                        return
            while cmd_conn.poll():  # drain everything that arrived since the last step
                if handle(cmd_conn.recv()):
                    return
            if not engine.has_unfinished_requests():
                continue
            idle_reported = False
            waiting_before = {r.request_id for r in sched.waiting}
            idle_before = not sched.running
            try:
                outputs = engine.step()
            except Exception as exc:  # noqa: BLE001 - every in-flight request is told
                in_flight = [r.request_id for r in list(sched.running) + list(sched.waiting)]
                engine.reset()
                out_conn.send(("failed", in_flight,
                               f"engine step failed: {type(exc).__name__}: {exc}"))
                continue
            if not outputs:
                if idle_before and waiting_before and not engine.last_step_scheduled:
                    stuck = sched.waiting[0].request_id
                    engine.abort_request(stuck)
                    out_conn.send(("failed", [stuck],
                                   "prompt is too long for the KV cache; it can never be scheduled"))
                continue
            rows: list[StepRow] = [
                (o.request_id, list(o.new_token_ids), o.finished,
                 o.finish_reason.value if o.finish_reason else None)
                for o in outputs]
            out_conn.send(("step", rows, snapshot()))
    except Exception:  # noqa: BLE001
        try:
            out_conn.send(("fatal", traceback.format_exc()))
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        if engine is not None:
            engine.shutdown()  # tensor-parallel workers, if any


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
        cmd_recv, cmd_send = self._ctx.Pipe(duplex=False)
        out_recv, out_send = self._ctx.Pipe(duplex=False)
        self._proc = self._ctx.Process(target=_run_engine_core, args=(self.spec, cmd_recv, out_send),
                                       name="pagedserve-engine-core", daemon=True)
        self._proc.start()
        cmd_recv.close()
        out_send.close()
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
