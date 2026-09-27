"""AsyncLLMEngine: the sync `LLMEngine` step loop on a worker thread, consumed from asyncio.

The worker thread owns the engine. It drains add/abort commands, calls `engine.step()`
while work remains and parks on an Event otherwise. Each step's outputs cross back into
the event loop with one `loop.call_soon_threadsafe` and land in per-request
`asyncio.Queue`s that `generate()` drains. `_streams` is touched only on the loop thread.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from pagedserve.engine import LLMEngine
from pagedserve.kv.block_manager import BlockManagerStats
from pagedserve.sched.request import RequestOutput, SamplingParams


class EngineNotRunningError(RuntimeError):
    """The worker thread is not running: `start()` was never called or `stop()` already was."""


@dataclass(frozen=True)
class _Add:
    request_id: str
    prompt: str | list[int]
    params: SamplingParams
    metadata: dict | None
    arrival_time: float


@dataclass(frozen=True)
class _Abort:
    request_id: str


@dataclass
class _Counters:
    requests_received: int = 0
    requests_finished: int = 0
    requests_aborted: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    steps: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    step_errors: int = 0


@dataclass(frozen=True)
class _Snapshot:
    """Scheduler/KV state captured by the worker; read from any thread."""

    kv: BlockManagerStats
    num_running: int
    num_waiting: int


class AsyncLLMEngine:
    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        engine.keep_stats = False  # engine.stats grows without bound; the server keeps counters
        self._commands: queue.SimpleQueue[_Add | _Abort] = queue.SimpleQueue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._streams: dict[str, asyncio.Queue[RequestOutput | BaseException]] = {}
        self._counters = _Counters()
        self._snapshot = self._take_snapshot()

    # ---- lifecycle (call from the event-loop thread) -------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Spawn the worker thread bound to the running event loop."""
        if self.is_running:
            raise RuntimeError("engine already started")
        self._loop = asyncio.get_running_loop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pagedserve-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Join the worker after its current step, drop in-flight requests, fail open streams."""
        if self._thread is None:
            return
        self._stop.set()
        self._wake.set()
        self._thread.join()
        self._thread = None
        self.engine.reset()
        self._snapshot = self._take_snapshot()
        self._fail_all(EngineNotRunningError("engine stopped"))

    # ---- request API -----------------------------------------------------------------------
    async def generate(self, request_id: str, prompt: str | list[int],
                       sampling_params: SamplingParams,
                       metadata: dict | None = None) -> AsyncIterator[RequestOutput]:
        """Stream the engine's outputs for one request until it finishes.

        Closing the iterator early (consumer cancelled, client gone) aborts the request in
        the engine so its KV blocks are freed. Engine-side failures are raised here.
        """
        if not self.is_running:
            raise EngineNotRunningError("engine is not running")
        if request_id in self._streams:
            raise ValueError(f"duplicate request_id {request_id!r}")
        stream: asyncio.Queue[RequestOutput | BaseException] = asyncio.Queue()
        self._streams[request_id] = stream
        self._submit(_Add(request_id, prompt, sampling_params, metadata, time.perf_counter()))
        finished = False
        try:
            while not finished:
                item = await stream.get()
                if isinstance(item, BaseException):
                    raise item
                finished = item.finished
                yield item
        finally:
            self._streams.pop(request_id, None)
            if not finished and self.is_running:
                self._submit(_Abort(request_id))

    def metrics(self) -> dict[str, int | float | bool]:
        c, snap = self._counters, self._snapshot
        return {
            "engine_running": self.is_running,
            "requests_running": snap.num_running,
            "requests_waiting": snap.num_waiting,
            "requests_received_total": c.requests_received,
            "requests_finished_total": c.requests_finished,
            "requests_aborted_total": c.requests_aborted,
            "prompt_tokens_total": c.prompt_tokens,
            "generated_tokens_total": c.generated_tokens,
            "steps_total": c.steps,
            "prefill_steps_total": c.prefill_steps,
            "decode_steps_total": c.decode_steps,
            "step_errors_total": c.step_errors,
            "kv_blocks_total": snap.kv.num_blocks,
            "kv_blocks_free": snap.kv.num_free,
            "kv_blocks_used": snap.kv.num_used,
            "kv_cache_usage": snap.kv.num_used / snap.kv.num_blocks,
            "kv_block_utilization": snap.kv.utilization,
        }

    # ---- loop-thread helpers ---------------------------------------------------------------
    def _submit(self, cmd: _Add | _Abort) -> None:
        self._commands.put(cmd)
        self._wake.set()

    def _deliver(self, outputs: list[RequestOutput]) -> None:
        for out in outputs:
            stream = self._streams.get(out.request_id)
            if stream is not None:
                stream.put_nowait(out)

    def _fail(self, request_ids: list[str], exc: BaseException) -> None:
        for rid in request_ids:
            stream = self._streams.get(rid)
            if stream is not None:
                stream.put_nowait(exc)

    def _fail_all(self, exc: BaseException) -> None:
        self._fail(list(self._streams), exc)

    # ---- worker thread ----------------------------------------------------------------------
    def _post(self, fn: Callable[..., None], *args: object) -> None:
        assert self._loop is not None
        try:
            self._loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            pass  # loop already closed: nobody is left to hear it

    def _take_snapshot(self) -> _Snapshot:
        sched = self.engine.scheduler
        return _Snapshot(self.engine.block_manager.stats(), sched.num_running, sched.num_waiting)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain_commands()
                if self.engine.has_unfinished_requests():
                    self._step()
                else:
                    self._snapshot = self._take_snapshot()
                    self._wake.wait()
                    self._wake.clear()
        except BaseException as exc:  # worker is dying: consumers must not hang
            self._post(self._fail_all, exc)
            raise

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            if isinstance(cmd, _Add):
                try:
                    req = self.engine.add_request(cmd.request_id, cmd.prompt, cmd.params,
                                                  arrival_time=cmd.arrival_time,
                                                  metadata=cmd.metadata)
                except Exception as exc:  # engine-side validation (empty / too long prompt)
                    self._post(self._fail, [cmd.request_id], exc)
                else:
                    self._counters.requests_received += 1
                    self._counters.prompt_tokens += req.num_prompt_tokens
            elif self.engine.scheduler.get_request(cmd.request_id) is not None:
                self.engine.abort_request(cmd.request_id)
                self._counters.requests_aborted += 1

    def _step(self) -> None:
        sched = self.engine.scheduler
        waiting_before = {r.request_id for r in sched.waiting}
        idle_before = not sched.running
        try:
            outputs = self.engine.step()
        except Exception as exc:
            self._counters.step_errors += 1
            in_flight = [r.request_id for r in list(sched.running) + list(sched.waiting)]
            self.engine.reset()  # every in-flight request is gone; its stream is told why
            self._post(self._fail, in_flight, exc)
            self._snapshot = self._take_snapshot()
            return
        if not outputs:
            if idle_before and waiting_before:
                # With nothing running every block was free, yet the head of the queue could
                # not be admitted (prompt larger than the KV cache): it would spin forever.
                stuck = sched.waiting[0].request_id
                self.engine.abort_request(stuck)
                self._post(self._fail, [stuck], ValueError(
                    "prompt is too long for the KV cache; it can never be scheduled"))
            self._snapshot = self._take_snapshot()
            return
        c = self._counters
        c.steps += 1
        if any(out.request_id in waiting_before for out in outputs):
            c.prefill_steps += 1
        else:
            c.decode_steps += 1
        for out in outputs:
            c.generated_tokens += len(out.new_token_ids)
            if out.finished:
                c.requests_finished += 1
                self.engine.detok.reset(out.request_id)  # the engine keeps it forever otherwise
        self._snapshot = self._take_snapshot()
        self._post(self._deliver, outputs)
