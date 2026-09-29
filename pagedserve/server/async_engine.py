"""AsyncLLMEngine: the sync `LLMEngine` step loop on a worker thread, consumed from asyncio.

The worker thread owns the engine. It drains add/abort commands, calls `engine.step()`
while work remains and parks on an Event otherwise. Each step's outputs cross back into
the event loop with one `loop.call_soon_threadsafe` and land in per-request
`asyncio.Queue`s that `generate()` drains. `_streams` is touched only on the loop thread.
"""

from __future__ import annotations

import asyncio
import dataclasses
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pagedserve.engine import LLMEngine
from pagedserve.kv.block_manager import BlockManagerStats
from pagedserve.sched.request import FinishReason, RequestOutput, SamplingParams
from pagedserve.tokenizer import IncrementalDetokenizer, Tokenizer

if TYPE_CHECKING:
    from pagedserve.server.engine_core import EngineSpec



async def _drain_batches(stream: "asyncio.Queue[RequestOutput | BaseException]"):
    """Yield lists of outputs: one awaited item plus everything already queued behind it.
    Under load the API process runs behind the engine and a request's queue holds several
    tokens; delivering them in one SSE write is what keeps the per-token cost bounded."""
    finished = False
    while not finished:
        item = await stream.get()
        if isinstance(item, BaseException):
            raise item
        batch = [item]
        finished = item.finished
        while not finished:
            try:
                nxt = stream.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(nxt, BaseException):
                raise nxt
            batch.append(nxt)
            finished = nxt.finished
        yield batch


async def _items(batches: "AsyncIterator[list[RequestOutput]]") -> "AsyncIterator[RequestOutput]":
    async for batch in batches:
        for item in batch:
            yield item


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
    # Server-side latency sums over finished requests (seconds), measured from the
    # request's arrival at the API process: the client-side numbers a load generator
    # reports include its own queueing, which at a 200-request burst is most of the TTFT.
    ttft_s_sum: float = 0.0
    ttft_count: int = 0
    tpot_s_sum: float = 0.0
    tpot_count: int = 0
    e2e_s_sum: float = 0.0
    e2e_count: int = 0

    def observe(self, ttft_s: float | None, tpot_s: float | None, e2e_s: float | None) -> None:
        if ttft_s is not None:
            self.ttft_s_sum += ttft_s
            self.ttft_count += 1
        if tpot_s is not None:
            self.tpot_s_sum += tpot_s
            self.tpot_count += 1
        if e2e_s is not None:
            self.e2e_s_sum += e2e_s
            self.e2e_count += 1

    def latency_metrics(self) -> dict[str, float | int]:
        return {"ttft_s_sum": self.ttft_s_sum, "ttft_count": self.ttft_count,
                "tpot_s_sum": self.tpot_s_sum, "tpot_count": self.tpot_count,
                "e2e_s_sum": self.e2e_s_sum, "e2e_count": self.e2e_count}


class SharedCounters:
    """Every API worker's `_Counters` side by side in shared memory, so `/metrics` on any
    worker reports the server's totals (`serve --api-workers N`): each worker writes only
    its own row after it changes a counter, and a read sums the rows. Created by the
    supervisor and handed to the workers at spawn."""

    FIELDS = tuple(f.name for f in dataclasses.fields(_Counters))

    def __init__(self, n_workers: int, ctx=None) -> None:
        import multiprocessing as mp

        self.n_workers = n_workers
        self.arr = (ctx or mp.get_context("spawn")).RawArray("d", n_workers * len(self.FIELDS))

    def publish(self, worker: int, c: _Counters) -> None:
        base = worker * len(self.FIELDS)
        for i, name in enumerate(self.FIELDS):
            self.arr[base + i] = getattr(c, name)

    def total(self) -> _Counters:
        k = len(self.FIELDS)
        sums = [sum(self.arr[w * k + i] for w in range(self.n_workers)) for i in range(k)]
        vals = {}
        for name, v in zip(self.FIELDS, sums):
            vals[name] = v if name.endswith("_sum") else int(round(v))
        return _Counters(**vals)


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
    def vocab_size(self) -> int:
        return self.engine.model_config.vocab_size

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
        async for item in _items(self.generate_batches(request_id, prompt, sampling_params, metadata)):
            yield item

    async def generate_batches(self, request_id: str, prompt: str | list[int],
                               sampling_params: SamplingParams,
                               metadata: dict | None = None) -> AsyncIterator[list[RequestOutput]]:
        """`generate`, one list per wake-up: every output queued for the request by then."""
        if not self.is_running:
            raise EngineNotRunningError("engine is not running")
        if request_id in self._streams:
            raise ValueError(f"duplicate request_id {request_id!r}")
        stream: asyncio.Queue[RequestOutput | BaseException] = asyncio.Queue()
        self._streams[request_id] = stream
        self._submit(_Add(request_id, prompt, sampling_params, metadata, time.perf_counter()))
        finished = False
        try:
            async for batch in _drain_batches(stream):
                finished = batch[-1].finished
                yield batch
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
            "spec_drafted_total": self.engine.spec_drafted,
            "spec_accepted_total": self.engine.spec_accepted,
            **c.latency_metrics(),
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
            if idle_before and waiting_before and not self.engine.last_step_scheduled:
                # With nothing running every block was free, yet the head of the queue could
                # not be admitted (prompt larger than the KV cache): it would spin forever.
                # (Under async scheduling a launched step returns its outputs next call, so
                # an empty result right after a launch is not that.)
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
                m = out.metrics or {}
                c.observe(m.get("ttft_s"), m.get("tpot_s"), m.get("e2e_s"))
                self.engine.detok.reset(out.request_id)  # the engine keeps it forever otherwise
        self._snapshot = self._take_snapshot()
        self._post(self._deliver, outputs)


# ---- engine core in its own process ------------------------------------------------------

@dataclass
class _ClientRequest:
    request_id: str
    prompt_len: int
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    arrival: float = 0.0
    first_token: float | None = None
    finished: bool = False  # set by the client (stop string) or the core


class AsyncEngineCoreClient:
    """Same interface as `AsyncLLMEngine`, with the engine in a subprocess (see
    `server/engine_core.py`). This side owns the tokenizer: it encodes prompts, decodes
    tokens incrementally (batched per step, off the event loop), checks stop strings, and
    delivers `RequestOutput`s into per-request asyncio queues exactly like the in-process
    engine does."""

    def __init__(self, spec: "EngineSpec", tokenizer: Tokenizer | None, core=None,
                 shared: "tuple[SharedCounters, int] | None" = None) -> None:
        """`core`: a handle on a core started elsewhere (`engine_core.AttachedCore`, one API
        worker of several); by default this client starts its own. `shared`: this worker's
        row in the server-wide counters (`SharedCounters`, index)."""
        from pagedserve.server.engine_core import EngineCoreProcess

        self.spec = spec
        self.tokenizer = tokenizer
        self.core = core if core is not None else EngineCoreProcess(spec)
        self._shared = shared
        self.detok = IncrementalDetokenizer(tokenizer)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: threading.Thread | None = None
        self._streams: dict[str, asyncio.Queue[RequestOutput | BaseException]] = {}
        self._reqs: dict[str, _ClientRequest] = {}  # touched only on the reader thread
        self._reqs_lock = threading.Lock()
        self._counters = _Counters()
        self._snapshot: dict[str, int | float] = {"num_running": 0, "num_waiting": 0,
                                                  "kv_blocks_total": 0, "kv_blocks_free": 0,
                                                  "kv_blocks_used": 0, "kv_block_utilization": 0.0}
        self._stopping = threading.Event()

    # ---- lifecycle -------------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self.core.alive and self._reader is not None and self._reader.is_alive()

    @property
    def engine(self) -> "AsyncEngineCoreClient":  # the app reads `.engine.tokenizer`
        return self

    @property
    def vocab_size(self) -> int:
        return int(self.core.info.get("vocab_size", 0)) or (1 << 31)

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("engine already started")
        self._loop = asyncio.get_running_loop()
        self._stopping.clear()
        self.core.start()
        self._reader = threading.Thread(target=self._read_loop, name="pagedserve-core-reader",
                                        daemon=True)
        self._reader.start()

    def stop(self) -> None:
        self._stopping.set()
        self.core.stop()
        if self._reader is not None:
            self._reader.join(5.0)
            self._reader = None
        with self._reqs_lock:
            self._reqs.clear()
        self._post(self._fail_all, EngineNotRunningError("engine stopped"))

    # ---- request API -------------------------------------------------------------------------
    async def generate(self, request_id: str, prompt: str | list[int],
                       sampling_params: SamplingParams,
                       metadata: dict | None = None) -> AsyncIterator[RequestOutput]:
        async for item in _items(self.generate_batches(request_id, prompt, sampling_params, metadata)):
            yield item

    async def generate_batches(self, request_id: str, prompt: str | list[int],
                               sampling_params: SamplingParams,
                               metadata: dict | None = None) -> AsyncIterator[list[RequestOutput]]:
        from pagedserve.server.engine_core import CoreRequest

        if not self.is_running:
            raise EngineNotRunningError("engine is not running")
        if request_id in self._streams:
            raise ValueError(f"duplicate request_id {request_id!r}")
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("engine has no tokenizer; pass prompt token ids")
            prompt_ids = self.tokenizer.encode(prompt)
        else:
            prompt_ids = list(prompt)
        stream: asyncio.Queue[RequestOutput | BaseException] = asyncio.Queue()
        self._streams[request_id] = stream
        now = time.perf_counter()
        with self._reqs_lock:
            self._reqs[request_id] = _ClientRequest(request_id, len(prompt_ids), sampling_params,
                                                    arrival=now)
        self._counters.requests_received += 1
        self._counters.prompt_tokens += len(prompt_ids)
        self._publish()
        self.core.send(("add", CoreRequest(request_id, prompt_ids, sampling_params, now)))
        finished = False
        try:
            async for batch in _drain_batches(stream):
                finished = batch[-1].finished
                yield batch
        finally:
            self._streams.pop(request_id, None)
            if not finished and self.is_running:
                self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        with self._reqs_lock:
            self._reqs.pop(request_id, None)
        self.detok.reset(request_id)
        self._counters.requests_aborted += 1
        self._publish()
        try:
            self.core.send(("abort", request_id))
        except Exception:  # noqa: BLE001 - core gone; nothing to abort
            pass

    def _publish(self) -> None:
        if self._shared is not None:
            shared, idx = self._shared
            shared.publish(idx, self._counters)

    def metrics(self) -> dict[str, int | float | bool]:
        c, snap = self._counters, self._snapshot
        if self._shared is not None:
            self._publish()
            c = self._shared[0].total()
        total = max(int(snap["kv_blocks_total"]), 1)
        return {
            "engine_running": self.is_running,
            "api_workers": self._shared[0].n_workers if self._shared is not None else 1,
            "api_worker": self._shared[1] if self._shared is not None else 0,  # who answered
            "requests_running": snap["num_running"],
            "requests_waiting": snap["num_waiting"],
            "requests_received_total": c.requests_received,
            "requests_finished_total": c.requests_finished,
            "requests_aborted_total": c.requests_aborted,
            "prompt_tokens_total": c.prompt_tokens,
            "generated_tokens_total": c.generated_tokens,
            "steps_total": c.steps,
            "prefill_steps_total": c.prefill_steps,
            "decode_steps_total": c.decode_steps,
            "step_errors_total": c.step_errors,
            "kv_blocks_total": snap["kv_blocks_total"],
            "kv_blocks_free": snap["kv_blocks_free"],
            "kv_blocks_used": snap["kv_blocks_used"],
            "kv_cache_usage": snap["kv_blocks_used"] / total,
            "kv_block_utilization": snap["kv_block_utilization"],
            "spec_drafted_total": int(snap.get("spec_drafted", 0)),
            "spec_accepted_total": int(snap.get("spec_accepted", 0)),
            **c.latency_metrics(),
        }

    # ---- loop-thread helpers ------------------------------------------------------------------
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

    def _post(self, fn: Callable[..., None], *args: object) -> None:
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            pass

    # ---- reader thread --------------------------------------------------------------------------
    def _read_loop(self) -> None:
        try:
            while not self._stopping.is_set():
                msg = self.core.recv(timeout=0.5)
                if msg is None:
                    if not self.core.alive:
                        raise EngineNotRunningError("engine core process died")
                    continue
                kind = msg[0]
                if kind == "step":
                    self._on_step(msg[1], msg[2])
                elif kind == "idle":
                    self._snapshot = msg[1]
                elif kind == "failed":
                    self._counters.step_errors += 1
                    self._publish()
                    with self._reqs_lock:
                        for rid in msg[1]:
                            self._reqs.pop(rid, None)
                    self._post(self._fail, list(msg[1]), RuntimeError(msg[2]))
                elif kind == "fatal":
                    raise RuntimeError(f"engine core died:\n{msg[1]}")
        except BaseException as exc:  # noqa: BLE001 - consumers must not hang
            if not self._stopping.is_set():
                self._post(self._fail_all, exc)

    def _on_step(self, rows: list[tuple], snapshot: dict[str, int | float]) -> None:
        """One engine step: append tokens, detokenize the batch, apply stop strings, deliver."""
        self._snapshot = snapshot
        c = self._counters
        c.steps += 1
        now = time.perf_counter()
        live: list[tuple[_ClientRequest, list[int], bool, str | None]] = []
        with self._reqs_lock:
            for rid, toks, finished, reason in rows:
                cr = self._reqs.get(rid)
                if cr is None or cr.finished:
                    continue  # aborted or already stopped by a stop string; core lag
                cr.output_ids.extend(toks)  # several per step under speculative decoding
                if cr.first_token is None:
                    cr.first_token = now
                live.append((cr, toks, finished, reason))
        if not live:
            return
        c.generated_tokens += sum(len(t) for _, t, _, _ in live)
        if any(cr.first_token == now and len(cr.output_ids) == len(t) for cr, t, _, _ in live):
            c.prefill_steps += 1
        else:
            c.decode_steps += 1
        deltas = self.detok.update_batch(
            [cr.request_id for cr, _, _, _ in live], [cr.output_ids for cr, _, _, _ in live],
            [cr.params.stop for cr, _, _, _ in live], finals=[fin for _, _, fin, _ in live])
        outputs: list[RequestOutput] = []
        for (cr, toks, finished, reason), (delta, matched) in zip(live, deltas, strict=True):
            if matched is not None and not finished:
                finished, reason = True, FinishReason.STOP.value
                self.core.send(("abort", cr.request_id))  # the core does not see stop strings
            fr = FinishReason(reason) if reason else None
            metrics: dict = {}
            if finished:
                cr.finished = True
                c.requests_finished += 1
                n_out = len(cr.output_ids)
                first = cr.first_token or now
                metrics = {"num_prompt_tokens": cr.prompt_len, "num_output_tokens": n_out,
                           "ttft_s": first - cr.arrival, "e2e_s": now - cr.arrival}
                if n_out > 1:
                    metrics["tpot_s"] = (now - first) / (n_out - 1)
                c.observe(metrics["ttft_s"], metrics.get("tpot_s"), metrics["e2e_s"])
                self.detok.reset(cr.request_id)
                with self._reqs_lock:
                    self._reqs.pop(cr.request_id, None)
            # `output_token_ids` is the request's live list, not a copy: at 200 streams and
            # 400 steps/s the per-token copy of a growing list was a quarter of a core, and
            # the consumers (the SSE route, the client) only read it.
            outputs.append(RequestOutput(
                request_id=cr.request_id, new_token_ids=toks, output_token_ids=cr.output_ids,
                finished=finished, finish_reason=fr, text_delta=delta, metrics=metrics))
        self._publish()
        self._post(self._deliver, outputs)
