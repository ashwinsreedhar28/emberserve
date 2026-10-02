"""In-process benchmark driver: replay a trace straight into an `LLMEngine`.

No HTTP, no tokenizer needed (traces carry token ids). Requests are added at their
`arrival_s` offsets against `time.perf_counter()`; the engine is stepped in between.
`static_batching=True` emulates the pre-Orca world: a new batch is admitted only when
the engine has NO unfinished requests, so the batch runs until its longest member ends.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from typing import TYPE_CHECKING

from emberserve.bench.metrics import RequestRecord
from emberserve.bench.trace import TraceRequest
from emberserve.sched.request import SamplingParams

if TYPE_CHECKING:
    from emberserve.engine import LLMEngine, StepStats


def run_offline_benchmark(engine: LLMEngine, trace: list[TraceRequest],
                          sampling: SamplingParams | None = None,
                          static_batching: bool = False,
                          progress: bool = False) -> tuple[list[RequestRecord], list[StepStats]]:
    """Returns (records, step_stats for this run).

    `sampling` is a template: its `max_tokens` is replaced by each trace request's
    `output_len` and `ignore_eos` is forced on, so the engine generates exactly the
    trace's output lengths (what the vLLM baseline does with `ignore_eos=True`).
    """
    template = sampling or SamplingParams.greedy()
    engine.keep_stats = True
    stats_start = len(engine.stats)
    pending: deque[TraceRequest] = deque(sorted(trace, key=lambda r: r.arrival_s))
    max_batch = engine.config.max_num_seqs
    recs: dict[str, RequestRecord] = {}
    t0 = time.perf_counter()
    n_done = 0

    def admit(req: TraceRequest, now: float) -> None:  # noqa: ARG001 - admission time
        sp = dataclasses.replace(template, max_tokens=req.output_len, ignore_eos=True)
        assert req.prompt_ids is not None, "offline benchmark needs prompt ids in the trace"
        # Latency counts from when the request was offered, not from when it was admitted:
        # a static batch's queue wait (and a continuous batch's wait for the running step)
        # used to be left out, which flattered static batching.
        arrived = t0 + req.arrival_s
        engine.add_request(req.request_id, req.prompt_ids, sp, arrival_time=arrived)
        recs[req.request_id] = RequestRecord(req.request_id, arrived, None, None,
                                             req.prompt_len, 0)

    while pending or engine.has_unfinished_requests():
        now = time.perf_counter()
        elapsed = now - t0
        if static_batching:
            if not engine.has_unfinished_requests():
                n = 0
                while pending and pending[0].arrival_s <= elapsed and n < max_batch:
                    admit(pending.popleft(), now)
                    n += 1
        else:
            while pending and pending[0].arrival_s <= elapsed:
                admit(pending.popleft(), now)
        if not engine.has_unfinished_requests():
            # Idle: sleep until the next arrival.
            if pending:
                time.sleep(max(0.0, pending[0].arrival_s - (time.perf_counter() - t0)))
            continue
        outs = engine.step()
        now = time.perf_counter()
        for out in outs:
            rec = recs[out.request_id]
            if rec.first_token_s is None:
                rec.first_token_s = now
            rec.output_tokens = len(out.output_token_ids)
            if out.finished:
                rec.finish_s = now
                m = out.metrics
                if "num_output_tokens" in m:
                    rec.output_tokens = m["num_output_tokens"]
                n_done += 1
                if progress and n_done % 10 == 0:
                    print(f"\r[offline] {n_done}/{len(trace)} done", end="", flush=True)
    if progress:
        print()
    records = [recs[r.request_id] for r in trace]
    for r in records:
        if r.finish_s is None:
            r.success = False
            r.error = "unfinished"
    return records, list(engine.stats[stats_start:])
