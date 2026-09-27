"""Continuous-batching scheduler (vLLM-v0 style, prefill priority).

Each `schedule()` call produces ONE step: either a prefill batch (new / re-admitted
requests, FIFO, bounded by seqs, token budget and free KV blocks) or a decode step
over every running request (one token each). Decode preempts the youngest running
request by recompute when the KV cache is full.

The scheduler owns the BlockManager because admission depends on block availability.
It never touches `num_computed_tokens` (the engine advances it after the model step)
except through `Request.reset_for_recompute()` on preemption.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from pagedserve.config import EngineConfig
from pagedserve.kv.block_manager import BlockManager, OutOfBlocksError
from pagedserve.sched.request import FinishReason, Request, RequestState


@dataclass
class SchedulerOutput:
    scheduled: list[Request]  # requests in this step, packed order
    is_prefill: bool
    query_lens: list[int]  # per scheduled request: tokens computed this step
    preempted: list[Request] = field(default_factory=list)  # evicted during this call

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager) -> None:
        self.config = config
        self.block_manager = block_manager
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # admission order, oldest first
        self._by_id: dict[str, Request] = {}  # unfinished requests only

    # ---- queue management ----------------------------------------------------
    def add_request(self, req: Request) -> None:
        """Enqueue a request. Raises ValueError for prompts the engine can never serve."""
        assert req.seq_id >= 0, "engine must assign seq_id before add_request"
        n = req.num_prompt_tokens
        if n == 0:
            raise ValueError("empty prompt")
        if n >= self.config.max_model_len:
            raise ValueError(f"prompt has {n} tokens, max_model_len is {self.config.max_model_len}")
        if n > self.config.max_num_batched_tokens:
            raise ValueError(f"prompt has {n} tokens, exceeds max_num_batched_tokens "
                             f"{self.config.max_num_batched_tokens}")
        if req.request_id in self._by_id:
            raise ValueError(f"duplicate request_id {req.request_id!r}")
        req.state = RequestState.WAITING
        self.waiting.append(req)
        self._by_id[req.request_id] = req

    def abort_request(self, request_id: str) -> Request | None:
        """Drop a request in any unfinished state. Returns it, or None if unknown."""
        req = self._by_id.get(request_id)
        if req is None:
            return None
        if req in self.waiting:  # WAITING or PREEMPTED
            self.waiting.remove(req)
        self._finish(req, FinishReason.ABORT)
        return req

    def finish_request(self, req: Request, reason: FinishReason) -> None:
        """Retire a request that produced its last token. Idempotent."""
        if req.is_finished:
            return
        if req in self.waiting:
            self.waiting.remove(req)
        self._finish(req, reason)

    def _finish(self, req: Request, reason: FinishReason) -> None:
        if req in self.running:
            self.running.remove(req)
        self.block_manager.free(req.seq_id)
        req.state = RequestState.FINISHED
        req.finish_reason = reason
        req.finished_time = time.perf_counter()
        self._by_id.pop(req.request_id, None)

    # ---- queries -------------------------------------------------------------
    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def get_request(self, request_id: str) -> Request | None:
        return self._by_id.get(request_id)

    # ---- scheduling ----------------------------------------------------------
    def schedule(self) -> SchedulerOutput:
        """Build the next step. Prefill wins whenever the head of `waiting` fits."""
        if self.waiting:
            out = self._schedule_prefill()
            if not out.is_empty:
                return out
        if self.running:
            return self._schedule_decode()
        return SchedulerOutput([], False, [])

    def _schedule_prefill(self) -> SchedulerOutput:
        cfg = self.config
        bm = self.block_manager
        batch: list[Request] = []
        query_lens: list[int] = []
        budget = 0
        now = time.perf_counter()
        while self.waiting:
            req = self.waiting[0]
            # A preempted request has num_computed_tokens == 0, so it re-prefills its
            # whole prompt+output history.
            query_len = req.num_tokens - req.num_computed_tokens
            # Admitted requests are already in `running`, so it alone counts the seqs.
            if len(self.running) >= cfg.max_num_seqs:
                break
            if budget + query_len > cfg.max_num_batched_tokens:
                break
            if not bm.can_allocate(req.num_tokens):
                break
            self.waiting.popleft()
            bm.allocate(req.seq_id, req.num_tokens)
            req.state = RequestState.RUNNING
            if req.first_scheduled_time is None:
                req.first_scheduled_time = now
            self.running.append(req)
            batch.append(req)
            query_lens.append(query_len)
            budget += query_len
        return SchedulerOutput(batch, True, query_lens)

    def _schedule_decode(self) -> SchedulerOutput:
        """Reserve one slot per running request, preempting the youngest on overflow."""
        bm = self.block_manager
        preempted: list[Request] = []
        i = 0
        while i < len(self.running):
            req = self.running[i]
            try:
                bm.append_slots(req.seq_id, 1)
            except OutOfBlocksError:
                victim = self._preempt_youngest()
                preempted.append(victim)
                if victim is req:
                    break  # nothing younger left to evict; `req` waits for re-prefill
                continue  # retry the same request with the freed blocks
            i += 1
        # Every request still in `running` has its slot; victims were removed.
        scheduled = list(self.running)
        return SchedulerOutput(scheduled, False, [1] * len(scheduled), preempted)

    def _preempt_youngest(self) -> Request:
        victim = self.running.pop()
        self.block_manager.free(victim.seq_id)
        victim.reset_for_recompute()
        # Victims are evicted youngest-first, so appendleft keeps their original order.
        self.waiting.appendleft(victim)
        return victim
