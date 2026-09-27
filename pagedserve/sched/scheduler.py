"""Continuous-batching scheduler (vLLM-v0 style, prefill priority).

Each `schedule()` call produces ONE step: either a prefill batch (new / re-admitted
requests, FIFO, bounded by seqs, token budget and free KV blocks) or a decode step
over every running request (one token each). Decode preempts the youngest running
request by recompute when the KV cache is full.

The scheduler owns the BlockManager because admission depends on block availability.
It never touches `num_computed_tokens` (the engine advances it after the model step)
except through `Request.reset_for_recompute()` on preemption and, with prefix caching, by
setting it to the cached-prefix length on admission (those tokens' K/V already exist).

Prefix caching: every `schedule()` first publishes the hashes of the full blocks each
running request has computed so far (its previous step wrote them), and so does
`finish_request` right before freeing, so finished requests leave reusable blocks behind.
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
    # Route through the prefill attention path (per-sequence causal attention over the
    # gathered context). True iff some query_len > 1 or some scheduled request is still
    # mid-prefill after this step. A step of pure decode slots is False, so CUDA graphs
    # apply to it. Not "the engine samples for every row": that is `prefill_complete`.
    is_prefill: bool
    query_lens: list[int]  # per scheduled request: tokens computed this step
    preempted: list[Request] = field(default_factory=list)  # evicted during this call
    # Per scheduled request: True when this step computes its last outstanding token
    # (num_computed_tokens + query_len == num_tokens), i.e. the engine samples for it.
    # False only for a partial prefill chunk (chunked prefill), which emits nothing.
    # Defaults to all-True, which is what the non-chunked paths produce.
    prefill_complete: list[bool] = field(default_factory=list)
    # Rows that are decode slots (one new token for a request whose prefill is done).
    # `num_tokens - num_decode_tokens` is the number of prompt/recompute tokens.
    num_decode_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.prefill_complete:
            self.prefill_complete = [True] * len(self.scheduled)
        assert len(self.prefill_complete) == len(self.query_lens) == len(self.scheduled)

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def num_prefill_tokens(self) -> int:
        return self.num_tokens - self.num_decode_tokens

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager) -> None:
        self.config = config
        self.block_manager = block_manager
        if config.enable_prefix_caching:
            # The engine builds a plain BlockManager; the scheduler owns the feature flag.
            block_manager.enable_prefix_caching()
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
        # With chunked prefill a prompt longer than the budget is simply split across steps.
        if not self.config.enable_chunked_prefill and n > self.config.max_num_batched_tokens:
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
        self._register_computed_blocks(req)
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
        if self.config.enable_prefix_caching:
            for req in self.running:
                self._register_computed_blocks(req)
        if self.config.enable_chunked_prefill:
            return self._schedule_chunked()
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
            # whole prompt+output history (minus whatever prefix the cache still holds).
            match = bm.match_prefix(req.all_token_ids) if cfg.enable_prefix_caching else None
            num_cached = match.num_cached_tokens if match is not None else 0
            query_len = req.num_tokens - num_cached
            # Admitted requests are already in `running`, so it alone counts the seqs.
            if len(self.running) >= cfg.max_num_seqs:
                break
            if budget + query_len > cfg.max_num_batched_tokens:
                break
            if match is None:
                if not bm.can_allocate(req.num_tokens):
                    break
            elif not bm.can_allocate_with_prefix(req.all_token_ids, match):
                break
            self.waiting.popleft()
            if match is None:
                bm.allocate(req.seq_id, req.num_tokens)
            else:
                _, req.num_computed_tokens = bm.allocate_with_prefix(
                    req.seq_id, req.all_token_ids, match)
            req.state = RequestState.RUNNING
            if req.first_scheduled_time is None:
                req.first_scheduled_time = now
            self.running.append(req)
            batch.append(req)
            query_lens.append(query_len)
            budget += query_len
        return SchedulerOutput(batch, True, query_lens)

    def _finishes_on_resolve(self, req: Request) -> bool:
        """Async scheduling: True when the token this request is still waiting to read back
        will end it on a length limit (`check_stop`'s rules with that token counted), so
        scheduling another decode slot for it would only compute a discarded token. EOS
        cannot be anticipated (the token is on the device), so EOS-ended requests do
        compute one extra token; length-ended ones never do."""
        if req.pending_row is None:
            return False
        return (req.num_output_tokens + 1 >= req.sampling_params.max_tokens
                or req.num_tokens + 1 >= self.config.max_model_len)

    def _schedule_decode(self) -> SchedulerOutput:
        """Reserve one slot per running request, preempting the youngest on overflow."""
        bm = self.block_manager
        preempted: list[Request] = []
        scheduled: list[Request] = []
        query_lens: list[int] = []
        i = 0
        while i < len(self.running):
            req = self.running[i]
            if self._finishes_on_resolve(req):
                i += 1
                continue
            n_slots = 1 + len(req.draft_tokens)  # speculative drafts ride in the same step
            try:
                bm.append_slots(req.seq_id, n_slots)
            except OutOfBlocksError:
                victim = self._preempt_youngest()
                preempted.append(victim)
                if victim is req:
                    break  # nothing younger left to evict; `req` waits for re-prefill
                continue  # retry the same request with the freed blocks
            scheduled.append(req)
            query_lens.append(n_slots)
            i += 1
        # Every scheduled request has its slot(s); victims were removed from `running`.
        # A draft row has query_len > 1 and attends through the cache like a chunk, so the
        # step routes through the prefill attention path when any request drafted.
        return SchedulerOutput(scheduled, any(q > 1 for q in query_lens), query_lens, preempted,
                               num_decode_tokens=sum(query_lens))

    # ---- chunked prefill -------------------------------------------------------------
    def _prefill_remaining(self, req: Request) -> int:
        """Reserved-but-unwritten slots of a running request: > 0 while its prefill (or
        re-prefill after preemption) is still in progress, 0 once every reserved slot
        holds K/V and the request needs a fresh decode slot.

        The block manager's reserved count is the reference, not `num_tokens`: after a
        completed prefill the sampled token is in `output_token_ids` but not yet in the
        cache, so `num_tokens - num_computed_tokens == 1` there as well as for a request
        whose last outstanding prompt token happens to be its own one-token chunk.
        """
        return self.block_manager.get_num_tokens(req.seq_id) - req.num_computed_tokens

    def _schedule_chunked(self) -> SchedulerOutput:
        """One mixed step: a decode slot for every running request whose prefill is done,
        then prefill chunks from the remaining token budget (running requests still
        mid-prefill first, oldest first; then new requests FIFO, admitted with their
        blocks for the full prompt and possibly a partial first chunk).

        Every step obeys `num_tokens <= max_num_batched_tokens`. Decode slots are never
        chunked away: if the running batch alone fills the budget, prefill waits.
        """
        cfg = self.config
        bm = self.block_manager
        budget = cfg.max_num_batched_tokens
        preempted: list[Request] = []
        # 1. Decode slots, youngest-first recompute-preemption on overflow (same as
        # `_schedule_decode`); mid-prefill requests are skipped, they hold their blocks.
        decode_rows: list[Request] = []
        i = 0
        while i < len(self.running):
            req = self.running[i]
            if self._prefill_remaining(req) > 0 or self._finishes_on_resolve(req):
                i += 1
                continue
            try:
                bm.append_slots(req.seq_id, 1 + len(req.draft_tokens))
            except OutOfBlocksError:
                # The victim is the youngest running request, which sits at or after `i`
                # (possibly a mid-prefill one), so `decode_rows` never loses a member.
                victim = self._preempt_youngest()
                preempted.append(victim)
                if victim is req:
                    break
                continue
            decode_rows.append(req)
            i += 1
        scheduled = list(decode_rows)
        query_lens = [1 + len(r.draft_tokens) for r in decode_rows]
        complete = [True] * len(decode_rows)
        budget_left = budget - sum(query_lens)
        # 2a. Running requests still mid-prefill, oldest first. (A decode row's slot was
        # just reserved, so its `_prefill_remaining` reads 1 now; skip those by identity.)
        decoding = {id(r) for r in decode_rows}
        for req in self.running:
            if budget_left <= 0:
                break
            if id(req) in decoding:
                continue
            remaining = self._prefill_remaining(req)
            if remaining <= 0:
                continue
            chunk = min(remaining, budget_left)
            scheduled.append(req)
            query_lens.append(chunk)
            complete.append(chunk == remaining)
            budget_left -= chunk
        # 2b. New / re-admitted requests, FIFO. Admission reserves blocks for the whole
        # prompt (plus kept outputs for a preempted request) exactly as `_schedule_prefill`
        # does; only the number of tokens computed this step is capped.
        now = time.perf_counter()
        while self.waiting and budget_left > 0:
            req = self.waiting[0]
            if len(self.running) >= cfg.max_num_seqs:
                break
            match = bm.match_prefix(req.all_token_ids) if cfg.enable_prefix_caching else None
            if match is None:
                if not bm.can_allocate(req.num_tokens):
                    break
            elif not bm.can_allocate_with_prefix(req.all_token_ids, match):
                break
            self.waiting.popleft()
            if match is None:
                bm.allocate(req.seq_id, req.num_tokens)
            else:
                _, req.num_computed_tokens = bm.allocate_with_prefix(
                    req.seq_id, req.all_token_ids, match)
            req.state = RequestState.RUNNING
            if req.first_scheduled_time is None:
                req.first_scheduled_time = now
            self.running.append(req)
            remaining = self._prefill_remaining(req)  # >= 1: match_prefix leaves one token
            chunk = min(remaining, budget_left)
            scheduled.append(req)
            query_lens.append(chunk)
            complete.append(chunk == remaining)
            budget_left -= chunk
        # Prefill routing is needed for multi-token rows and for rows that stay mid-prefill
        # (their next chunk must find this one's K/V in the cache). A completing 1-token
        # chunk is indistinguishable from a decode slot to every backend, so a step made
        # only of those keeps the decode path (and CUDA graphs).
        is_prefill = any(q > 1 for q in query_lens) or not all(complete)
        return SchedulerOutput(scheduled, is_prefill=is_prefill, query_lens=query_lens,
                               preempted=preempted, prefill_complete=complete,
                               num_decode_tokens=sum(query_lens[:len(decode_rows)]))

    def _register_computed_blocks(self, req: Request) -> None:
        """Publish hashes for the full blocks whose K/V `req` has already written."""
        bm = self.block_manager
        if self.config.enable_prefix_caching and bm.has_sequence(req.seq_id):
            bm.register_full_blocks(req.seq_id, req.all_token_ids[:req.num_computed_tokens])

    def _preempt_youngest(self) -> Request:
        victim = self.running.pop()
        # Its computed blocks stay cached (evictable), so re-admission gets prefix hits.
        self._register_computed_blocks(victim)
        self.block_manager.free(victim.seq_id)
        victim.reset_for_recompute()
        # Victims are evicted youngest-first, so appendleft keeps their original order.
        self.waiting.appendleft(victim)
        return victim
