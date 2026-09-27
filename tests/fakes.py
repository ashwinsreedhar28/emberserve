"""Tensor-free driver for the scheduler: a deterministic "model" plus two engine loops.

`FakeEngineLoop` drives Scheduler + BlockManager the way the real engine does
(schedule -> advance num_computed_tokens -> emit one token -> check_stop -> finish), so
scheduler behaviour can be tested end to end. `StaticBatchLoop` is the baseline that
only admits a new batch once the previous one has completely drained.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from pagedserve.config import EngineConfig
from pagedserve.kv.block_manager import BlockManager
from pagedserve.sampling import check_stop
from pagedserve.sched.request import Request, SamplingParams
from pagedserve.sched.scheduler import Scheduler, SchedulerOutput

EOS_TOKEN_ID = 1


def next_token(last_token: int, vocab: int = 256) -> int:
    """Deterministic stand-in for the model: the next token is a function of the last."""
    return (last_token * 7 + 3) % vocab


def expected_output(prompt_ids: list[int], max_tokens: int, vocab: int = 256) -> list[int]:
    """Continuation `next_token` produces for `prompt_ids`, ignoring eos."""
    out: list[int] = []
    tok = prompt_ids[-1]
    for _ in range(max_tokens):
        tok = next_token(tok, vocab)
        out.append(tok)
    return out


@dataclass
class StepRecord:
    """What one `step()` scheduled, captured before the fake model ran."""

    step: int
    is_prefill: bool
    scheduled: list[str]
    query_lens: list[int]
    num_tokens_at_schedule: list[int]  # req.num_tokens when it was scheduled
    preempted: list[str] = field(default_factory=list)


class FakeEngineLoop:
    """Continuous-batching loop: every request is handed to the scheduler on arrival."""

    def __init__(self, num_blocks: int, block_size: int = 4, max_num_seqs: int = 256,
                 max_num_batched_tokens: int = 8192, max_model_len: int = 4096) -> None:
        self.config = EngineConfig(block_size=block_size, num_gpu_blocks=num_blocks,
                                   max_num_seqs=max_num_seqs,
                                   max_num_batched_tokens=max_num_batched_tokens,
                                   max_model_len=max_model_len)
        self.block_manager = BlockManager(num_blocks, block_size)
        self.scheduler = Scheduler(self.config, self.block_manager)
        self.requests: dict[str, Request] = {}  # every request ever submitted
        self.outputs: dict[str, list[int]] = {}
        self.log: list[StepRecord] = []
        self.num_steps = 0
        self._next_seq_id = 0
        self._arrivals: list[tuple[int, Request]] = []  # (arrival step, request)

    # ---- request intake --------------------------------------------------------
    def _make_request(self, prompt_ids: list[int], max_tokens: int, ignore_eos: bool) -> Request:
        req = Request(request_id=f"r{self._next_seq_id}", prompt_token_ids=list(prompt_ids),
                      sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                                     ignore_eos=ignore_eos),
                      seq_id=self._next_seq_id)
        self._next_seq_id += 1
        self.requests[req.request_id] = req
        return req

    def add(self, prompt_ids: list[int], max_tokens: int, ignore_eos: bool = True) -> str:
        """Submit a request now. Returns its request_id."""
        req = self._make_request(prompt_ids, max_tokens, ignore_eos)
        self._admit(req)
        return req.request_id

    def add_at(self, step: int, prompt_ids: list[int], max_tokens: int,
               ignore_eos: bool = True) -> str:
        """Submit a request that arrives just before `step` runs (0 = the next step)."""
        req = self._make_request(prompt_ids, max_tokens, ignore_eos)
        self._arrivals.append((step, req))
        self._arrivals.sort(key=lambda a: a[0])
        return req.request_id

    def _admit(self, req: Request) -> None:
        self.scheduler.add_request(req)

    def _deliver_arrivals(self) -> None:
        while self._arrivals and self._arrivals[0][0] <= self.num_steps:
            _, req = self._arrivals.pop(0)
            self._admit(req)

    # ---- stepping --------------------------------------------------------------
    def step(self) -> SchedulerOutput:
        """One engine iteration: schedule, "run the model", emit a token per request."""
        self._deliver_arrivals()
        self._before_schedule()
        out = self.scheduler.schedule()
        self.log.append(StepRecord(
            step=self.num_steps, is_prefill=out.is_prefill,
            scheduled=[r.request_id for r in out.scheduled], query_lens=list(out.query_lens),
            num_tokens_at_schedule=[r.num_tokens for r in out.scheduled],
            preempted=[r.request_id for r in out.preempted]))
        for req, query_len in zip(out.scheduled, out.query_lens):
            req.num_computed_tokens += query_len
            tok = next_token(req.all_token_ids[-1])
            req.append_output(tok)
            self.outputs[req.request_id] = req.output_token_ids
            reason = check_stop(req, tok, EOS_TOKEN_ID, self.config.max_model_len)
            if reason is not None:
                self.scheduler.finish_request(req, reason)
        self.num_steps += 1
        return out

    def _before_schedule(self) -> None:
        """Hook for subclasses that gate admission."""

    def is_done(self) -> bool:
        return not self._arrivals and not self.scheduler.has_unfinished_requests()

    def run_until_done(self, max_steps: int = 10_000) -> int:
        """Step until every submitted request finished. Returns the number of steps taken."""
        start = self.num_steps
        while not self.is_done():
            if self.num_steps - start >= max_steps:
                raise RuntimeError(f"not done after {max_steps} steps")
            self.step()
        return self.num_steps - start


class StaticBatchLoop(FakeEngineLoop):
    """Baseline: a batch is formed only when the scheduler is idle, and nothing joins it
    until every member has finished. Admission mirrors the scheduler's own prefill rule."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pending: list[Request] = []

    def _admit(self, req: Request) -> None:
        self._pending.append(req)

    def _before_schedule(self) -> None:
        if self.scheduler.has_unfinished_requests() or not self._pending:
            return
        cfg, bm = self.config, self.block_manager
        batch: list[Request] = []
        budget = blocks = 0
        for req in self._pending:
            if len(batch) >= cfg.max_num_seqs:
                break
            if budget + req.num_prompt_tokens > cfg.max_num_batched_tokens:
                break
            if blocks + bm.blocks_needed(req.num_prompt_tokens) > bm.num_free_blocks:
                break
            batch.append(req)
            budget += req.num_prompt_tokens
            blocks += bm.blocks_needed(req.num_prompt_tokens)
        del self._pending[:len(batch)]
        for req in batch:
            self.scheduler.add_request(req)

    def is_done(self) -> bool:
        return super().is_done() and not self._pending


def poisson_arrivals(n: int, rate: float, rng: random.Random) -> list[int]:
    """`n` arrival steps with exponential inter-arrival times at `rate` per step."""
    t = 0.0
    steps: list[int] = []
    for _ in range(n):
        t += rng.expovariate(rate)
        steps.append(int(t))
    return steps
