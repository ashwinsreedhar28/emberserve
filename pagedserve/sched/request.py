"""Request lifecycle types shared by the scheduler, engine, and server.

A Request moves WAITING -> RUNNING -> FINISHED (or back to WAITING on preemption).
Token ids are the source of truth; detokenization happens at the edge (engine/server).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 1.0  # 0.0 -> greedy
    top_k: int = -1  # -1 -> disabled
    top_p: float = 1.0  # 1.0 -> disabled
    repetition_penalty: float = 1.0  # 1.0 -> disabled; >1 penalizes tokens already in prompt+output
    stop_token_ids: list[int] = field(default_factory=list)
    stop: list[str] = field(default_factory=list)  # stop strings, checked by the engine on decoded text
    ignore_eos: bool = False
    seed: int | None = None  # per-request RNG seed for reproducible sampling
    logprobs: int | None = None  # reserved for the server

    def __post_init__(self) -> None:
        assert self.max_tokens >= 1
        assert self.temperature >= 0.0
        assert self.top_k == -1 or self.top_k >= 1
        assert 0.0 < self.top_p <= 1.0
        assert self.repetition_penalty > 0.0

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    @classmethod
    def greedy(cls, max_tokens: int = 128, **kw) -> "SamplingParams":
        return cls(max_tokens=max_tokens, temperature=0.0, **kw)


class RequestState(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"  # evicted from RUNNING; recompute from scratch when re-admitted
    FINISHED = "finished"


class FinishReason(str, Enum):
    STOP = "stop"  # eos or stop token / stop string
    LENGTH = "length"  # max_tokens reached or model length limit
    ABORT = "abort"  # client disconnected / abort_request


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    # Integer id used by the BlockManager / attention backends. Assigned by the engine
    # (monotonic counter) when the request is added; -1 until then.
    seq_id: int = -1
    arrival_time: float = field(default_factory=time.perf_counter)
    state: RequestState = RequestState.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    # Number of prompt+output tokens whose K/V are already in the cache. For a fresh
    # request this is 0; after prefill it equals len(prompt_token_ids); it grows by one
    # per decode step. With prefix caching it can start > 0 (cached prefix blocks).
    num_computed_tokens: int = 0
    # Timestamps for TTFT/TPOT metrics.
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    finished_time: float | None = None
    # Free-form metadata for the server (e.g. prompt text, chat template info).
    metadata: dict = field(default_factory=dict)

    # ---- derived ---------------------------------------------------------
    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """Total tokens (prompt + generated) known so far."""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def is_prefill(self) -> bool:
        """True while some prompt tokens still need their K/V computed."""
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.state == RequestState.FINISHED

    def get_token(self, idx: int) -> int:
        return self.prompt_token_ids[idx] if idx < self.num_prompt_tokens \
            else self.output_token_ids[idx - self.num_prompt_tokens]

    def append_output(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def reset_for_recompute(self) -> None:
        """Preemption by recompute: keep generated tokens, drop cache position."""
        self.num_computed_tokens = 0
        self.state = RequestState.PREEMPTED


@dataclass
class RequestOutput:
    """Emitted by the engine after each step for every request that produced a token
    or finished. `new_token_ids` is the delta since the previous output."""

    request_id: str
    new_token_ids: list[int]
    output_token_ids: list[int]
    finished: bool
    finish_reason: FinishReason | None = None
    text_delta: str = ""  # filled by the engine's incremental detokenizer
    metrics: dict = field(default_factory=dict)
