"""A model-free engine with `LLMEngine`'s surface, for benchmarking the API layer alone.

`EngineSpec(engine_config, fake_step_ms=2.5)` makes the engine-core process run this
instead of a model: every `step_ms` it hands each running request one more token (ids
drawn from a fixed generator) and finishes it at its `max_tokens`. The pipe, the
unpickling, the detokenizer, the per-request queues, the SSE encoding and the socket
writes are then the whole cost, which is what the 0.5B saturation point is limited by
(the real core spends 39% of its active time blocked on the pipe to the API process).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from emberserve.config import EngineConfig, ModelConfig
from emberserve.dist import get_tp
from emberserve.sched.request import FinishReason, RequestOutput, SamplingParams


@dataclass
class _Fake:
    request_id: str
    prompt_len: int
    params: SamplingParams
    arrival_time: float
    output_ids: list[int] = field(default_factory=list)
    first_token_time: float | None = None


class _Stats:
    def __init__(self, n: int) -> None:
        self.num_blocks = n
        self.num_free = n
        self.num_used = 0
        self.utilization = 0.0


class _BlockManager:
    def __init__(self, n: int = 4096) -> None:
        self._stats = _Stats(n)

    def stats(self) -> _Stats:
        return self._stats


class _Scheduler:
    def __init__(self) -> None:
        self.running: list[_Fake] = []
        self.waiting: list[_Fake] = []

    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    def has_unfinished_requests(self) -> bool:
        return bool(self.running or self.waiting)


class FakeEngine:
    """`LLMEngine` as the engine-core loop sees it, with a clock instead of a model."""

    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig,
                 step_ms: float = 2.5, seed: int = 0) -> None:
        self.model_config = model_config
        self.config = engine_config
        self.step_s = step_ms / 1e3
        self.scheduler = _Scheduler()
        self.block_manager = _BlockManager()
        self.eos_token_ids = frozenset({model_config.eos_token_id})
        self.keep_stats = False
        self.last_step_scheduled = False
        self.spec_drafted = 0
        self.spec_accepted = 0
        self.tp = get_tp()
        self._rng = random.Random(seed)
        self._next_step = time.perf_counter()

    # ---- request API (what engine_core calls) --------------------------------------------
    def add_request(self, request_id: str, prompt_ids: list[int], params: SamplingParams,
                    arrival_time: float | None = None) -> None:
        self.scheduler.waiting.append(_Fake(request_id, len(prompt_ids), params,
                                            arrival_time or time.perf_counter()))

    def abort_request(self, request_id: str) -> None:
        s = self.scheduler
        s.running = [r for r in s.running if r.request_id != request_id]
        s.waiting = [r for r in s.waiting if r.request_id != request_id]

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    def reset(self) -> None:
        self.scheduler.running.clear()
        self.scheduler.waiting.clear()

    def shutdown(self) -> None:
        pass

    def step(self) -> list[RequestOutput]:
        s = self.scheduler
        # keep the step cadence: sleep out the remainder of the step period
        now = time.perf_counter()
        if now < self._next_step:
            time.sleep(self._next_step - now)
        self._next_step = max(self._next_step + self.step_s, time.perf_counter())
        s.running.extend(s.waiting)
        s.waiting.clear()
        self.last_step_scheduled = bool(s.running)
        outs: list[RequestOutput] = []
        vocab = self.model_config.vocab_size
        still: list[_Fake] = []
        now = time.perf_counter()
        for r in s.running:
            tok = self._rng.randrange(2, vocab)
            r.output_ids.append(tok)
            if r.first_token_time is None:
                r.first_token_time = now
            finished = len(r.output_ids) >= r.params.max_tokens
            metrics = {}
            if finished:
                n = len(r.output_ids)
                metrics = {"num_prompt_tokens": r.prompt_len, "num_output_tokens": n,
                           "ttft_s": r.first_token_time - r.arrival_time, "e2e_s": now - r.arrival_time}
                if n > 1:
                    metrics["tpot_s"] = (now - r.first_token_time) / (n - 1)
            else:
                still.append(r)
            outs.append(RequestOutput(request_id=r.request_id, new_token_ids=[tok],
                                      output_token_ids=list(r.output_ids), finished=finished,
                                      finish_reason=FinishReason.LENGTH if finished else None,
                                      text_delta="", metrics=metrics))
        s.running = still
        return outs


class ByteTokenizer:
    """Bytes as tokens: enough for the detokenizer to run its real code path with the fake
    engine (`scripts/bench_api_layer.py`); module-level so API worker processes can unpickle
    it."""

    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode())

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return bytes(i % 256 for i in ids).decode(errors="replace")

    def decode_batch(self, batch: list[list[int]], skip_special_tokens: bool = False) -> list[str]:
        return [self.decode(ids, skip_special_tokens) for ids in batch]
