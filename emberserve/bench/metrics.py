"""Per-request records and their aggregate summary.

Metric names are fixed across every driver (offline, HTTP, vLLM baseline) so results
are comparable: `ttft_ms`, `tpot_ms`, `e2e_ms`, `throughput_tok_s`, `requests_per_s`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

PERCENTILES = (50, 90, 99)


@dataclass
class RequestRecord:
    request_id: str
    arrival_s: float  # when the request was issued (perf_counter seconds)
    first_token_s: float | None  # when the first output token arrived
    finish_s: float | None  # when the last token arrived
    prompt_tokens: int
    output_tokens: int
    success: bool = True
    error: str | None = None
    # Gaps between consecutive streamed chunks after the first (ms). One engine step per
    # chunk while the server keeps up, so their distribution is the server's step-time
    # distribution as a client sees it: the p90/p99 are the steps that carried prompts.
    chunk_gaps_ms: list[float] = field(default_factory=list)
    # When a failed request gave up (perf_counter seconds): failures still take time, and
    # the run's duration has to include it.
    end_s: float | None = None

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_s is None:
            return None
        return self.first_token_s - self.arrival_s

    @property
    def e2e_s(self) -> float | None:
        if self.finish_s is None:
            return None
        return self.finish_s - self.arrival_s

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token after the first one."""
        if self.finish_s is None or self.first_token_s is None or self.output_tokens < 2:
            return None
        return (self.finish_s - self.first_token_s) / (self.output_tokens - 1)


@dataclass
class Stat:
    mean: float = float("nan")
    p50: float = float("nan")
    p90: float = float("nan")
    p99: float = float("nan")

    @classmethod
    def of(cls, values: list[float]) -> "Stat":
        if not values:
            return cls()
        a = np.asarray(values, dtype=np.float64)
        p = np.percentile(a, PERCENTILES)
        return cls(mean=float(a.mean()), p50=float(p[0]), p90=float(p[1]), p99=float(p[2]))


@dataclass
class BenchSummary:
    num_requests: int
    completed: int
    failed: int
    duration_s: float
    requests_per_s: float
    throughput_tok_s: float  # output tokens per wall second
    total_throughput_tok_s: float  # prompt + output tokens per wall second
    ttft_ms: Stat = field(default_factory=Stat)
    tpot_ms: Stat = field(default_factory=Stat)
    e2e_ms: Stat = field(default_factory=Stat)
    goodput_rps: float | None = None
    slo_ttft_ms: float | None = None
    slo_tpot_ms: float | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    itl_ms: Stat = field(default_factory=Stat)  # inter-chunk gaps over every stream

    def to_dict(self) -> dict:
        return asdict(self)

    def one_line(self, name: str = "") -> str:
        itl = ""
        if self.itl_ms.p50 == self.itl_ms.p50:  # not NaN: a streamed run
            itl = f" itl p50/p90/p99={self.itl_ms.p50:.1f}/{self.itl_ms.p90:.1f}/{self.itl_ms.p99:.1f}ms"
        return (f"{name:>22s} tok/s={self.throughput_tok_s:8.1f} req/s={self.requests_per_s:6.2f}"
                f" ttft p50/p99={self.ttft_ms.p50:7.1f}/{self.ttft_ms.p99:7.1f}ms"
                f" tpot p50/p99={self.tpot_ms.p50:6.1f}/{self.tpot_ms.p99:6.1f}ms{itl}"
                f" ok={self.completed}/{self.num_requests}")


def summarize(records: list[RequestRecord], slo_ttft_ms: float | None = None,
              slo_tpot_ms: float | None = None, wall_s: float | None = None) -> BenchSummary:
    """Aggregate. `wall_s` defaults to (last end - first arrival): the last successful
    finish or the last failure, whichever is later, which is what a load generator measures
    from first send to last byte. (Failures used to be left out, so a run whose failed
    requests kept timing out after the last success had a shorter duration and a higher
    throughput than it delivered.) Only successful requests count as output."""
    ok = [r for r in records if r.success and r.finish_s is not None]
    if wall_s is None:
        ends = [r.finish_s for r in ok] + [r.end_s for r in records
                                           if not r.success and r.end_s is not None]
        if ends:
            t0 = min(r.arrival_s for r in records)
            wall_s = max(ends) - t0
        else:
            wall_s = 0.0
    wall = max(wall_s, 1e-9)
    ttft = [r.ttft_s * 1e3 for r in ok if r.ttft_s is not None]
    tpot = [r.tpot_s * 1e3 for r in ok if r.tpot_s is not None]
    e2e = [r.e2e_s * 1e3 for r in ok if r.e2e_s is not None]
    out_tok = sum(r.output_tokens for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)
    goodput = None
    if slo_ttft_ms is not None or slo_tpot_ms is not None:
        good = 0
        for r in ok:
            t, p = r.ttft_s, r.tpot_s
            if slo_ttft_ms is not None and (t is None or t * 1e3 > slo_ttft_ms):
                continue
            if slo_tpot_ms is not None and p is not None and p * 1e3 > slo_tpot_ms:
                continue
            good += 1
        goodput = good / wall
    return BenchSummary(
        num_requests=len(records), completed=len(ok), failed=len(records) - len(ok),
        duration_s=wall_s, requests_per_s=len(ok) / wall, throughput_tok_s=out_tok / wall,
        total_throughput_tok_s=(in_tok + out_tok) / wall,
        ttft_ms=Stat.of(ttft), tpot_ms=Stat.of(tpot), e2e_ms=Stat.of(e2e),
        goodput_rps=goodput, slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms,
        prompt_tokens=in_tok, output_tokens=out_tok,
        itl_ms=Stat.of([g for r in ok for g in r.chunk_gaps_ms]))


def records_to_json(records: list[RequestRecord]) -> list[dict]:
    return [asdict(r) for r in records]
