"""Synthetic ShareGPT-like request traces.

A trace is a list of `TraceRequest`s: prompt length, output length, arrival offset and
(for tokenizer-free runs) the prompt token ids. It is fully determined by its seed, so
every system in a comparison (pagedserve offline, pagedserve HTTP, vLLM) sees the SAME
requests in the SAME order at the SAME times.

    trace = generate_trace(n=200, seed=0, request_rate=4.0)
    json.dump(trace_to_json(trace), open("trace.json", "w"))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Words used to build text prompts of a target token length. Short common words that
# most BPE tokenizers map to a single token, so len(tokens) ~= len(words).
_WORDS = ("the quick brown fox jumps over the lazy dog while a small cat sleeps under "
          "an old oak tree near the river bank and the wind blows softly through tall "
          "green grass as birds sing in the bright morning light").split()

MIN_LEN = 4


@dataclass
class TraceRequest:
    request_id: str
    prompt_len: int
    output_len: int
    arrival_s: float
    prompt_ids: list[int] | None = None
    shared_prefix_len: int = 0


@dataclass(frozen=True)
class LogNormal:
    """Length distribution: lognormal with the given median and sigma (in log space)."""

    median: float
    sigma: float

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.lognormal(mean=float(np.log(self.median)), sigma=self.sigma, size=n)


DEFAULT_PROMPT_DIST = LogNormal(median=200.0, sigma=0.8)
DEFAULT_OUTPUT_DIST = LogNormal(median=150.0, sigma=0.7)


def generate_trace(n: int, seed: int = 0,
                   prompt_len_dist: LogNormal = DEFAULT_PROMPT_DIST,
                   output_len_dist: LogNormal = DEFAULT_OUTPUT_DIST,
                   request_rate: float | None = None,
                   shared_prefix_len: int = 0,
                   vocab_size: int = 151_936,
                   max_prompt_len: int = 2048,
                   max_output_len: int = 2048) -> list[TraceRequest]:
    """Draw `n` requests. Arrivals are Poisson at `request_rate` req/s (exponential
    inter-arrival gaps); `None` (or inf) puts every request at t=0.

    Prompt ids are random in [2, vocab_size) so they never contain eos/pad (0/1) for the
    tiny test model; a shared prefix of `shared_prefix_len` ids is drawn once from the
    seed and prepended to every prompt (counted inside prompt_len).
    """
    assert n >= 1
    rng = np.random.default_rng(seed)
    lo = max(MIN_LEN, shared_prefix_len + 1) if shared_prefix_len else MIN_LEN
    prompt_lens = np.clip(np.rint(prompt_len_dist.sample(rng, n)), lo, max_prompt_len)
    output_lens = np.clip(np.rint(output_len_dist.sample(rng, n)), MIN_LEN, max_output_len)
    if request_rate is None or request_rate == float("inf") or request_rate <= 0:
        arrivals = np.zeros(n)
    else:
        gaps = rng.exponential(scale=1.0 / request_rate, size=n)
        gaps[0] = 0.0
        arrivals = np.cumsum(gaps)
    prefix = rng.integers(2, vocab_size, size=shared_prefix_len).tolist()
    out: list[TraceRequest] = []
    for i in range(n):
        plen = int(prompt_lens[i])
        body = rng.integers(2, vocab_size, size=plen - shared_prefix_len).tolist()
        out.append(TraceRequest(
            request_id=f"req-{i:05d}", prompt_len=plen, output_len=int(output_lens[i]),
            arrival_s=float(arrivals[i]), prompt_ids=prefix + body,
            shared_prefix_len=shared_prefix_len))
    return out


def render_prompt(req: TraceRequest, tokenizer: Any = None) -> str | list[int]:
    """The prompt to send. With a tokenizer (anything with `.encode(str) -> list[int]`
    and `.decode(list[int]) -> str`) build TEXT of about `prompt_len` tokens; the shared
    prefix is the same leading text for every request. Without one, return the token ids.
    """
    if tokenizer is None:
        assert req.prompt_ids is not None, "trace has no prompt ids; pass a tokenizer"
        return req.prompt_ids
    prefix_words = [_WORDS[i % len(_WORDS)] for i in range(req.shared_prefix_len)]
    # Salt the body with the request index so prompts differ beyond the shared prefix.
    salt = int(req.request_id.rsplit("-", 1)[-1]) if req.request_id[-1].isdigit() else 0
    n_body = max(req.prompt_len - req.shared_prefix_len, 1)
    body_words = [_WORDS[(salt * 7 + i * 3) % len(_WORDS)] for i in range(n_body)]
    text = " ".join(prefix_words + body_words)
    ids = tokenizer.encode(text)
    while len(ids) < req.prompt_len:
        text += " " + " ".join(_WORDS[: req.prompt_len - len(ids)])
        ids = tokenizer.encode(text)
    if len(ids) > req.prompt_len:
        text = tokenizer.decode(ids[: req.prompt_len])
    return text


def trace_to_json(trace: list[TraceRequest], include_ids: bool = True) -> list[dict]:
    rows = [asdict(r) for r in trace]
    if not include_ids:
        for row in rows:
            row["prompt_ids"] = None
    return rows


def trace_from_json(rows: list[dict]) -> list[TraceRequest]:
    return [TraceRequest(**row) for row in rows]


def to_json(trace: list[TraceRequest], **kw) -> str:
    return json.dumps(trace_to_json(trace, **kw))


def from_json(s: str) -> list[TraceRequest]:
    return trace_from_json(json.loads(s))


def trace_summary(trace: list[TraceRequest]) -> dict[str, float]:
    p = np.array([r.prompt_len for r in trace])
    o = np.array([r.output_len for r in trace])
    return {"n": len(trace), "prompt_mean": float(p.mean()), "prompt_p50": float(np.median(p)),
            "output_mean": float(o.mean()), "output_p50": float(np.median(o)),
            "span_s": float(trace[-1].arrival_s - trace[0].arrival_s) if trace else 0.0}
