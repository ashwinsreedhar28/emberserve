"""Token sampling and stop checking.

The Sampler consumes the logits of the LAST position of every sequence in a step:
`logits[i]` belongs to `requests[i]`. Stop STRINGS are checked by the engine on decoded
text; this module only knows token ids.
"""

from __future__ import annotations

import itertools

import torch

from pagedserve.sched.request import FinishReason, Request

_GENERATOR_KEY = "_generator"
# Requests without an explicit seed get a process-wide counter-based seed, offset so they
# do not collide with the small seeds users typically pass.
_UNSEEDED_BASE = 0x5EED_0000
_unseeded_counter = itertools.count()


def get_generator(request: Request, device: str | torch.device) -> torch.Generator:
    """Per-request RNG, created on first use and cached in `request.metadata`."""
    gen = request.metadata.get(_GENERATOR_KEY)
    if gen is None:
        seed = request.sampling_params.seed
        if seed is None:
            seed = _UNSEEDED_BASE + next(_unseeded_counter)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        request.metadata[_GENERATOR_KEY] = gen
    return gen


def apply_repetition_penalty(logits: torch.Tensor, token_ids: list[int],
                             penalty: float) -> torch.Tensor:
    """HF semantics over the SET of seen tokens: score / p if score > 0 else score * p.

    Out-of-place; `logits` is a 1-D [vocab] tensor.
    """
    if penalty == 1.0 or not token_ids:
        return logits
    seen = torch.tensor(sorted(set(token_ids)), dtype=torch.long, device=logits.device)
    scores = logits[seen]
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits.index_put((seen,), scores)


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    threshold = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter. A token is kept if the cumulative mass BEFORE it is < p, so the
    top token always survives."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs
    sorted_logits = sorted_logits.masked_fill(mass_before >= p, float("-inf"))
    return torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)


class Sampler:
    """Turns a [num_seqs, vocab] logits tensor into one token id per sequence."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, requests: list[Request]) -> list[int]:
        assert logits.dim() == 2 and logits.shape[0] == len(requests), \
            f"logits {tuple(logits.shape)} vs {len(requests)} requests"
        logits = logits.to(self.device)
        # TODO(perf): group rows by identical SamplingParams and run the filters batched;
        # a per-row loop is correct but costs O(num_seqs) small kernel launches per step.
        return [self._sample_row(logits[i], req) for i, req in enumerate(requests)]

    def _sample_row(self, logits: torch.Tensor, request: Request) -> int:
        params = request.sampling_params
        logits = logits.float()
        if params.repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, request.all_token_ids,
                                              params.repetition_penalty)
        if params.is_greedy:
            return int(torch.argmax(logits).item())
        logits = logits / params.temperature
        logits = _apply_top_k(logits, params.top_k)
        logits = _apply_top_p(logits, params.top_p)
        probs = torch.softmax(logits, dim=-1)
        gen = get_generator(request, self.device)
        return int(torch.multinomial(probs, 1, generator=gen).item())


def check_stop(request: Request, token_id: int, eos_token_id: int,
               max_model_len: int) -> FinishReason | None:
    """Decide whether `request` is done. Call AFTER `token_id` was appended to its output.

    STOP wins over LENGTH when both apply. Stop strings are the engine's job.
    """
    params = request.sampling_params
    if token_id == eos_token_id and not params.ignore_eos:
        return FinishReason.STOP
    if token_id in params.stop_token_ids:
        return FinishReason.STOP
    if request.num_output_tokens >= params.max_tokens or request.num_tokens >= max_model_len:
        return FinishReason.LENGTH
    return None
