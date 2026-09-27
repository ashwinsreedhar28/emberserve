"""Token sampling and stop checking.

The Sampler consumes the logits of the LAST position of every sequence in a step:
`logits[i]` belongs to `requests[i]`. Stop STRINGS are checked by the engine on decoded
text; this module only knows token ids.

The batch is sampled with ONE device->host transfer per step. The first version looped
over rows and called `.item()` on each, i.e. one CUDA sync per running sequence per step;
on an A100 that alone made TPOT grow linearly with concurrency (4.5 ms at batch ~1 to
32 ms at batch ~200) while vLLM's stayed flat.
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


def _filter_rows(logits: torch.Tensor, top_k: list[int], top_p: list[float]) -> torch.Tensor:
    """Batched top-k then top-p over [n, vocab] with per-row k and p, matching the per-row
    `_apply_top_k` / `_apply_top_p` semantics exactly (value threshold for k, so ties at the
    k-th value survive; nucleus keeps every token whose preceding mass is < p)."""
    n, vocab = logits.shape
    need_k = any(0 < k < vocab for k in top_k)
    need_p = any(p < 1.0 for p in top_p)
    if not need_k and not need_p:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    if need_k:
        k_idx = torch.tensor([min(k, vocab) - 1 if k > 0 else vocab - 1 for k in top_k],
                             device=logits.device).unsqueeze(1)
        threshold = sorted_logits.gather(1, k_idx)  # k-th largest per row
        sorted_logits = sorted_logits.masked_fill(sorted_logits < threshold, float("-inf"))
    if need_p:
        probs = torch.softmax(sorted_logits, dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs
        p_col = torch.tensor([p if p < 1.0 else 2.0 for p in top_p],
                             device=logits.device).unsqueeze(1)
        sorted_logits = sorted_logits.masked_fill(mass_before >= p_col, float("-inf"))
    return torch.empty_like(logits).scatter_(1, sorted_idx, sorted_logits)


class Sampler:
    """Turns a [num_seqs, vocab] logits tensor into one token id per sequence."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, requests: list[Request]) -> list[int]:
        assert logits.dim() == 2 and logits.shape[0] == len(requests), \
            f"logits {tuple(logits.shape)} vs {len(requests)} requests"
        logits = logits.to(self.device).float()
        params = [r.sampling_params for r in requests]
        # Repetition penalty is per row by construction (each row's own token set); it is
        # off by default and stays a launch per penalized row, still with no sync.
        for i, (req, p) in enumerate(zip(requests, params, strict=True)):
            if p.repetition_penalty != 1.0:
                logits[i] = apply_repetition_penalty(logits[i], req.all_token_ids,
                                                     p.repetition_penalty)
        if all(p.is_greedy for p in params):
            return torch.argmax(logits, dim=-1).tolist()  # the whole step: one transfer

        n = len(requests)
        tokens = torch.empty(n, dtype=torch.long, device=logits.device)
        greedy = [i for i, p in enumerate(params) if p.is_greedy]
        rand = [i for i, p in enumerate(params) if not p.is_greedy]
        if greedy:
            g_idx = torch.tensor(greedy, device=logits.device)
            tokens[g_idx] = torch.argmax(logits[g_idx], dim=-1)
        r_idx = torch.tensor(rand, device=logits.device)
        sub = logits[r_idx]
        temps = torch.tensor([params[i].temperature for i in rand],
                             device=logits.device).unsqueeze(1)
        sub = _filter_rows(sub / temps, [params[i].top_k for i in rand],
                           [params[i].top_p for i in rand])
        probs = torch.softmax(sub, dim=-1)
        # Per-request generators keep seeds reproducible; multinomial is a launch per row
        # but the results stay on device until the single `.tolist()` below.
        draws = [torch.multinomial(probs[j], 1, generator=get_generator(requests[i], self.device))
                 for j, i in enumerate(rand)]
        tokens[r_idx] = torch.cat(draws)
        return tokens.tolist()


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
