"""Qwen2 model tests on the tiny config with the naive backend.

The shared metadata builders (`make_prefill_meta`, `make_decode_meta`, `tiny_model`) are
imported by the other model-side test files.
"""

from __future__ import annotations

import torch

from emberserve.attn.base import AttnMetadata
from emberserve.attn.naive import NaiveAttentionBackend
from emberserve.config import ModelConfig
from emberserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic

torch.set_num_threads(2)

SEED = 1234
ATOL = 1e-4


def tiny_model(seed: int = SEED, **overrides) -> Qwen2ForCausalLM:
    """A deterministic fp32 tiny model in eval mode."""
    model = Qwen2ForCausalLM(ModelConfig.tiny(**overrides))
    reset_parameters_deterministic(model, seed)
    return model.eval()


def make_prefill_meta(seq_ids: list[int], query_lens: list[int],
                      num_cached_tokens: list[int] | None = None) -> AttnMetadata:
    """Prefill metadata. Without cached tokens, positions per sequence are `arange(len)`;
    with a cached prefix they start at `num_cached_tokens[i]`."""
    cached = num_cached_tokens or [0] * len(seq_ids)
    positions = torch.cat([torch.arange(c, c + q, dtype=torch.int64)
                           for c, q in zip(cached, query_lens)])
    return AttnMetadata(
        is_prefill=True,
        seq_ids=list(seq_ids),
        query_lens=list(query_lens),
        context_lens=[c + q for c, q in zip(cached, query_lens)],
        positions=positions,
        num_cached_tokens=list(cached),
    )


def make_decode_meta(seq_ids: list[int], context_lens: list[int]) -> AttnMetadata:
    """Decode metadata: one query token per sequence at position `context_len - 1`."""
    return AttnMetadata(
        is_prefill=False,
        seq_ids=list(seq_ids),
        query_lens=[1] * len(seq_ids),
        context_lens=list(context_lens),
        positions=torch.tensor([c - 1 for c in context_lens], dtype=torch.int64),
    )


def _backend(model: Qwen2ForCausalLM) -> NaiveAttentionBackend:
    return NaiveAttentionBackend(model.config, device="cpu", dtype=torch.float32)


def _random_prompt(n: int, gen: torch.Generator, vocab: int = 256) -> torch.Tensor:
    return torch.randint(0, vocab, (n,), generator=gen)


@torch.no_grad()
def _prefill_all_logits(model: Qwen2ForCausalLM, backend: NaiveAttentionBackend,
                        seq_id: int, prompt: torch.Tensor) -> torch.Tensor:
    meta = make_prefill_meta([seq_id], [len(prompt)])
    return model.forward_logits_all(prompt, backend, meta)


@torch.no_grad()
def _decode_logits(model: Qwen2ForCausalLM, backend: NaiveAttentionBackend,
                   seq_ids: list[int], tokens: torch.Tensor,
                   context_lens: list[int]) -> torch.Tensor:
    meta = make_decode_meta(seq_ids, context_lens)
    return model.compute_logits(model(tokens, backend, meta), meta)


@torch.no_grad()
def generate_greedy(model: Qwen2ForCausalLM, backend_factory, prompt_ids: torch.Tensor,
                    n: int) -> list[int]:
    """Greedy-decode `n` tokens for one prompt on a fresh backend."""
    backend = backend_factory()
    meta = make_prefill_meta([0], [len(prompt_ids)])
    logits = model.compute_logits(model(prompt_ids, backend, meta), meta)
    out: list[int] = []
    context = len(prompt_ids)
    for _ in range(n):
        tok = int(logits[0].argmax())
        out.append(tok)
        context += 1
        logits = _decode_logits(model, backend, [0], torch.tensor([tok]), [context])
    return out


@torch.no_grad()
def _generate_greedy_batched(model: Qwen2ForCausalLM, backend: NaiveAttentionBackend,
                             prompts: list[torch.Tensor], n: int) -> list[list[int]]:
    """Greedy-decode `n` tokens for several prompts together (one packed batch per step)."""
    seq_ids = list(range(len(prompts)))
    lens = [len(p) for p in prompts]
    meta = make_prefill_meta(seq_ids, lens)
    logits = model.compute_logits(model(torch.cat(prompts), backend, meta), meta)
    outs: list[list[int]] = [[] for _ in prompts]
    contexts = list(lens)
    for _ in range(n):
        toks = logits.argmax(-1)
        for i, t in enumerate(toks.tolist()):
            outs[i].append(t)
        contexts = [c + 1 for c in contexts]
        logits = _decode_logits(model, backend, seq_ids, toks, contexts)
    return outs


def test_forward_shapes() -> None:
    model = tiny_model()
    backend = _backend(model)
    gen = torch.Generator().manual_seed(0)
    prompt = _random_prompt(9, gen)
    meta = make_prefill_meta([3], [9])
    hidden = model(prompt, backend, meta)
    assert hidden.shape == (9, model.config.hidden_size)
    assert model.compute_logits(hidden).shape == (9, model.config.vocab_size)
    assert model.compute_logits(hidden, meta).shape == (1, model.config.vocab_size)
    torch.testing.assert_close(model.compute_logits(hidden, meta)[0],
                               model.compute_logits(hidden)[-1])


def test_incremental_equals_full() -> None:
    model = tiny_model()
    gen = torch.Generator().manual_seed(0)
    prompt = _random_prompt(12, gen)

    full = _prefill_all_logits(model, _backend(model), 7, prompt)  # [12, vocab]

    backend = _backend(model)
    partial = _prefill_all_logits(model, backend, 7, prompt[:5])  # [5, vocab]
    torch.testing.assert_close(partial, full[:5], atol=ATOL, rtol=0)
    for t in range(5, 12):
        step = _decode_logits(model, backend, [7], prompt[t:t + 1], [t + 1])  # [1, vocab]
        torch.testing.assert_close(step[0], full[t], atol=ATOL, rtol=0)


def test_batched_prefill_matches_single() -> None:
    model = tiny_model()
    gen = torch.Generator().manual_seed(1)
    a, b = _random_prompt(7, gen), _random_prompt(11, gen)

    single_a = _prefill_all_logits(model, _backend(model), 0, a)
    single_b = _prefill_all_logits(model, _backend(model), 0, b)

    meta = make_prefill_meta([10, 11], [7, 11])
    with torch.no_grad():
        batched = model.forward_logits_all(torch.cat([a, b]), _backend(model), meta)
    torch.testing.assert_close(batched[:7], single_a, atol=ATOL, rtol=0)
    torch.testing.assert_close(batched[7:], single_b, atol=ATOL, rtol=0)


def test_batched_decode_matches_single() -> None:
    model = tiny_model()
    gen = torch.Generator().manual_seed(2)
    a, b = _random_prompt(7, gen), _random_prompt(11, gen)
    forced = [_random_prompt(2, gen) for _ in range(3)]  # one token per sequence per step

    # Reference: each sequence alone.
    singles: list[list[torch.Tensor]] = []
    for prompt, col in ((a, 0), (b, 1)):
        backend = _backend(model)
        _prefill_all_logits(model, backend, 0, prompt)
        steps = []
        for s, toks in enumerate(forced):
            steps.append(_decode_logits(model, backend, [0], toks[col:col + 1],
                                        [len(prompt) + s + 1])[0])
        singles.append(steps)

    # Batched: both sequences in every step.
    backend = _backend(model)
    meta = make_prefill_meta([5, 6], [7, 11])
    with torch.no_grad():
        model(torch.cat([a, b]), backend, meta)
    for s, toks in enumerate(forced):
        out = _decode_logits(model, backend, [5, 6], toks, [7 + s + 1, 11 + s + 1])
        torch.testing.assert_close(out[0], singles[0][s], atol=ATOL, rtol=0)
        torch.testing.assert_close(out[1], singles[1][s], atol=ATOL, rtol=0)


def test_greedy_generation_deterministic_and_batch_invariant() -> None:
    model = tiny_model()
    gen = torch.Generator().manual_seed(3)
    a, b = _random_prompt(6, gen), _random_prompt(9, gen)
    factory = lambda: _backend(model)  # noqa: E731

    alone_1 = generate_greedy(model, factory, a, 8)
    alone_2 = generate_greedy(model, factory, a, 8)
    assert alone_1 == alone_2, "greedy decoding must be deterministic"

    together = _generate_greedy_batched(model, factory(), [a, b], 8)
    assert together[0] == alone_1, "batching must not change sequence A's greedy output"
    assert together[1] == generate_greedy(model, factory, b, 8)


def test_tied_embeddings() -> None:
    tied = tiny_model(tie_word_embeddings=True)
    assert tied.lm_head.weight is tied.model.embed_tokens.weight
    assert "lm_head.weight" in tied.state_dict()

    untied = tiny_model(tie_word_embeddings=False)
    assert untied.lm_head.weight is not untied.model.embed_tokens.weight
    assert not torch.equal(untied.lm_head.weight, untied.model.embed_tokens.weight)


def test_deterministic_init_is_reproducible() -> None:
    m1, m2, m3 = tiny_model(seed=7), tiny_model(seed=7), tiny_model(seed=8)
    for (n1, p1), (_, p2), (_, p3) in zip(m1.named_parameters(), m2.named_parameters(),
                                          m3.named_parameters()):
        assert torch.equal(p1, p2), n1
        if "layernorm" not in n1 and not n1.endswith("norm.weight"):
            assert not torch.equal(p1, p3), n1
