"""Sampler filters, per-request RNG, and check_stop."""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from emberserve.sampling import Sampler, apply_repetition_penalty, check_stop, get_generator
from emberserve.sched.request import FinishReason, Request, SamplingParams

VOCAB = 32


def make_request(params: SamplingParams, prompt: list[int] | None = None,
                 output: list[int] | None = None, rid: str = "r") -> Request:
    req = Request(request_id=rid, prompt_token_ids=prompt or [3, 4, 5],
                  sampling_params=params, seq_id=0)
    req.output_token_ids = list(output or [])
    return req


def random_logits(rows: int = 1, vocab: int = VOCAB, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, vocab, generator=g)


def draw(sampler: Sampler, logits: torch.Tensor, req: Request, n: int) -> list[int]:
    return [sampler.sample(logits, [req])[0] for _ in range(n)]


# ---- batched filters == per-row reference ---------------------------------------


def _reference_filter(row: torch.Tensor, k: int, p: float) -> torch.Tensor:
    from emberserve.sampling import _apply_top_k, _apply_top_p

    return _apply_top_p(_apply_top_k(row, k), p)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_filter_matches_per_row_reference(seed):
    from emberserve.sampling import _filter_rows

    logits = random_logits(6, seed=seed)
    logits[0, :4] = logits[0, 0]  # ties at the k-th value must survive top-k
    ks = [-1, 1, 3, 3, VOCAB, 5]
    ps = [1.0, 1.0, 0.5, 0.9, 0.3, 1.0]
    got = _filter_rows(logits.clone(), ks, ps)
    for i in range(6):
        want = _reference_filter(logits[i].clone(), ks[i], ps[i])
        assert torch.equal(got[i], want), (i, ks[i], ps[i])


def test_batched_step_matches_row_by_row_sampling():
    """One sample() call over a mixed batch == sampling each row alone with the same seeds."""
    logits = random_logits(5, seed=7)
    specs = [SamplingParams(temperature=0.0), SamplingParams(temperature=0.8, seed=1),
             SamplingParams(temperature=1.0, top_k=4, seed=2),
             SamplingParams(temperature=1.2, top_p=0.7, seed=3),
             SamplingParams(temperature=0.5, top_k=6, top_p=0.9, repetition_penalty=1.3, seed=4)]
    batched = Sampler("cpu").sample(logits, [make_request(sp, rid=f"b{i}") for i, sp in enumerate(specs)])
    single = [Sampler("cpu").sample(logits[i:i + 1], [make_request(sp, rid=f"s{i}")])[0]
              for i, sp in enumerate(specs)]
    assert batched == single


# ---- greedy / temperature / top-k / top-p ---------------------------------------

def test_greedy_is_argmax():
    logits = random_logits(rows=4)
    reqs = [make_request(SamplingParams.greedy(), rid=f"r{i}") for i in range(4)]
    assert Sampler("cpu").sample(logits, reqs) == logits.argmax(dim=-1).tolist()


def test_greedy_ignores_seed_and_is_stable():
    logits = random_logits()
    req = make_request(SamplingParams.greedy(seed=7))
    assert set(draw(Sampler("cpu"), logits, req, 20)) == {int(logits.argmax())}


def test_low_temperature_sharpens_but_keeps_mode():
    logits = random_logits(seed=3)
    argmax = int(logits.argmax())
    hot = draw(Sampler("cpu"), logits, make_request(SamplingParams(temperature=1.0, seed=1)), 200)
    cold = draw(Sampler("cpu"), logits, make_request(SamplingParams(temperature=0.5, seed=1)), 200)
    assert Counter(cold).most_common(1)[0][0] == argmax
    # Sharpening: the mode takes a strictly larger share of the draws.
    assert Counter(cold)[argmax] > Counter(hot)[argmax]


def test_top_k_one_is_argmax():
    logits = random_logits(seed=5)
    req = make_request(SamplingParams(temperature=1.0, top_k=1, seed=11))
    assert set(draw(Sampler("cpu"), logits, req, 50)) == {int(logits.argmax())}


def test_tiny_top_p_is_argmax():
    logits = random_logits(seed=6)
    req = make_request(SamplingParams(temperature=1.0, top_p=0.01, seed=11))
    assert set(draw(Sampler("cpu"), logits, req, 50)) == {int(logits.argmax())}


def test_top_k_three_stays_in_top_three():
    # Flat logits so unfiltered sampling would spread over the whole vocab.
    logits = random_logits(seed=8) * 0.1
    top3 = set(torch.topk(logits[0], 3).indices.tolist())
    req = make_request(SamplingParams(temperature=1.0, top_k=3, seed=2))
    seen = set(draw(Sampler("cpu"), logits, req, 500))
    assert seen <= top3
    assert len(seen) == 3, "500 draws over 3 near-uniform tokens should hit all of them"


def test_top_p_keeps_the_nucleus_only():
    logits = torch.full((1, VOCAB), -10.0)
    logits[0, [4, 9, 17]] = torch.tensor([3.0, 2.0, 1.0])  # ~0.66, 0.24, 0.09
    req = make_request(SamplingParams(temperature=1.0, top_p=0.7, seed=4))
    # mass before token 9 is 0.66 < 0.7 (kept); before 17 it is 0.90 (dropped).
    assert set(draw(Sampler("cpu"), logits, req, 300)) == {4, 9}


# ---- repetition penalty ---------------------------------------------------------

def test_repetition_penalty_positive_logit():
    logits = torch.zeros(VOCAB)
    logits[5], logits[6] = 2.0, 1.5
    out = apply_repetition_penalty(logits, [5, 5, 1], 2.0)
    assert out[5].item() == pytest.approx(1.0)  # divided
    assert out[6].item() == pytest.approx(1.5)  # untouched
    assert out[1].item() == pytest.approx(0.0)
    assert logits[5].item() == 2.0, "must be out of place"


def test_repetition_penalty_negative_logit():
    logits = torch.full((VOCAB,), -3.0)
    logits[5], logits[6] = -0.5, -0.8
    out = apply_repetition_penalty(logits, [5], 2.0)
    assert out[5].item() == pytest.approx(-1.0)  # multiplied, pushed further down
    assert out[6].item() == pytest.approx(-0.8)


def test_repetition_penalty_flips_greedy_choice():
    logits = torch.zeros(1, VOCAB)
    logits[0, 5], logits[0, 6] = 2.0, 1.5
    penalised = make_request(SamplingParams.greedy(repetition_penalty=2.0), output=[5])
    plain = make_request(SamplingParams.greedy(), output=[5])
    s = Sampler("cpu")
    assert s.sample(logits, [plain]) == [5]
    assert s.sample(logits, [penalised]) == [6]


# ---- RNG ------------------------------------------------------------------------

def test_same_seed_reproduces_and_different_seeds_diverge():
    logits = random_logits(seed=9)
    s = Sampler("cpu")
    a = draw(s, logits, make_request(SamplingParams(seed=123), rid="a"), 50)
    b = draw(s, logits, make_request(SamplingParams(seed=123), rid="b"), 50)
    c = draw(s, logits, make_request(SamplingParams(seed=124), rid="c"), 50)
    assert a == b
    assert a != c


def test_unseeded_requests_get_distinct_generators():
    r1, r2 = make_request(SamplingParams(), rid="a"), make_request(SamplingParams(), rid="b")
    g1, g2 = get_generator(r1, "cpu"), get_generator(r2, "cpu")
    assert g1 is not g2 and g1.initial_seed() != g2.initial_seed()
    assert get_generator(r1, "cpu") is g1, "generator is cached on the request"


def test_mixed_batch_rows_are_independent():
    logits = random_logits(rows=3, seed=10)
    s = Sampler("cpu")
    greedy = make_request(SamplingParams.greedy(), rid="g")
    sampled = make_request(SamplingParams(seed=77), rid="s")
    penal = make_request(SamplingParams.greedy(repetition_penalty=1.5), rid="p")
    batched = [s.sample(logits, [greedy, sampled, penal]) for _ in range(20)]
    # Row 1 of the batch matches sampling that row on its own with the same seed.
    alone = draw(Sampler("cpu"), logits[1:2], make_request(SamplingParams(seed=77), rid="s2"), 20)
    assert [b[1] for b in batched] == alone
    assert all(b[0] == int(logits[0].argmax()) for b in batched)
    assert all(b[2] == s.sample(logits[2:3], [penal])[0] for b in batched)


# ---- check_stop -----------------------------------------------------------------

EOS = 1


def stop_after(params: SamplingParams, output: list[int], max_model_len: int = 100):
    req = make_request(params, prompt=[3, 4, 5], output=output)
    return check_stop(req, output[-1], EOS, max_model_len)


@pytest.mark.parametrize("params,output,max_model_len,expected", [
    (SamplingParams(max_tokens=8), [9, EOS], 100, FinishReason.STOP),
    (SamplingParams(max_tokens=8, ignore_eos=True), [9, EOS], 100, None),
    (SamplingParams(max_tokens=8, stop_token_ids=[42]), [9, 42], 100, FinishReason.STOP),
    (SamplingParams(max_tokens=8, stop_token_ids=[42]), [9, 43], 100, None),
    (SamplingParams(max_tokens=2), [9, 10], 100, FinishReason.LENGTH),
    (SamplingParams(max_tokens=3), [9, 10], 100, None),
    # 3 prompt + 2 output = 5 tokens hits max_model_len.
    (SamplingParams(max_tokens=8), [9, 10], 5, FinishReason.LENGTH),
    (SamplingParams(max_tokens=8), [9, 10], 6, None),
    # STOP takes precedence over LENGTH when both apply.
    (SamplingParams(max_tokens=2), [9, EOS], 100, FinishReason.STOP),
    (SamplingParams(max_tokens=2, ignore_eos=True), [9, EOS], 100, FinishReason.LENGTH),
])
def test_check_stop_matrix(params, output, max_model_len, expected):
    assert stop_after(params, output, max_model_len) == expected
