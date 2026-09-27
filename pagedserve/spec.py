"""Speculative decoding by n-gram lookup (prompt lookup decoding, Saxena 2023).

A decode step normally computes one token per running request. With speculation the
engine *guesses* the next `k` tokens for a request, runs them through the model together
with the real last token as a `k + 1`-query step over the cached context (the same
attention path as a chunked-prefill chunk), and keeps the longest prefix of guesses the
model's own greedy choices agree with, plus the model's token after it. Every accepted
guess is a decode step that did not have to happen; a rejected one costs the extra query
rows of one step, which at small batch is nearly free because the step is weight-read
bound. The output is *exactly* what plain greedy decoding produces: verification consults
the same logits at every position.

The proposer here is the cheapest one: the last `n` tokens of the sequence are looked up
earlier in the sequence and the tokens that followed that earlier occurrence are the
guess (code, quoted text, repeated names, lists). No draft model, no extra weights.
Only greedy requests are drafted; a sampled request decodes one token per step.
"""

from __future__ import annotations


def propose_ngram(tokens: list[int], max_ngram: int, k: int, min_ngram: int = 1) -> list[int]:
    """Up to `k` draft tokens for the sequence `tokens` (prompt + generated so far).

    Tries the longest suffix n-gram first (`max_ngram` down to `min_ngram`), searching
    for its most recent earlier occurrence; the tokens that followed it are the draft,
    cut at the end of the sequence. Empty when nothing matches."""
    if k <= 0 or not tokens:
        return []
    total = len(tokens)
    for n in range(min(max_ngram, total - 1), min_ngram - 1, -1):
        key = tokens[total - n:]
        # earlier occurrences only: the window may not include the suffix itself
        for start in range(total - n - 1, -1, -1):
            if tokens[start:start + n] == key:
                # an earlier occurrence is always followed by at least the suffix's start
                return tokens[start + n:start + n + k]
    return []


def accepted_prefix(drafts: list[int], sampled: list[int]) -> int:
    """How many leading drafts the model agreed with: `sampled[j]` is the model's token at
    the position whose input was draft `j - 1` (or the real last token for j = 0), so draft
    `j` is right iff `sampled[j] == drafts[j]`."""
    n = 0
    for d, s in zip(drafts, sampled):
        if d != s:
            break
        n += 1
    return n
