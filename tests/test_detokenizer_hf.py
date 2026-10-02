"""Incremental detokenizer vs full decode with the real Qwen tokenizer (skips without it).

The CPU suite covers the byte-level stub; this is the check that the sliding-window decode
agrees with `tokenizer.decode(all_ids)` across BPE merge boundaries, multi-byte characters
(CJK, emoji), and stop-string holdback.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"

pytestmark = pytest.mark.hf
if not MODEL.exists():
    pytest.skip("model files missing", allow_module_level=True)
pytest.importorskip("transformers")

from emberserve.tokenizer import IncrementalDetokenizer, Tokenizer  # noqa: E402

TEXTS = [
    "The quick brown fox jumps over the lazy dog. Numbers: 3.14159, 1,000,000; code: x[i]=y;",
    "日本語のテキストと絵文字 🚀🔥 を混ぜる。Ünïcödé façade naïve — em dash… “quotes”",
    "def f(x):\n    return x ** 2  # squared\n\n\tTabs\tand   spaces\n",
    "STOP here. But not before this sentence ends properly, thank you very much.",
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer(MODEL)


def stream(detok: IncrementalDetokenizer, ids: list[int], stop: list[str]) -> tuple[str, str | None]:
    out, matched = "", None
    for n in range(1, len(ids) + 1):
        delta, m = detok.update("r", ids[:n], stop, final=(n == len(ids)))
        out += delta
        if m is not None:
            matched = m
            break
    return out, matched


@pytest.mark.parametrize("text", TEXTS)
def test_incremental_equals_full_decode(tok: Tokenizer, text: str) -> None:
    ids = tok.encode(text)
    got, matched = stream(IncrementalDetokenizer(tok), ids, [])
    assert matched is None
    assert got == tok.decode(ids, skip_special_tokens=True)


def test_random_ids_never_desync(tok: Tokenizer) -> None:
    """Random token ids produce lots of partial UTF-8 sequences; the window must hold them
    back and still end up equal to the full decode (with the trailing U+FFFD dropped)."""
    rng = random.Random(0)
    vocab = tok.raw.vocab_size
    for _ in range(20):
        ids = [rng.randrange(vocab) for _ in range(rng.randrange(1, 80))]
        got, _ = stream(IncrementalDetokenizer(tok), ids, [])
        want = tok.decode(ids, skip_special_tokens=True)
        assert got == want.rstrip("�") or got == want


def test_stop_string_truncates_and_holds_prefix(tok: Tokenizer) -> None:
    ids = tok.encode("Count: one, two, three, END OF LIST, four, five")
    got, matched = stream(IncrementalDetokenizer(tok), ids, ["END OF LIST"])
    assert matched == "END OF LIST"
    assert got == "Count: one, two, three, "
    # A partial prefix of a stop string at the end of the emitted text is held back until
    # it either completes or is ruled out.
    detok = IncrementalDetokenizer(tok)
    ids = tok.encode("Count: one, END")
    out = "".join(detok.update("r", ids[:n], ["END OF LIST"])[0] for n in range(1, len(ids) + 1))
    assert out == "Count: one, "
    delta, m = detok.update("r", ids, ["END OF LIST"], final=True)
    assert m is None and out + delta == "Count: one, END"


def test_decode_batch_equals_decode(tok: Tokenizer) -> None:
    """The Rust batch path must be exactly what per-call `decode` returns (it is used only
    when the HF tokenizer applies no post-processing)."""
    rng = random.Random(1)
    vocab = tok.raw.vocab_size
    batch = [[rng.randrange(vocab) for _ in range(rng.randrange(1, 12))] for _ in range(50)]
    batch += [tok.encode(t)[:7] for t in TEXTS]
    for skip in (True, False):
        assert tok.decode_batch(batch, skip_special_tokens=skip) == \
            [tok.decode(ids, skip_special_tokens=skip) for ids in batch]


def test_update_batch_equals_update(tok: Tokenizer) -> None:
    ids_a, ids_b = tok.encode(TEXTS[0]), tok.encode(TEXTS[1])
    single = IncrementalDetokenizer(tok)
    batched = IncrementalDetokenizer(tok)
    out_s, out_b = ["", ""], ["", ""]
    for n in range(1, max(len(ids_a), len(ids_b)) + 1):
        a, b = ids_a[:n], ids_b[:n]
        out_s[0] += single.update("a", a, [])[0]
        out_s[1] += single.update("b", b, ["never"])[0]
        da, db = batched.update_batch(["a", "b"], [a, b], [[], ["never"]])
        out_b[0] += da[0]
        out_b[1] += db[0]
    assert out_b == out_s
