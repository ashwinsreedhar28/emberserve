"""NaiveAttentionBackend against the reference `causal_softmax_attention`."""

from __future__ import annotations

import pytest
import torch

from emberserve.attn.base import causal_softmax_attention
from emberserve.attn.naive import NaiveAttentionBackend
from emberserve.config import ModelConfig
from tests.test_model import make_decode_meta, make_prefill_meta

torch.set_num_threads(2)

CFG = ModelConfig.tiny()  # 4 query heads, 2 kv heads, head_dim 16
H, HKV, D = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim


def _qkv(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.randn(n, H, D, generator=gen), torch.randn(n, HKV, D, generator=gen),
            torch.randn(n, HKV, D, generator=gen))


def _backend() -> NaiveAttentionBackend:
    return NaiveAttentionBackend(CFG, device="cpu", dtype=torch.float32)


def test_gqa_prefill_matches_reference() -> None:
    gen = torch.Generator().manual_seed(0)
    q, k, v = _qkv(9, gen)
    out = _backend().forward(0, q, k, v, make_prefill_meta([1], [9]))
    assert out.shape == (9, H, D)
    torch.testing.assert_close(out, causal_softmax_attention(q, k, v, 9))


def test_prefill_then_decode_matches_reference() -> None:
    gen = torch.Generator().manual_seed(1)
    q, k, v = _qkv(6, gen)
    backend = _backend()
    for layer in range(CFG.num_hidden_layers):
        backend.forward(layer, q[:4], k[:4], v[:4], make_prefill_meta([1], [4]))
    for t in range(4, 6):
        meta = make_decode_meta([1], [t + 1])
        for layer in range(CFG.num_hidden_layers):
            out = backend.forward(layer, q[t:t + 1], k[t:t + 1], v[t:t + 1], meta)
            expected = causal_softmax_attention(q[t:t + 1], k[:t + 1], v[:t + 1], 1)
            torch.testing.assert_close(out, expected)
        assert backend.num_cached_tokens(1, layer_idx=1) == t + 1


def test_packed_batch_isolates_sequences() -> None:
    gen = torch.Generator().manual_seed(2)
    qa, ka, va = _qkv(5, gen)
    qb, kb, vb = _qkv(3, gen)
    backend = _backend()
    meta = make_prefill_meta([10, 20], [5, 3])
    out = backend.forward(0, torch.cat([qa, qb]), torch.cat([ka, kb]), torch.cat([va, vb]), meta)
    torch.testing.assert_close(out[:5], causal_softmax_attention(qa, ka, va, 5))
    torch.testing.assert_close(out[5:], causal_softmax_attention(qb, kb, vb, 3))

    # One decode step for both; sequence order in the batch is arbitrary.
    qa2, ka2, va2 = _qkv(1, gen)
    qb2, kb2, vb2 = _qkv(1, gen)
    meta = make_decode_meta([20, 10], [4, 6])
    out = backend.forward(0, torch.cat([qb2, qa2]), torch.cat([kb2, ka2]),
                          torch.cat([vb2, va2]), meta)
    torch.testing.assert_close(
        out[:1], causal_softmax_attention(qb2, torch.cat([kb, kb2]), torch.cat([vb, vb2]), 1))
    torch.testing.assert_close(
        out[1:], causal_softmax_attention(qa2, torch.cat([ka, ka2]), torch.cat([va, va2]), 1))


def test_fresh_prefill_replaces_stale_state() -> None:
    """Re-prefilling a seq_id with num_cached_tokens == 0 starts from scratch (recompute)."""
    gen = torch.Generator().manual_seed(3)
    q, k, v = _qkv(4, gen)
    backend = _backend()
    backend.forward(0, q, k, v, make_prefill_meta([1], [4]))
    q2, k2, v2 = _qkv(3, gen)
    out = backend.forward(0, q2, k2, v2, make_prefill_meta([1], [3]))
    torch.testing.assert_close(out, causal_softmax_attention(q2, k2, v2, 3))
    assert backend.num_cached_tokens(1) == 3


def test_prefix_cached_prefill_appends() -> None:
    """A prefill with num_cached_tokens > 0 extends the existing entry."""
    gen = torch.Generator().manual_seed(4)
    q, k, v = _qkv(7, gen)
    backend = _backend()
    backend.forward(0, q[:3], k[:3], v[:3], make_prefill_meta([1], [3]))
    out = backend.forward(0, q[3:], k[3:], v[3:], make_prefill_meta([1], [4], [3]))
    torch.testing.assert_close(out, causal_softmax_attention(q[3:], k, v, 4))


def test_free_sequence_removes_state() -> None:
    gen = torch.Generator().manual_seed(5)
    q, k, v = _qkv(4, gen)
    backend = _backend()
    for layer in range(CFG.num_hidden_layers):
        backend.forward(layer, q, k, v, make_prefill_meta([1, 2], [2, 2]))
    backend.free_sequence(1)
    assert backend.num_cached_tokens(1, 0) == 0 and backend.num_cached_tokens(1, 1) == 0
    assert backend.num_cached_tokens(2, 0) == 2 and backend.num_cached_tokens(2, 1) == 2
    backend.free_sequence(999)  # unknown ids are a no-op
    backend.reset()
    assert backend.num_cached_tokens(2, 0) == 0


def test_context_len_mismatch_raises() -> None:
    gen = torch.Generator().manual_seed(6)
    q, k, v = _qkv(5, gen)
    backend = _backend()
    backend.forward(0, q[:4], k[:4], v[:4], make_prefill_meta([1], [4]))
    # Decode claims 7 tokens of context but the cache only holds 4 + 1.
    with pytest.raises(AssertionError, match="context_len"):
        backend.forward(0, q[4:], k[4:], v[4:], make_decode_meta([1], [7]))
