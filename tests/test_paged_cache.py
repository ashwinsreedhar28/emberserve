"""PagedKVCache: write/gather round trips, batched gather masks, dtype, sizing."""

import torch

from emberserve.attn.paged_torch import build_block_tables_tensor
from emberserve.config import ModelConfig
from emberserve.kv.block_manager import BlockManager
from emberserve.kv.cache import PagedKVCache

CFG = ModelConfig.tiny(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                       hidden_size=64)  # head_dim 16
BLOCK = 4


def _cache(dtype=torch.float32, num_blocks=8) -> PagedKVCache:
    return PagedKVCache(CFG, num_blocks=num_blocks, block_size=BLOCK, device="cpu", dtype=dtype)


def _write_seq(cache, bm, seq_id, n, layer, gen):
    """Allocate + write n random tokens for seq_id; returns (k, v) written."""
    bm.allocate(seq_id, n)
    k = torch.randn(n, CFG.num_key_value_heads, CFG.head_dim, generator=gen)
    v = torch.randn(n, CFG.num_key_value_heads, CFG.head_dim, generator=gen)
    slots = torch.tensor(bm.slot_mapping(seq_id, 0, n), dtype=torch.int64)
    cache.write(layer, k, v, slots)
    return k, v


def test_shapes_and_repr():
    cache = _cache()
    assert len(cache.k_cache) == CFG.num_hidden_layers
    assert cache.k_cache[0].shape == (8, BLOCK, 2, 16) and cache.v_cache[1].shape == (8, BLOCK, 2, 16)
    assert cache.memory_bytes() == 2 * 2 * 8 * BLOCK * 2 * 16 * 4
    assert "PagedKVCache" in repr(cache) and "num_blocks=8" in repr(cache)


def test_write_gather_round_trip_scrambled_table():
    gen = torch.Generator().manual_seed(0)
    cache, bm = _cache(), BlockManager(8, BLOCK)
    bm.allocate(100, 8)  # occupy [0, 1] then free so the test sequence gets a scrambled table
    bm.allocate(101, 4)  # [2]
    bm.free(100)
    k, v = _write_seq(cache, bm, 0, 10, layer=1, gen=gen)  # [3, 4, 5]
    table = bm.get_block_table(0)
    assert table == [3, 4, 5]
    gk, gv = cache.gather(1, table, 10)
    assert gk.shape == (10, 2, 16)
    torch.testing.assert_close(gk, k)
    torch.testing.assert_close(gv, v)
    # Layer 0 was never written for this sequence: still zeros.
    assert cache.gather(0, table, 10)[0].abs().sum() == 0
    # A -1-padded row of a batch table is accepted too.
    padded = build_block_tables_tensor([table, [2, 3, 4, 5, 6]], "cpu")
    torch.testing.assert_close(cache.gather(1, padded[0], 10)[0], k)


def test_partial_write_then_append():
    gen = torch.Generator().manual_seed(1)
    cache, bm = _cache(), BlockManager(8, BLOCK)
    k, v = _write_seq(cache, bm, 0, 5, layer=0, gen=gen)
    bm.append_slots(0, 1)
    k1 = torch.randn(1, 2, 16, generator=gen)
    v1 = torch.randn(1, 2, 16, generator=gen)
    cache.write(0, k1, v1, torch.tensor(bm.slot_mapping(0, 5, 1)))
    gk, gv = cache.gather(0, bm.get_block_table(0), 6)
    torch.testing.assert_close(gk, torch.cat([k, k1]))
    torch.testing.assert_close(gv, torch.cat([v, v1]))


def test_gather_batch_ragged_with_mask():
    gen = torch.Generator().manual_seed(2)
    cache, bm = _cache(num_blocks=12), BlockManager(12, BLOCK)
    lens = [3, 9, 6]
    written = [_write_seq(cache, bm, i, n, layer=0, gen=gen) for i, n in enumerate(lens)]
    tables = build_block_tables_tensor([bm.get_block_table(i) for i in range(3)], "cpu")
    assert tables.shape == (3, 3) and tables.dtype == torch.int32
    assert tables[0].tolist() == [0, -1, -1]

    k, v, valid = cache.gather_batch(0, tables, lens)
    assert k.shape == (3, 9, 2, 16) and v.shape == k.shape
    assert valid.dtype == torch.bool and valid.shape == (3, 9)
    assert valid.sum(dim=1).tolist() == lens
    for i, n in enumerate(lens):
        assert valid[i, :n].all() and not valid[i, n:].any()
        torch.testing.assert_close(k[i, :n], written[i][0])
        torch.testing.assert_close(v[i, :n], written[i][1])
    # Tensor context_lens works the same way.
    k2, _, valid2 = cache.gather_batch(0, tables, torch.tensor(lens))
    torch.testing.assert_close(k2, k)
    assert torch.equal(valid2, valid)


def test_fp16_storage_round_trip():
    gen = torch.Generator().manual_seed(3)
    cache, bm = _cache(dtype=torch.float16), BlockManager(8, BLOCK)
    assert cache.k_cache[0].dtype == torch.float16
    k, v = _write_seq(cache, bm, 0, 7, layer=1, gen=gen)  # fp32 in, fp16 stored
    gk, gv = cache.gather(1, bm.get_block_table(0), 7)
    assert gk.dtype == torch.float16
    torch.testing.assert_close(gk.float(), k, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(gv.float(), v, atol=1e-3, rtol=1e-3)


def test_reset_zeroes():
    gen = torch.Generator().manual_seed(4)
    cache, bm = _cache(), BlockManager(8, BLOCK)
    _write_seq(cache, bm, 0, 4, layer=0, gen=gen)
    assert cache.k_cache[0].abs().sum() > 0
    cache.reset()
    assert all(t.abs().sum() == 0 for t in cache.k_cache + cache.v_cache)


def test_num_blocks_for_bytes_real_config():
    cfg = ModelConfig()  # Qwen2.5-0.5B: 24 layers, 2 kv heads, head_dim 64
    assert cfg.head_dim == 64
    assert cfg.kv_bytes_per_token(torch.float16) == 2 * 24 * 2 * 64 * 2 == 12_288
    per_block = 12_288 * 16
    assert per_block == 196_608
    n = PagedKVCache.num_blocks_for_bytes(cfg, 16, 1 << 30, torch.float16)
    assert n == (1 << 30) // 196_608 == 5461
    assert PagedKVCache.num_blocks_for_bytes(cfg, 16, 196_607, torch.float16) == 0
    # fp32 doubles the per-token cost and halves the block count.
    assert PagedKVCache.num_blocks_for_bytes(cfg, 16, 1 << 30, torch.float32) == 2730
