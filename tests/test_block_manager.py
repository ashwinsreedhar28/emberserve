"""BlockManager: allocation, slot mapping, refcounts, and exhaustion."""

import pytest

from pagedserve.kv.block_manager import BlockManager, OutOfBlocksError


def test_blocks_needed_ceil():
    bm = BlockManager(num_blocks=8, block_size=4)
    assert [bm.blocks_needed(n) for n in (0, 1, 4, 5, 8, 9)] == [0, 1, 1, 2, 2, 3]


def test_allocate_append_free_round_trip():
    bm = BlockManager(num_blocks=8, block_size=4)
    table = bm.allocate(seq_id=0, num_tokens=5)
    assert len(table) == 2 and bm.get_block_table(0) is table
    assert bm.get_num_tokens(0) == 5 and bm.num_free_blocks == 6
    assert bm.has_sequence(0) and not bm.has_sequence(1)

    # 5 -> 8 tokens fits in the second block; the 9th needs a third.
    assert bm.append_slots(0, 3) == []
    assert bm.get_num_tokens(0) == 8 and len(table) == 2
    new = bm.append_slots(0, 1)
    assert len(new) == 1 and table[-1] == new[0] and bm.num_free_blocks == 5

    bm.free(0)
    assert not bm.has_sequence(0) and bm.num_free_blocks == 8
    assert all(bm.ref_count(b) == 0 for b in range(8))
    bm.free(0)  # freeing twice is a no-op
    assert bm.num_free_blocks == 8


def test_slot_mapping_contiguous():
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.allocate(seq_id=7, num_tokens=6)  # blocks [0, 1]
    assert bm.slot_mapping(7, 0, 6) == [0, 1, 2, 3, 4, 5]
    assert bm.slot_mapping(7, 4, 2) == [4, 5]
    with pytest.raises(AssertionError):
        bm.slot_mapping(7, 6, 3)  # past the allocated 8 slots


def test_slot_mapping_scrambled_table():
    bm = BlockManager(num_blocks=6, block_size=4)
    bm.allocate(seq_id=0, num_tokens=8)  # A: blocks [0, 1]
    bm.allocate(seq_id=1, num_tokens=8)  # B: blocks [2, 3]
    bm.free(0)  # free list is now [4, 5, 0, 1]
    table = bm.allocate(seq_id=2, num_tokens=10)  # C takes [4, 5, 0]
    assert table == [4, 5, 0]
    expected = [4 * 4 + i for i in range(4)] + [5 * 4 + i for i in range(4)] + [0, 1]
    assert bm.slot_mapping(2, 0, 10) == expected
    assert bm.slot_mapping(2, 7, 3) == [23, 0, 1]
    # 10 -> 13 tokens spills into a fourth block, which takes the next free id, 1.
    assert bm.append_slots(2, 3) == [1]
    assert table == [4, 5, 0, 1]
    assert bm.slot_mapping(2, 12, 1) == [1 * 4 + 0]


def test_out_of_blocks_on_allocate_and_append():
    bm = BlockManager(num_blocks=3, block_size=2)
    assert bm.can_allocate(6) and not bm.can_allocate(7)
    with pytest.raises(OutOfBlocksError):
        bm.allocate(seq_id=0, num_tokens=7)
    assert bm.num_free_blocks == 3  # failed allocate leaves nothing behind

    bm.allocate(seq_id=0, num_tokens=2)
    bm.allocate(seq_id=1, num_tokens=4)
    assert bm.num_free_blocks == 0
    assert bm.can_append_slots(0, 0) and not bm.can_append_slots(0, 1)
    with pytest.raises(OutOfBlocksError):
        bm.append_slots(0, 1)
    assert bm.get_num_tokens(0) == 2 and len(bm.get_block_table(0)) == 1

    bm.free(1)
    assert bm.can_append_slots(0, 4) and not bm.can_append_slots(0, 5)
    assert len(bm.append_slots(0, 4)) == 2


def test_refcount_sharing():
    bm = BlockManager(num_blocks=4, block_size=4)
    table = bm.allocate(seq_id=0, num_tokens=4)
    shared = table[0]
    bm.share_block(shared)
    assert bm.ref_count(shared) == 2
    # A second sequence references the shared block plus one of its own.
    bm._tables[1] = [shared, bm._pop_free()]
    bm._num_tokens[1] = 6

    bm.free(0)
    assert bm.ref_count(shared) == 1
    assert shared not in bm._free and bm.num_free_blocks == 2
    bm.free(1)
    assert bm.ref_count(shared) == 0
    assert shared in bm._free and bm.num_free_blocks == 4
    with pytest.raises(AssertionError):
        bm.share_block(shared)  # cannot share a free block


def test_stats_and_utilization():
    bm = BlockManager(num_blocks=8, block_size=4)
    assert bm.stats().utilization == 1.0  # nothing allocated
    bm.allocate(seq_id=0, num_tokens=5)  # 2 blocks, 5 of 8 slots used
    bm.allocate(seq_id=1, num_tokens=4)  # 1 block, full
    s = bm.stats()
    assert (s.num_blocks, s.num_free, s.num_used) == (8, 5, 3)
    assert (s.used_token_slots, s.allocated_token_slots) == (9, 12)
    assert s.utilization == pytest.approx(9 / 12)


def test_reset():
    bm = BlockManager(num_blocks=4, block_size=2)
    bm.allocate(seq_id=0, num_tokens=4)
    bm.share_block(bm.get_block_table(0)[0])
    bm.reset()
    assert bm.num_free_blocks == 4 and not bm.has_sequence(0)
    assert all(bm.ref_count(b) == 0 for b in range(4))
    assert bm.allocate(seq_id=0, num_tokens=8) == [0, 1, 2, 3]
