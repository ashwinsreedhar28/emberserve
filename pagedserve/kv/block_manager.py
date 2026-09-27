"""BlockManager: allocates fixed-size KV cache blocks to sequences.

Pure Python, no tensors. This is the PagedAttention bookkeeping layer:
  * a free list of physical block ids
  * one block table (list of physical block ids) per sequence
  * slot mapping: logical token position -> physical slot = block_id * block_size + offset

Reference counts exist from day one so prefix caching (Thu) can share blocks between
sequences without changing the interface: a block is returned to the free list only
when its refcount drops to zero.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


class OutOfBlocksError(RuntimeError):
    pass


@dataclass
class BlockManagerStats:
    num_blocks: int
    num_free: int
    num_used: int
    used_token_slots: int  # tokens actually stored
    allocated_token_slots: int  # num_used * block_size

    @property
    def utilization(self) -> float:
        """Fraction of allocated slots holding a real token (1.0 = no internal fragmentation)."""
        return self.used_token_slots / self.allocated_token_slots if self.allocated_token_slots else 1.0


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        assert num_blocks > 0 and block_size > 0
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))
        self._ref_count: list[int] = [0] * num_blocks
        self._tables: dict[int, list[int]] = {}
        self._num_tokens: dict[int, int] = {}  # tokens stored per sequence

    # ---- queries -------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self._tables

    def get_block_table(self, seq_id: int) -> list[int]:
        return self._tables[seq_id]

    def get_num_tokens(self, seq_id: int) -> int:
        return self._num_tokens[seq_id]

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= self.num_free_blocks

    def can_append_slots(self, seq_id: int, num_new_tokens: int = 1) -> bool:
        """Can `num_new_tokens` more tokens be stored for seq_id (allocating blocks as needed)?"""
        need_blocks = self.blocks_needed(self._num_tokens[seq_id] + num_new_tokens) - len(self._tables[seq_id])
        return need_blocks <= 0 or need_blocks <= self.num_free_blocks

    def stats(self) -> BlockManagerStats:
        used = self.num_blocks - self.num_free_blocks
        return BlockManagerStats(
            num_blocks=self.num_blocks,
            num_free=self.num_free_blocks,
            num_used=used,
            used_token_slots=sum(self._num_tokens.values()),
            allocated_token_slots=sum(len(t) for t in self._tables.values()) * self.block_size,
        )

    # ---- mutation --------------------------------------------------------------
    def _pop_free(self) -> int:
        if not self._free:
            raise OutOfBlocksError("no free KV blocks")
        b = self._free.popleft()
        assert self._ref_count[b] == 0
        self._ref_count[b] = 1
        return b

    def _release(self, block_id: int) -> None:
        assert self._ref_count[block_id] > 0
        self._ref_count[block_id] -= 1
        if self._ref_count[block_id] == 0:
            self._free.append(block_id)

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Reserve blocks for a new sequence with `num_tokens` tokens about to be written
        (the prompt). Returns the block table. Raises OutOfBlocksError if it can't."""
        assert seq_id not in self._tables, f"seq {seq_id} already allocated"
        n = self.blocks_needed(num_tokens)
        if n > self.num_free_blocks:
            raise OutOfBlocksError(f"need {n} blocks, have {self.num_free_blocks}")
        table = [self._pop_free() for _ in range(n)]
        self._tables[seq_id] = table
        self._num_tokens[seq_id] = num_tokens
        return table

    def append_slots(self, seq_id: int, num_new_tokens: int = 1) -> list[int]:
        """Make room for `num_new_tokens` more tokens (decode: 1). Allocates a new block
        when the last block is full. Returns ids of any newly allocated blocks."""
        table = self._tables[seq_id]
        cur = self._num_tokens[seq_id]
        need = self.blocks_needed(cur + num_new_tokens) - len(table)
        if need > self.num_free_blocks:
            raise OutOfBlocksError(f"need {need} blocks, have {self.num_free_blocks}")
        new = [self._pop_free() for _ in range(max(need, 0))]
        table.extend(new)
        self._num_tokens[seq_id] = cur + num_new_tokens
        return new

    def free(self, seq_id: int) -> None:
        """Release every block of a sequence (finished, aborted, or preempted)."""
        table = self._tables.pop(seq_id, None)
        self._num_tokens.pop(seq_id, None)
        if table is None:
            return
        for b in table:
            self._release(b)

    def reset(self) -> None:
        self._free = deque(range(self.num_blocks))
        self._ref_count = [0] * self.num_blocks
        self._tables.clear()
        self._num_tokens.clear()

    # ---- slot mapping ------------------------------------------------------------
    def slot_mapping(self, seq_id: int, start_pos: int, num_tokens: int) -> list[int]:
        """Physical slots for logical positions [start_pos, start_pos + num_tokens)."""
        table = self._tables[seq_id]
        assert start_pos + num_tokens <= len(table) * self.block_size, "slots not allocated"
        out = []
        for pos in range(start_pos, start_pos + num_tokens):
            out.append(table[pos // self.block_size] * self.block_size + pos % self.block_size)
        return out

    # ---- prefix caching hooks (used from Thu; safe no-ops before) -----------------
    def ref_count(self, block_id: int) -> int:
        return self._ref_count[block_id]

    def share_block(self, block_id: int) -> None:
        """Increment refcount so a second sequence can reference a cached block."""
        assert self._ref_count[block_id] > 0, "cannot share a free block"
        self._ref_count[block_id] += 1
