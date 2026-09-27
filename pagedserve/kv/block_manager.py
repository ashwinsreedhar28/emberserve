"""BlockManager: allocates fixed-size KV cache blocks to sequences.

Pure Python, no tensors. This is the PagedAttention bookkeeping layer:
  * a free list of physical block ids
  * one block table (list of physical block ids) per sequence
  * slot mapping: logical token position -> physical slot = block_id * block_size + offset

Reference counts let prefix caching share blocks between sequences without changing the
interface: a block leaves a sequence's table only when its refcount drops to zero. With
`enable_prefix_caching`, every physical block is in exactly one of three states:
  * referenced  (refcount > 0)                     - in one or more block tables
  * free        (refcount 0, no hash)              - on the free list
  * evictable   (refcount 0, registered hash)      - content still valid, LRU-reclaimable
Free and evictable blocks are both allocatable; the free list is drained first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from pagedserve.kv.prefix_cache import PrefixCache, PrefixCacheStats


class OutOfBlocksError(RuntimeError):
    pass


@dataclass
class BlockManagerStats:
    num_blocks: int
    num_free: int  # allocatable: free list + evictable
    num_used: int
    used_token_slots: int  # tokens actually stored
    allocated_token_slots: int  # num_used * block_size
    num_evictable: int = 0
    prefix_cache: PrefixCacheStats | None = None

    @property
    def utilization(self) -> float:
        """Fraction of allocated slots holding a real token (1.0 = no internal fragmentation)."""
        return self.used_token_slots / self.allocated_token_slots if self.allocated_token_slots else 1.0


@dataclass(frozen=True)
class PrefixMatch:
    """Dry-run result of `BlockManager.match_prefix`: what an allocation would reuse.

    Valid only until the next mutation of the BlockManager.
    """

    hits: tuple[tuple[int, int], ...]  # (hash, block_id) per cached block, table order
    chain_hash: int | None  # hash of the prefix covered by `hits`
    num_cached_tokens: int  # len(hits) * block_size
    num_new_blocks: int  # fresh blocks the allocation still needs
    num_evictable_hits: int  # hits currently parked in the evictable LRU

    @property
    def num_hits(self) -> int:
        return len(self.hits)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int,
                 enable_prefix_caching: bool = False) -> None:
        assert num_blocks > 0 and block_size > 0
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))
        self._ref_count: list[int] = [0] * num_blocks
        self._tables: dict[int, list[int]] = {}
        self._num_tokens: dict[int, int] = {}  # tokens stored per sequence
        self._prefix: PrefixCache | None = None
        # Per sequence: (leading table blocks whose hash is known, hash of that prefix).
        self._registered: dict[int, tuple[int, int | None]] = {}
        if enable_prefix_caching:
            self.enable_prefix_caching()

    def enable_prefix_caching(self) -> None:
        """Turn on prefix caching (idempotent). Only valid before any allocation."""
        if self._prefix is None:
            assert not self._tables, "enable prefix caching before allocating"
            self._prefix = PrefixCache()

    @property
    def prefix_caching_enabled(self) -> bool:
        return self._prefix is not None

    # ---- queries -------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        """Allocatable blocks: the free list plus evictable cached blocks."""
        return len(self._free) + self.num_evictable_blocks

    @property
    def num_evictable_blocks(self) -> int:
        return self._prefix.num_evictable if self._prefix is not None else 0

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
            num_evictable=self.num_evictable_blocks,
            prefix_cache=self._prefix.stats() if self._prefix is not None else None,
        )

    # ---- mutation --------------------------------------------------------------
    def _pop_free(self) -> int:
        """Take a block from the free list, or evict the LRU cached block if it is empty."""
        if self._free:
            b = self._free.popleft()
        else:
            b = self._prefix.evict_one() if self._prefix is not None else None
            if b is None:
                raise OutOfBlocksError("no free KV blocks")
        assert self._ref_count[b] == 0
        self._ref_count[b] = 1
        return b

    def _release(self, block_id: int) -> None:
        assert self._ref_count[block_id] > 0
        self._ref_count[block_id] -= 1
        if self._ref_count[block_id] > 0:
            return
        if self._prefix is not None and self._prefix.hash_of(block_id) is not None:
            self._prefix.mark_evictable(block_id)  # keep the content, reclaim lazily
        else:
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
        if self._prefix is not None:
            self._registered[seq_id] = (0, None)
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
        self._registered.pop(seq_id, None)
        if table is None:
            return
        # With caching, release deepest-first so the tail of a chain is the LRU victim
        # and the longest usable prefix survives eviction (a chain is only useful up to
        # its first gap). Without caching the free-list order is unchanged.
        for b in (reversed(table) if self._prefix is not None else table):
            self._release(b)

    def reset(self) -> None:
        self._free = deque(range(self.num_blocks))
        self._ref_count = [0] * self.num_blocks
        self._tables.clear()
        self._num_tokens.clear()
        self._registered.clear()
        if self._prefix is not None:
            self._prefix.reset()

    # ---- slot mapping ------------------------------------------------------------
    def slot_mapping(self, seq_id: int, start_pos: int, num_tokens: int) -> list[int]:
        """Physical slots for logical positions [start_pos, start_pos + num_tokens)."""
        table = self._tables[seq_id]
        assert start_pos + num_tokens <= len(table) * self.block_size, "slots not allocated"
        out = []
        for pos in range(start_pos, start_pos + num_tokens):
            out.append(table[pos // self.block_size] * self.block_size + pos % self.block_size)
        return out

    # ---- sharing -------------------------------------------------------------------
    def ref_count(self, block_id: int) -> int:
        return self._ref_count[block_id]

    def share_block(self, block_id: int) -> None:
        """Increment refcount so a second sequence can reference a cached block. An
        evictable block (refcount 0, content valid) is resurrected; a free block is an error."""
        if self._ref_count[block_id] == 0:
            assert self._prefix is not None and self._prefix.is_evictable(block_id), \
                "cannot share a free block"
            self._prefix.unmark(block_id)
        self._ref_count[block_id] += 1

    # ---- prefix caching ---------------------------------------------------------------
    def _block_tokens(self, token_ids: list[int], i: int) -> tuple[int, ...]:
        return tuple(token_ids[i * self.block_size:(i + 1) * self.block_size])

    def match_prefix(self, token_ids: list[int]) -> PrefixMatch:
        """Dry run: walk the cache over the full blocks of `token_ids` until the first miss.

        At least one token is always left to compute (the model needs logits for the last
        prompt token), so for a prompt that is an exact multiple of block_size the final
        block is never taken from the cache even when it is present.
        """
        prefix = self._prefix
        assert prefix is not None, "prefix caching is disabled"
        n = len(token_ids)
        max_hits = (n - 1) // self.block_size
        hits: list[tuple[int, int]] = []
        prev: int | None = None
        evictable = 0
        for i in range(max_hits):
            h = prefix.block_hash(prev, self._block_tokens(token_ids, i))
            b = prefix.lookup(h)
            if b is None:
                break
            hits.append((h, b))
            evictable += prefix.is_evictable(b)
            prev = h
        return PrefixMatch(hits=tuple(hits), chain_hash=prev,
                           num_cached_tokens=len(hits) * self.block_size,
                           num_new_blocks=self.blocks_needed(n) - len(hits),
                           num_evictable_hits=evictable)

    def can_allocate_with_prefix(self, token_ids: list[int],
                                 match: PrefixMatch | None = None) -> bool:
        """Would `allocate_with_prefix(token_ids)` succeed right now? Hits parked in the
        evictable LRU are reused, not consumed, so they do not count against the budget."""
        m = match if match is not None else self.match_prefix(token_ids)
        return m.num_new_blocks <= self.num_free_blocks - m.num_evictable_hits

    def allocate_with_prefix(self, seq_id: int, token_ids: list[int],
                             match: PrefixMatch | None = None) -> tuple[list[int], int]:
        """Allocate a table for a new sequence, sharing cached blocks for its longest
        cached prefix. Returns `(block_table, num_cached_tokens)`; the caller computes only
        tokens `[num_cached_tokens:]`. `match` is an optional fresh `match_prefix` result.
        """
        assert seq_id not in self._tables, f"seq {seq_id} already allocated"
        prefix = self._prefix
        assert prefix is not None, "prefix caching is disabled"
        m = match if match is not None else self.match_prefix(token_ids)
        assert all(prefix.lookup(h) == b for h, b in m.hits), "stale PrefixMatch"
        if not self.can_allocate_with_prefix(token_ids, m):
            raise OutOfBlocksError(f"need {m.num_new_blocks} blocks, have "
                                   f"{self.num_free_blocks - m.num_evictable_hits}")
        table: list[int] = []
        for _, b in m.hits:
            self.share_block(b)
            table.append(b)
        table.extend(self._pop_free() for _ in range(m.num_new_blocks))
        self._tables[seq_id] = table
        self._num_tokens[seq_id] = len(token_ids)
        self._registered[seq_id] = (m.num_hits, m.chain_hash)
        prefix.record(hits=m.num_hits, misses=(len(token_ids) - 1) // self.block_size - m.num_hits)
        return table, m.num_cached_tokens

    def num_registered_blocks(self, seq_id: int) -> int:
        """Leading blocks of `seq_id` whose hash is already known (shared or registered)."""
        return self._registered[seq_id][0]

    def register_full_blocks(self, seq_id: int, token_ids: list[int]) -> int:
        """Publish the hash of every full block of `seq_id` not yet registered.

        `token_ids` must be exactly the tokens whose K/V have been WRITTEN for this
        sequence (a prefix of its history), so only those blocks become discoverable.
        If another block already owns a hash (two sequences computed the same prefix
        concurrently) this block stays unhashed and is freed normally later. O(new blocks).
        Returns the number of blocks newly walked.
        """
        prefix = self._prefix
        assert prefix is not None, "prefix caching is disabled"
        assert len(token_ids) <= self._num_tokens[seq_id], "tokens beyond the reserved slots"
        table = self._tables[seq_id]
        n_reg, prev = self._registered[seq_id]
        n_full = len(token_ids) // self.block_size
        for i in range(n_reg, n_full):
            h = prefix.block_hash(prev, self._block_tokens(token_ids, i))
            assert prefix.hash_of(table[i]) is None, "fresh block already hashed"
            prefix.insert(h, table[i])
            prev = h
        self._registered[seq_id] = (max(n_full, n_reg), prev)
        return max(n_full - n_reg, 0)
