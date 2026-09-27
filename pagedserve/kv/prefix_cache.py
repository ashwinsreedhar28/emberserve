"""PrefixCache: content-addressed index over KV blocks for automatic prefix caching.

A full block's identity is a chain hash of its own token ids and the hash of the block
before it, so equal hashes mean equal token prefixes from position 0 (vLLM-style). The
BlockManager keeps a block's K/V alive after its last owner leaves by parking it here as
EVICTABLE (refcount 0, content still valid); such blocks are reclaimed least-recently-used
only once the free list is empty. Pure Python, no tensors.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class PrefixCacheStats:
    hits: int  # eligible prompt blocks served from the cache
    misses: int  # eligible prompt blocks that had to be computed
    evictions: int
    num_cached_blocks: int  # blocks currently registered under a hash (shared or evictable)
    num_evictable: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class PrefixCache:
    """hash -> block id (and back) plus an LRU of evictable blocks."""

    def __init__(self) -> None:
        self._block_of: dict[int, int] = {}  # hash -> block_id
        self._hash_of: dict[int, int] = {}  # block_id -> hash
        self._evictable: OrderedDict[int, None] = OrderedDict()  # LRU: oldest first
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ---- hashing -------------------------------------------------------------
    @staticmethod
    def block_hash(prev_hash: int | None, token_ids: tuple[int, ...]) -> int:
        """Chain hash of one full block: 64-bit blake2b over (prev_hash, token_ids).

        `prev_hash` is None for the first block of a sequence, so a block's hash covers
        every token from position 0 through the end of the block.
        """
        h = hashlib.blake2b(digest_size=8)
        h.update(struct.pack("<?Q", prev_hash is not None, prev_hash or 0))
        h.update(struct.pack(f"<{len(token_ids)}q", *token_ids))
        return int.from_bytes(h.digest(), "little")

    # ---- hash <-> block ------------------------------------------------------
    def lookup(self, h: int) -> int | None:
        """Block holding the prefix with hash `h`, or None. Does not touch the stats."""
        return self._block_of.get(h)

    def insert(self, h: int, block_id: int) -> bool:
        """Register `block_id` under `h`. Returns False (and changes nothing) if another
        block already owns the hash: the first writer of a prefix wins."""
        owner = self._block_of.get(h)
        if owner is not None:
            return owner == block_id
        assert block_id not in self._hash_of, "block already registered under another hash"
        self._block_of[h] = block_id
        self._hash_of[block_id] = h
        return True

    def remove(self, block_id: int) -> None:
        """Forget `block_id` entirely (its hash and any evictable entry)."""
        h = self._hash_of.pop(block_id, None)
        if h is not None:
            del self._block_of[h]
        self._evictable.pop(block_id, None)

    def hash_of(self, block_id: int) -> int | None:
        return self._hash_of.get(block_id)

    @property
    def num_cached_blocks(self) -> int:
        return len(self._hash_of)

    # ---- evictable LRU --------------------------------------------------------
    def mark_evictable(self, block_id: int) -> None:
        """Park a registered block whose refcount just dropped to 0 (most recently used)."""
        assert block_id in self._hash_of, "only hashed blocks are evictable"
        assert block_id not in self._evictable, "already evictable"
        self._evictable[block_id] = None

    def unmark(self, block_id: int) -> None:
        """A lookup hit resurrected the block: it is referenced again."""
        assert block_id in self._evictable, "block is not evictable"
        del self._evictable[block_id]

    def is_evictable(self, block_id: int) -> bool:
        return block_id in self._evictable

    @property
    def num_evictable(self) -> int:
        return len(self._evictable)

    def evict_one(self) -> int | None:
        """Drop the least recently used evictable block (and its hash) and return its id."""
        if not self._evictable:
            return None
        block_id, _ = self._evictable.popitem(last=False)
        self.remove(block_id)
        self.evictions += 1
        return block_id

    # ---- stats ------------------------------------------------------------------
    def record(self, hits: int, misses: int) -> None:
        """Account one prefix match: `hits` blocks reused, `misses` eligible blocks not."""
        self.hits += hits
        self.misses += misses

    @property
    def hit_rate(self) -> float:
        return self.stats().hit_rate

    def stats(self) -> PrefixCacheStats:
        return PrefixCacheStats(hits=self.hits, misses=self.misses, evictions=self.evictions,
                                num_cached_blocks=self.num_cached_blocks,
                                num_evictable=self.num_evictable)

    def reset(self) -> None:
        self._block_of.clear()
        self._hash_of.clear()
        self._evictable.clear()
        self.hits = self.misses = self.evictions = 0
