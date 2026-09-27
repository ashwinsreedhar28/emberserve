"""Triton PagedAttention decode kernel + backend.

Why this exists: upstream flash-attn's paged decode (`flash_attn_with_kvcache`) demands
`page_block_size % 256 == 0`, so `paged_flash` forces `--block-size 256` and wastes up to
255 slots per sequence to internal fragmentation. This backend runs decode through a
hand-written Triton kernel that only needs `block_size % 16 == 0`, so `--block-size 16`
is legal on the GPU again and more sequences fit in the same KV memory. Prefill uses
flash-attn's packed *varlen* kernel, which has no block-size constraint: fresh prompts
attend over this step's own packed k/v, and chunk rows that attend through the cache
get their context gathered out of the paged cache first (`_prefill_mixed`, decode rows
of the same step still go through the Triton kernel). With a 256-multiple block size the
whole `paged_flash` backend is the prefill delegate; without flash-attn, `paged_torch`.

Kernel (`_paged_decode_kernel`, flash-decoding style, one pass, fp32 accumulation):

* grid `(B, Hkv, num_splits)`. A program owns one sequence, one KV head and ALL of that
  head's `groups = H // Hkv` query heads at once (`q_group [GROUPS_PAD, D]`), so every
  K/V element is read from HBM exactly once per KV head, not once per query head. That
  is the GQA win (Qwen2.5-0.5B: 14 query heads over 2 KV heads, so 7x less K/V traffic
  than a per-query-head kernel).
* the context is walked in tiles of `TILE` positions (`TILE` divides `BLOCK_SIZE`, so a
  tile never straddles two physical blocks); per tile the physical block id is read from
  the block table, `K/V [TILE, D]` are loaded with positions `>= context_len` masked,
  scores `[GROUPS_PAD, TILE]` are formed by broadcast-multiply + `tl.sum` over D (D=64/128
  is small enough that CUDA cores keep this memory-bound; `tl.dot` would need the group
  dim padded to 16), and the online softmax keeps a running max `m`, running sum `l` and
  accumulator `acc [GROUPS_PAD, D]`, rescaling by `exp(m_old - m_new)` per tile.
* split-K (flash-decoding): with `num_splits > 1` each program handles a contiguous
  range of tiles and writes its unnormalised `(m, l, acc)` to scratch; `_reduce_kernel`
  (grid `(B, H)`) merges the partials. This is what makes B=1 with a long context use
  more than `Hkv` SMs. `num_splits` is chosen from SHAPES only (batch, table width), never
  from tensor values, so a captured CUDA graph replays the same launch every time; the
  tile range of each split is computed INSIDE the kernel from the sequence's real context
  length (under graphs the table is padded to max_model_len, and partitioning that
  maximum on the host left every real tile to split 0: 43 us per layer at batch 1 for a
  320-token context on Moonlight).
* `GROUPS_PAD` is `groups` rounded up to a power of two (Qwen2.5-0.5B has 7 groups);
  padded query rows load zeros and are never stored.

The kernel is CUDA-graph safe: no host syncs, no `.item()`, every loop bound comes from
device scalars, and the split-K scratch buffers are cached per (num_splits, B) so a
replay reads the same addresses that were captured.

Importable without triton / CUDA: the import happens in `is_available()`, the backend
constructor and `paged_attention_decode`. Set `TRITON_INTERPRET=1` before importing
triton to run the kernel on the CPU through Triton's interpreter (that is how
tests/test_paged_triton.py runs here).
"""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor

from pagedserve.attn.base import AttentionBackend, AttnMetadata
from pagedserve.attn.paged_flash import block_tables_nonneg, context_lens_tensor
from pagedserve.attn.paged_torch import PagedTorchAttentionBackend
from pagedserve.config import ModelConfig
from pagedserve.kv.cache import PagedKVCache

BLOCK_MULTIPLE = 16
SUPPORTED_HEAD_DIMS = (64, 128)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_SPLITS = 16
SPLIT_CONTEXT = 1024  # split-K only pays off past ~1k keys per program (A100 A/B, results/kernels_a100_*.json)


def interpreter_enabled() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def is_available() -> bool:
    """True when triton imports and a CUDA device is present (or the interpreter is on)."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available() or interpreter_enabled()


# =====================================================================================
# kernels (defined lazily so this module imports without triton)
# =====================================================================================
_KERNELS: dict | None = None


def _kernels() -> dict:
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        raise ImportError("PagedTritonAttentionBackend needs triton: pip install triton") from e

    @triton.jit
    def _paged_decode_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, bt_ptr, ctx_ptr,
        m_part_ptr, l_part_ptr, acc_part_ptr,
        scale,
        stride_qb, stride_qh,
        stride_kb, stride_ks, stride_kh,
        stride_ob, stride_oh,
        stride_bt,
        GROUPS: tl.constexpr, GROUPS_PAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr, D: tl.constexpr,
        SPLIT_K: tl.constexpr, USE_DOT: tl.constexpr,
    ):
        b = tl.program_id(0)
        kvh = tl.program_id(1)
        split = tl.program_id(2)
        TILES_PER_BLOCK: tl.constexpr = BLOCK_SIZE // TILE

        ctx = tl.load(ctx_ptr + b)  # int32 scalar, number of valid keys
        g = tl.arange(0, GROUPS_PAD)
        d = tl.arange(0, D)
        s = tl.arange(0, TILE)
        g_valid = g < GROUPS
        heads = kvh * GROUPS + g  # query heads served by this KV head

        q_off = b * stride_qb + heads[:, None] * stride_qh + d[None, :]
        if USE_DOT:
            # tensor-core path: keep q in its storage dtype, scale the fp32 scores instead
            q = tl.load(q_ptr + q_off, mask=g_valid[:, None], other=0.0)  # [G, D]
        else:
            q = tl.load(q_ptr + q_off, mask=g_valid[:, None], other=0.0).to(tl.float32)  # [G, D]
            q = q * scale

        m_i = tl.full([GROUPS_PAD], float("-inf"), tl.float32)
        l_i = tl.zeros([GROUPS_PAD], tl.float32)
        acc = tl.zeros([GROUPS_PAD, D], tl.float32)

        # Partition the REAL context (a device value) across the splits: under CUDA graphs
        # the block table is padded to max_model_len, so a host-side partition of the
        # shape-derived maximum would hand every real tile to split 0.
        num_tiles = tl.cdiv(ctx, TILE)
        tiles_per_split = tl.cdiv(num_tiles, tl.num_programs(2))
        tile_start = split * tiles_per_split
        tile_end = tl.minimum(tile_start + tiles_per_split, num_tiles)

        kv_head_off = kvh * stride_kh
        for t in range(tile_start, tile_end):
            blk = t // TILES_PER_BLOCK
            phys = tl.load(bt_ptr + b * stride_bt + blk).to(tl.int64)
            pos = t * TILE + s  # logical positions of this tile
            in_blk = (t % TILES_PER_BLOCK) * TILE + s  # offsets inside the physical block
            valid = pos < ctx
            kv_off = phys * stride_kb + in_blk[:, None] * stride_ks + kv_head_off + d[None, :]
            if USE_DOT:
                k = tl.load(k_ptr + kv_off, mask=valid[:, None], other=0.0)  # [T, D] storage dtype
                v = tl.load(v_ptr + kv_off, mask=valid[:, None], other=0.0)  # [T, D]
                scores = tl.dot(q, tl.trans(k.to(q.dtype))) * scale  # [G, T], fp32 accumulate
            else:
                k = tl.load(k_ptr + kv_off, mask=valid[:, None], other=0.0).to(tl.float32)  # [T, D]
                v = tl.load(v_ptr + kv_off, mask=valid[:, None], other=0.0).to(tl.float32)  # [T, D]
                scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2)  # [G, T]
            scores = tl.where(valid[None, :], scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])  # [G, T]
            l_i = l_i * alpha + tl.sum(p, axis=1)
            if USE_DOT:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # [G, D]
            else:
                acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
            m_i = m_new

        if SPLIT_K:
            # partial layout: m/l [S, B, H], acc [S, B, H, D]  (H = Hkv * GROUPS)
            H = tl.num_programs(1) * GROUPS
            row = (split * tl.num_programs(0) + b) * H + heads  # [G]
            tl.store(m_part_ptr + row, m_i, mask=g_valid)
            tl.store(l_part_ptr + row, l_i, mask=g_valid)
            tl.store(acc_part_ptr + row[:, None] * D + d[None, :], acc, mask=g_valid[:, None])
        else:
            out = acc / l_i[:, None]
            o_off = b * stride_ob + heads[:, None] * stride_oh + d[None, :]
            tl.store(out_ptr + o_off, out.to(out_ptr.dtype.element_ty), mask=g_valid[:, None])

    @triton.jit
    def _reduce_kernel(
        m_part_ptr, l_part_ptr, acc_part_ptr, out_ptr,
        stride_ob, stride_oh,
        num_splits,
        NUM_SPLITS_PAD: tl.constexpr, D: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        B = tl.num_programs(0)
        H = tl.num_programs(1)
        sp = tl.arange(0, NUM_SPLITS_PAD)
        d = tl.arange(0, D)
        sp_valid = sp < num_splits
        row = (sp * B + b) * H + h  # [S]
        m = tl.load(m_part_ptr + row, mask=sp_valid, other=float("-inf"))
        lsum = tl.load(l_part_ptr + row, mask=sp_valid, other=0.0)
        acc = tl.load(acc_part_ptr + row[:, None] * D + d[None, :],
                      mask=sp_valid[:, None], other=0.0)  # [S, D]
        m_all = tl.max(m, axis=0)
        w = tl.exp(m - m_all)  # empty splits (m = -inf) get weight 0
        l_all = tl.sum(w * lsum, axis=0)
        out = tl.sum(acc * w[:, None], axis=0) / l_all
        tl.store(out_ptr + b * stride_ob + h * stride_oh + d, out.to(out_ptr.dtype.element_ty))

    _KERNELS = {"decode": _paged_decode_kernel, "reduce": _reduce_kernel}
    return _KERNELS


# =====================================================================================
# host-side launcher
# =====================================================================================
def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


TILE_BUDGET = 8192  # fp32 elements of the [GROUPS_PAD, TILE, D] broadcast temporary


def _tile_for(block_size: int, head_dim: int, groups_pad: int) -> int:
    """Positions per inner tile.

    The score/PV products materialise a `[GROUPS_PAD, TILE, D]` fp32 temporary that lives
    in registers, so TILE shrinks as the group count or head_dim grows: 16 for the real
    config (groups 7 -> 8, D=64), 32 for 4-or-fewer groups at D=64. Always a multiple of
    16 that divides `block_size` (a tile never straddles two physical blocks).
    """
    tile = max(16, min(32, TILE_BUDGET // (groups_pad * head_dim)))
    tile = 1 << (tile.bit_length() - 1)  # round down to a power of two
    return min(block_size, tile)


_SM_COUNT: dict[int, int] = {}


def num_sms(device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def default_num_splits(batch: int, num_kv_heads: int, max_context: int,
                       device: torch.device) -> int:
    """Split-K factor from SHAPES only (never tensor values), so a captured CUDA graph
    replays exactly what was captured.

    `PAGEDSERVE_TRITON_SPLITS=<n>` forces a value (1 disables split-K) for A/B runs.
    Otherwise: launch enough programs to keep every SM busy. `batch * num_kv_heads`
    programs already exist; if that is below ~2 per SM, split each sequence's context
    into `ceil(2 * SMs / programs)` pieces, capped by MAX_SPLITS and by how many tiles
    the (shape-derived) maximum context actually has. The earlier heuristic split by
    context length alone and paid the reduce launch even at B=128, where 256 programs
    already fill an RTX 4090 (128 SMs) - that was the 0.14 ms floor in results/kernels.json.
    """
    forced = os.environ.get("PAGEDSERVE_TRITON_SPLITS")
    if forced:
        return max(1, int(forced))
    programs = max(1, batch * num_kv_heads)
    target = 2 * num_sms(device)
    if programs >= target:
        return 1
    want = -(-target // programs)
    by_context = max(1, -(-max_context // SPLIT_CONTEXT))
    return max(1, min(MAX_SPLITS, want, by_context))


class _Scratch:
    """Split-K partial buffers, cached per (num_splits, B, H, D, device) for graph replay."""

    def __init__(self) -> None:
        self._bufs: dict[tuple, tuple[Tensor, Tensor, Tensor]] = {}

    def get(self, num_splits: int, batch: int, num_heads: int, head_dim: int,
            device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        key = (num_splits, batch, num_heads, head_dim, str(device))
        if key not in self._bufs:
            m = torch.empty((num_splits, batch, num_heads), dtype=torch.float32, device=device)
            lsum = torch.empty_like(m)
            acc = torch.empty((num_splits, batch, num_heads, head_dim), dtype=torch.float32,
                              device=device)
            self._bufs[key] = (m, lsum, acc)
        return self._bufs[key]


_SCRATCH = _Scratch()


VARIANTS = ("sum", "dot")


def default_variant() -> str:
    """`PAGEDSERVE_TRITON_VARIANT=sum|dot`; default `dot` (A100 A/B: sum was 3.3-10.8x behind
    flash-attn, dot is 1.2-2.4x; both validated against the fp32 golden and paged_torch).

    `sum`: broadcast-multiply + tl.sum on CUDA cores, GROUPS padded to a power of two,
           the [G, TILE, D] fp32 temporary limits TILE to 16-32.
    `dot`: tl.dot on tensor cores for QK^T and PV, GROUPS padded up to >= 16, K/V stay in
           storage dtype, larger TILE (up to 64). P is cast to the V dtype before PV (same
           as flash-attn), so fp16 caches lose ~1e-3 relative in P.
    """
    v = os.environ.get("PAGEDSERVE_TRITON_VARIANT", "dot").lower()
    if v not in VARIANTS:
        raise ValueError(f"PAGEDSERVE_TRITON_VARIANT must be one of {VARIANTS}, got {v!r}")
    return v


def paged_attention_decode(q: Tensor, k_cache: Tensor, v_cache: Tensor, block_tables: Tensor,
                           context_lens: Tensor, scale: float, *, num_splits: int | None = None,
                           out: Tensor | None = None, num_warps: int = 4,
                           variant: str | None = None) -> Tensor:
    """One-token-per-sequence paged attention.

    q [B, H, D]; k_cache/v_cache [num_blocks, block_size, Hkv, D]; block_tables
    [B, max_blocks] int32 (padding entries must be VALID ids: they are never read past
    `ceil(context_len / block_size)`, but must not be negative); context_lens [B] int32
    counting the token written this step. Returns [B, H, D] in q.dtype.
    """
    kern = _kernels()
    B, H, D = q.shape
    num_blocks, block_size, Hkv, Dk = k_cache.shape
    assert Dk == D and v_cache.shape == k_cache.shape, (q.shape, k_cache.shape, v_cache.shape)
    assert D in SUPPORTED_HEAD_DIMS, f"head_dim {D} not in {SUPPORTED_HEAD_DIMS}"
    assert block_size % BLOCK_MULTIPLE == 0, f"block_size {block_size} % {BLOCK_MULTIPLE} != 0"
    assert H % Hkv == 0, (H, Hkv)
    assert k_cache.dtype in SUPPORTED_DTYPES, k_cache.dtype
    assert block_tables.dtype == torch.int32 and block_tables.shape[0] == B, block_tables.shape
    assert context_lens.dtype == torch.int32 and context_lens.shape == (B,), context_lens.shape
    assert q.stride(2) == 1 and k_cache.stride(3) == 1 and v_cache.stride(3) == 1
    assert block_tables.stride(1) == 1 and context_lens.stride(0) == 1
    groups = H // Hkv
    variant = variant or default_variant()
    assert variant in VARIANTS, variant
    use_dot = variant == "dot"
    if use_dot:
        assert q.dtype == k_cache.dtype, f"dot variant needs q dtype == cache dtype ({q.dtype} vs {k_cache.dtype})"
        groups_pad = max(16, _next_pow2(groups))
        tile = min(block_size, 64)
    else:
        groups_pad = _next_pow2(groups)
        tile = _tile_for(block_size, D, groups_pad)
    max_blocks = block_tables.shape[1]
    max_context = max_blocks * block_size
    num_tiles_max = max(1, -(-max_context // tile))
    if num_splits is None:
        num_splits = default_num_splits(B, Hkv, max_context, q.device)
    num_splits = max(1, min(int(num_splits), num_tiles_max))

    if out is None:
        out = torch.empty_like(q)
    assert out.shape == q.shape and out.stride(2) == 1
    if num_splits == 1:
        m_p = l_p = acc_p = out  # unused placeholders (SPLIT_K=False branch never touches them)
    else:
        m_p, l_p, acc_p = _SCRATCH.get(num_splits, B, H, D, q.device)

    kern["decode"][(B, Hkv, num_splits)](
        q, k_cache, v_cache, out, block_tables, context_lens,
        m_p, l_p, acc_p,
        float(scale),
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        GROUPS=groups, GROUPS_PAD=groups_pad,
        BLOCK_SIZE=block_size, TILE=tile, D=D,
        SPLIT_K=num_splits > 1, USE_DOT=use_dot,
        num_warps=num_warps,
    )
    if num_splits > 1:
        kern["reduce"][(B, H)](
            m_p, l_p, acc_p, out,
            out.stride(0), out.stride(1),
            num_splits,
            NUM_SPLITS_PAD=_next_pow2(num_splits), D=D,
            num_warps=reduce_num_warps(_next_pow2(num_splits), D),
        )
    return out


def reduce_num_warps(num_splits_pad: int, d: int) -> int:
    """Warps for `_reduce_kernel`: its `[NUM_SPLITS_PAD, D]` fp32 tile lives in registers,
    so grow the warp count with the tile (1 warp up to 2k elements, 8 warps at 16k+)."""
    return max(1, min(8, (num_splits_pad * d) // 2048))


# =====================================================================================
# backend
# =====================================================================================
class PagedTritonAttentionBackend(AttentionBackend):
    """PagedAttention over a PagedKVCache: Triton decode kernel, delegated prefill."""

    def __init__(self, config: ModelConfig, cache: PagedKVCache,
                 num_splits: int | None = None, variant: str | None = None) -> None:
        assert cache.num_kv_heads == config.num_key_value_heads
        assert cache.head_dim == config.head_dim
        if cache.device.type != "cuda" and not interpreter_enabled():
            raise RuntimeError(f"paged_triton needs a CUDA cache (or TRITON_INTERPRET=1), "
                               f"got device {cache.device}")
        if cache.device.type == "cuda" and interpreter_enabled():
            # The interpreter runs the kernel on the CPU with host copies: silently 100x
            # slower, and illegal inside a CUDA-graph capture. Refuse rather than "work".
            raise RuntimeError("TRITON_INTERPRET=1 is set but the KV cache is on CUDA; unset it "
                               "(the interpreter is for CPU-only test runs)")
        if cache.dtype not in SUPPORTED_DTYPES:
            raise RuntimeError(f"paged_triton needs an fp16/bf16/fp32 KV cache, got {cache.dtype}")
        if cache.block_size % BLOCK_MULTIPLE != 0:
            raise RuntimeError(f"paged_triton needs block_size % {BLOCK_MULTIPLE} == 0, "
                               f"got block_size={cache.block_size}")
        if config.head_dim not in SUPPORTED_HEAD_DIMS:
            raise RuntimeError(f"paged_triton supports head_dim in {SUPPORTED_HEAD_DIMS}, "
                               f"got {config.head_dim}")
        _kernels()  # fail early if triton is missing
        self.config = config
        self.cache = cache
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.num_splits = num_splits  # None -> shape heuristic
        self.variant = variant or default_variant()
        self._varlen_fn = None  # flash_attn_varlen_func when flash-attn can run here
        self._prefill = self._make_prefill_delegate()

    def _make_prefill_delegate(self) -> AttentionBackend:
        """Prefill goes to flash-attn when it can run here, else the torch reference.

        With a block size flash's paged path accepts (multiple of 256) the delegate is the
        full `paged_flash` backend sharing OUR cache; only its private `_prefill*` helpers
        are called (after our single `cache.write`), so K/V is written once per step.
        With the block sizes this backend exists for (16, 32, ...) flash's paged kernel is
        off the table, but its packed *varlen* kernel is not: fresh prompts attend over
        this step's own packed k/v, and rows that attend through the cache (chunked-prefill
        chunks, cached prefixes) get their context gathered out of the paged cache into a
        packed buffer first (`_prefill_mixed`). Only without flash-attn at all does prefill
        fall back to the torch reference.
        """
        from pagedserve.attn import paged_flash
        flash_ok = (paged_flash.is_available() and self.cache.device.type == "cuda"
                    and self.cache.dtype in paged_flash.SUPPORTED_DTYPES)
        if flash_ok:
            self._varlen_fn = paged_flash._import_flash()[2]
            if self.cache.block_size % paged_flash.required_block_multiple() == 0:
                return paged_flash.PagedFlashAttentionBackend(self.config, self.cache)
        return PagedTorchAttentionBackend(self.config, self.cache)

    @property
    def prefill_backend_name(self) -> str:
        if self._varlen_fn is not None and isinstance(self._prefill, PagedTorchAttentionBackend):
            return "flash_varlen+triton_decode"
        return type(self._prefill).__name__

    # ---- entry point -----------------------------------------------------------------
    def forward(self, layer_idx: int, q: Tensor, k: Tensor, v: Tensor,
                meta: AttnMetadata) -> Tensor:
        assert meta.slot_mapping is not None and meta.block_tables is not None \
            and meta.block_size is not None, "paged backend needs slot_mapping/block_tables/block_size"
        assert meta.block_size == self.cache.block_size, (meta.block_size, self.cache.block_size)
        assert q.shape == (meta.num_tokens, self.num_heads, self.head_dim), q.shape
        assert meta.block_tables.shape[0] == meta.num_seqs, meta.block_tables.shape
        self.cache.write(layer_idx, k, v, meta.slot_mapping)
        if not meta.is_prefill:
            return self._decode(layer_idx, q, meta)
        return self._delegate_prefill(layer_idx, q, k, v, meta)

    # ---- decode: the Triton kernel ---------------------------------------------------
    def _decode(self, layer_idx: int, q: Tensor, meta: AttnMetadata) -> Tensor:
        assert all(n == 1 for n in meta.query_lens), "decode step expects query_lens == 1"
        return paged_attention_decode(
            q.contiguous(), self.cache.k_cache[layer_idx], self.cache.v_cache[layer_idx],
            block_tables_nonneg(meta), context_lens_tensor(meta, self.cache.device),
            self.scale, num_splits=self.num_splits, variant=self.variant,
        )

    # ---- prefill: delegate's helpers, K/V already written ----------------------------
    def _delegate_prefill(self, layer_idx: int, q: Tensor, k: Tensor, v: Tensor,
                          meta: AttnMetadata) -> Tensor:
        d = self._prefill
        if self._varlen_fn is None:
            return d._prefill(layer_idx, q, meta)
        if q.dtype != self.cache.dtype:
            raise RuntimeError(f"paged_triton/flash prefill: q dtype {q.dtype} must equal the "
                               f"cache dtype {self.cache.dtype}")
        fresh = not meta.num_cached_tokens or all(c == 0 for c in meta.num_cached_tokens)
        if isinstance(d, PagedTorchAttentionBackend):  # block size flash's paged path rejects
            if fresh:
                return self._prefill_varlen(q, k, v, meta)
            return self._prefill_mixed(layer_idx, q, meta)
        if fresh:
            return d._prefill_varlen(q, k, v, meta)
        return d._prefill_kvcache(layer_idx, q, meta)

    def _prefill_varlen(self, q: Tensor, k: Tensor, v: Tensor, meta: AttnMetadata) -> Tensor:
        """Fresh prompts: this step's packed k/v are the whole context, so flash's varlen
        kernel runs straight on them (no block-size constraint, nothing gathered)."""
        max_q = max(meta.query_lens)
        cu = meta.cu_seqlens_q
        return self._varlen_fn(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max_q,
                               max_seqlen_k=max_q, softmax_scale=self.scale, causal=True)

    def _prefill_mixed(self, layer_idx: int, q: Tensor, meta: AttnMetadata) -> Tensor:
        """Rows attending through the cache with query_len > 1 (a chunked-prefill chunk, a
        cached-prefix prompt), usually beside many decode rows.

        Decode rows (query_len == 1) go through the Triton decode kernel in one batched
        launch, exactly as a pure decode step would. Each chunk row's FULL context is
        gathered out of the paged cache into a packed [sum ctx, Hkv, D] buffer (cheap: the
        chunk is compute-bound anyway and there are only a few such rows per step) and the
        chunk queries attend over it with flash varlen, whose causal mask is bottom-right
        aligned for seqlen_q < seqlen_k: query j of a chunk lands on key
        `context_len - query_len + j`, the packed-layout contract.
        """
        from pagedserve.attn.paged_flash import MixedPlan
        plan = meta.mixed_plan
        if plan is None:
            plan = meta.mixed_plan = MixedPlan.build(meta, self.cache.device)
        out = q.new_empty(q.shape)
        if plan.dec_tokens is not None:
            o = paged_attention_decode(
                q.index_select(0, plan.dec_tokens), self.cache.k_cache[layer_idx],
                self.cache.v_cache[layer_idx], plan.dec_bt, plan.dec_ctx, self.scale,
                num_splits=self.num_splits, variant=self.variant)
            out.index_copy_(0, plan.dec_tokens, o)
        if plan.pre_rows:
            plan.packed_prefill(meta, self.cache.device)
            k_store, v_store = self.cache.k_cache[layer_idx], self.cache.v_cache[layer_idx]
            ks, vs = [], []
            for r, i in enumerate(plan.pre_seqs):  # no host sync: pre_bt is already non-negative
                ctx = meta.context_lens[i]
                idx = plan.pre_bt[r, :self.cache.blocks_needed(ctx)].long()
                ks.append(k_store[idx].view(-1, self.num_kv_heads, self.head_dim)[:ctx])
                vs.append(v_store[idx].view(-1, self.num_kv_heads, self.head_dim)[:ctx])
            k_ctx = ks[0] if len(ks) == 1 else torch.cat(ks, dim=0)
            v_ctx = vs[0] if len(vs) == 1 else torch.cat(vs, dim=0)
            o = self._varlen_fn(
                q.index_select(0, plan.pre_tokens), k_ctx, v_ctx,
                cu_seqlens_q=plan.pre_cu_q, cu_seqlens_k=plan.pre_cu_k,
                max_seqlen_q=plan.pre_max_q, max_seqlen_k=plan.pre_max_k,
                softmax_scale=self.scale, causal=True)
            out.index_copy_(0, plan.pre_tokens, o)
        return out

    # ---- lifecycle ---------------------------------------------------------------------
    def free_sequence(self, seq_id: int) -> None:
        """No-op: the BlockManager owns block lifetime; slots are simply overwritten."""

    def reset(self) -> None:
        self.cache.reset()
