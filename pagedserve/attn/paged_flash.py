"""flash-attn PagedAttention backend (GPU only).

Same contract and same numbers as `paged_torch.PagedTorchAttentionBackend`; only the
attention kernel differs. This module imports without flash-attn / CUDA: the import
happens inside `is_available()` and the constructor.

flash-attn API assumptions (verified against the flash-attn 2.6.3, 2.7.4.post1 and
2.8.3.post1 sources, `flash_attn/flash_attn_interface.py` + `csrc/flash_attn/flash_api.cpp`):

* `flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None,
      cache_seqlens=<Tensor [B] int32>, block_table=<Tensor [B, max_blocks] int32>,
      softmax_scale=float, causal=True)`
    - q: [B, seqlen_q, H, D]; k_cache/v_cache: [num_blocks, page_block_size, Hkv, D]
      (exactly `PagedKVCache.k_cache[layer]`), returns [B, seqlen_q, H, D] in q.dtype.
    - `cache_seqlens[b]` = number of valid keys for row b. With `causal=True` the mask is
      aligned bottom-right: query row i attends keys `<= cache_seqlens[b] - seqlen_q + i`.
      We write this step's k/v ourselves first and pass `cache_seqlens = context_lens`
      (which already includes this step's tokens), so the queries are the LAST
      `seqlen_q` positions of the cache, which is the `base.py` semantic. A cached prefix
      (num_cached_tokens > 0) therefore needs no special handling.
    - **`page_block_size % 256 == 0` is required by every upstream release we checked
      (2.6.3 .. 2.8.3.post1 and `main`)**: `TORCH_CHECK(!paged_KV || page_block_size %
      256 == 0)`. Only vLLM's `vllm-flash-attn` fork relaxes this to 16. So the engine
      must run `--block-size 256` with this backend; the constructor asserts it.
    - fp16/bf16 only, head_dim % 8 == 0, head_dim <= 256, sm80+ (RTX 4090 / A40 ok).
    - Entries of `block_table` beyond `ceil(cache_seqlens[b] / page_block_size)` are never
      dereferenced (the kernel bounds its KV loop by cache_seqlens), but we still replace
      the -1 padding with block 0 so no negative index can ever reach the kernel.
* `flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
      max_seqlen_k, softmax_scale=float, causal=True)` with packed q/k/v [total, heads, D]
    - used for prefill steps with NO cached prefix: this step's packed k/v ARE the whole
      context, so no gather from the paged cache is needed (the write still happens so
      decode can read it back).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from pagedserve.attn.base import AttentionBackend, AttnMetadata
from pagedserve.config import ModelConfig
from pagedserve.kv.cache import PagedKVCache

# Upstream flash-attn paged-KV page size constraint (see module docstring).
FLASH_PAGE_MULTIPLE = 256
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def is_available() -> bool:
    """True when flash-attn imports and a CUDA device is present."""
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def _import_flash():
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    except ImportError as e:  # pragma: no cover - GPU only
        raise ImportError("PagedFlashAttentionBackend needs flash-attn>=2.6: "
                          "pip install flash-attn --no-build-isolation") from e
    return flash_attn, flash_attn_with_kvcache, flash_attn_varlen_func


def required_block_multiple(flash_attn_module=None) -> int:
    """Page-size multiple the installed flash-attn demands for paged KV.

    Every upstream `flash_attn` release checked (2.6.3, 2.7.x, 2.8.3.post1, main) requires
    256. If a future release relaxes it, lower the value here keyed on
    `flash_attn.__version__`; until then the gate is unconditional.
    """
    return FLASH_PAGE_MULTIPLE


def context_lens_tensor(meta: AttnMetadata, device: torch.device) -> Tensor:
    """`meta.context_lens_t` (device int32) — built once per step and cached on meta."""
    if meta.context_lens_t is None:
        meta.context_lens_t = torch.tensor(meta.context_lens, dtype=torch.int32, device=device)
    return meta.context_lens_t


def block_tables_nonneg(meta: AttnMetadata) -> Tensor:
    """`meta.block_tables` with -1 padding replaced by 0, int32, cached on meta."""
    if meta.block_tables_nonneg is None:
        assert meta.block_tables is not None
        meta.block_tables_nonneg = meta.block_tables.to(torch.int32).clamp_min(0).contiguous()
    return meta.block_tables_nonneg


class MixedPlan:
    """Per-step index plan for `_prefill_kvcache`: which packed rows are single-query decode
    rows (batched into one call) and which are multi-query prefill rows (padded among
    themselves), with each group's context lengths and block tables already gathered."""

    __slots__ = ("dec_tokens", "dec_ctx", "dec_bt", "pre_rows", "pre_max_q", "pre_ctx", "pre_bt")

    def __init__(self) -> None:
        self.dec_tokens: Tensor | None = None
        self.dec_ctx: Tensor | None = None
        self.dec_bt: Tensor | None = None
        self.pre_rows: list[tuple[int, int]] = []  # (packed start, query_len) per prefill seq
        self.pre_max_q = 0
        self.pre_ctx: Tensor | None = None
        self.pre_bt: Tensor | None = None

    @classmethod
    def build(cls, meta: AttnMetadata, device: torch.device) -> "MixedPlan":
        plan = cls()
        ctx = context_lens_tensor(meta, device)
        bt = block_tables_nonneg(meta)
        dec_tokens: list[int] = []
        dec_seqs: list[int] = []
        pre_seqs: list[int] = []
        start = 0
        for i, n in enumerate(meta.query_lens):
            if n == 1:
                dec_tokens.append(start)
                dec_seqs.append(i)
            else:
                pre_seqs.append(i)
                plan.pre_rows.append((start, n))
            start += n
        if dec_seqs:
            idx = torch.tensor(dec_seqs, dtype=torch.long, device=device)
            plan.dec_tokens = torch.tensor(dec_tokens, dtype=torch.long, device=device)
            plan.dec_ctx = ctx.index_select(0, idx)
            plan.dec_bt = bt.index_select(0, idx)
        if pre_seqs:
            idx = torch.tensor(pre_seqs, dtype=torch.long, device=device)
            plan.pre_max_q = max(n for _, n in plan.pre_rows)
            plan.pre_ctx = ctx.index_select(0, idx)
            plan.pre_bt = bt.index_select(0, idx)
        return plan


class PagedFlashAttentionBackend(AttentionBackend):
    """PagedAttention over a PagedKVCache computed by flash-attn kernels."""

    def __init__(self, config: ModelConfig, cache: PagedKVCache) -> None:
        assert cache.num_kv_heads == config.num_key_value_heads
        assert cache.head_dim == config.head_dim
        if cache.device.type != "cuda":
            raise RuntimeError(f"paged_flash needs a CUDA cache, got device {cache.device}")
        if cache.dtype not in SUPPORTED_DTYPES:
            raise RuntimeError(f"paged_flash needs an fp16/bf16 KV cache, got {cache.dtype}; "
                               "run the engine with --dtype float16")
        self._fa, self._kvcache_fn, self._varlen_fn = _import_flash()
        mult = required_block_multiple(self._fa)
        if cache.block_size % mult != 0:
            raise RuntimeError(
                f"flash-attn {getattr(self._fa, '__version__', '?')} requires the paged KV "
                f"block size to be a multiple of {mult}, got block_size={cache.block_size}. "
                "Run the engine with --block-size 256 for the paged_flash backend.")
        assert config.head_dim % 8 == 0 and config.head_dim <= 256, config.head_dim
        self.config = config
        self.cache = cache
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

    # ---- entry point -----------------------------------------------------------------
    def forward(self, layer_idx: int, q: Tensor, k: Tensor, v: Tensor,
                meta: AttnMetadata) -> Tensor:
        assert meta.slot_mapping is not None and meta.block_tables is not None \
            and meta.block_size is not None, "paged backend needs slot_mapping/block_tables/block_size"
        assert meta.block_size == self.cache.block_size, (meta.block_size, self.cache.block_size)
        assert q.shape == (meta.num_tokens, self.num_heads, self.head_dim), q.shape
        assert meta.block_tables.shape[0] == meta.num_seqs, meta.block_tables.shape
        if q.dtype not in SUPPORTED_DTYPES or q.dtype != self.cache.dtype:
            raise RuntimeError(f"paged_flash: q dtype {q.dtype} must equal the cache dtype "
                               f"{self.cache.dtype} (fp16/bf16)")
        self.cache.write(layer_idx, k, v, meta.slot_mapping)
        if not meta.is_prefill:
            return self._decode(layer_idx, q, meta)
        if not meta.num_cached_tokens or all(c == 0 for c in meta.num_cached_tokens):
            return self._prefill_varlen(q, k, v, meta)
        return self._prefill_kvcache(layer_idx, q, meta)

    # ---- decode: one query per sequence, paged KV ------------------------------------
    def _decode(self, layer_idx: int, q: Tensor, meta: AttnMetadata) -> Tensor:
        assert all(n == 1 for n in meta.query_lens), "decode step expects query_lens == 1"
        batch = meta.num_seqs
        out = self._kvcache_fn(
            q.view(batch, 1, self.num_heads, self.head_dim),
            self.cache.k_cache[layer_idx], self.cache.v_cache[layer_idx],
            k=None, v=None,
            cache_seqlens=context_lens_tensor(meta, self.cache.device),
            block_table=block_tables_nonneg(meta),
            softmax_scale=self.scale, causal=True,
        )
        return out.view(batch, self.num_heads, self.head_dim)

    # ---- prefill without cached prefix: packed varlen over this step's own k/v --------
    def _prefill_varlen(self, q: Tensor, k: Tensor, v: Tensor, meta: AttnMetadata) -> Tensor:
        max_q = max(meta.query_lens)
        cu = meta.cu_seqlens_q
        assert cu is not None and cu.dtype == torch.int32
        return self._varlen_fn(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max_q, max_seqlen_k=max_q,
            softmax_scale=self.scale, causal=True,
        )

    # ---- prefill with cached context: decode rows batched, prefill rows padded ---------
    def _prefill_kvcache(self, layer_idx: int, q: Tensor, meta: AttnMetadata) -> Tensor:
        """A step where some sequences attend through the cache with query_len > 1 (a
        chunked-prefill chunk or a cached-prefix prompt), usually alongside many decode rows.

        Rows with query_len == 1 go through one batched decode call. The few rows with
        query_len > 1 are LEFT-padded among themselves to [P, max_q, H, D] and attend with
        flash's bottom-right causal alignment: padded row r maps to key
        `context_lens[b] - max_q + r`, so real query j lands on `context_lens[b] -
        query_lens[b] + j`; the leading padding rows are discarded. Padding *every*
        sequence to max_q (the first version) allocated [B, max_q, H, D] per layer and
        computed attention for B x max_q queries: at 200 decodes plus one 512-token chunk
        that was 100k padded queries per layer, and the chunked-prefill ablation ran 2x
        slower than prefill-priority because of it.
        """
        plan = meta.mixed_plan
        if plan is None:
            plan = meta.mixed_plan = MixedPlan.build(meta, self.cache.device)
        out = q.new_empty(q.shape)
        if plan.dec_tokens is not None:
            nd = plan.dec_tokens.numel()
            o = self._kvcache_fn(
                q.index_select(0, plan.dec_tokens).view(nd, 1, self.num_heads, self.head_dim),
                self.cache.k_cache[layer_idx], self.cache.v_cache[layer_idx], k=None, v=None,
                cache_seqlens=plan.dec_ctx, block_table=plan.dec_bt,
                softmax_scale=self.scale, causal=True)
            out.index_copy_(0, plan.dec_tokens, o.view(nd, self.num_heads, self.head_dim))
        if plan.pre_rows:
            np_, max_q = len(plan.pre_rows), plan.pre_max_q
            qpad = q.new_zeros((np_, max_q, self.num_heads, self.head_dim))
            for i, (start, n) in enumerate(plan.pre_rows):
                qpad[i, max_q - n:] = q[start:start + n]
            o = self._kvcache_fn(
                qpad, self.cache.k_cache[layer_idx], self.cache.v_cache[layer_idx], k=None, v=None,
                cache_seqlens=plan.pre_ctx, block_table=plan.pre_bt,
                softmax_scale=self.scale, causal=True)
            for i, (start, n) in enumerate(plan.pre_rows):
                out[start:start + n] = o[i, max_q - n:]
        return out

    # ---- lifecycle ---------------------------------------------------------------------
    def free_sequence(self, seq_id: int) -> None:
        """No-op: the BlockManager owns block lifetime; slots are simply overwritten."""

    def reset(self) -> None:
        self.cache.reset()
