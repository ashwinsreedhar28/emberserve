"""CUDA-graph replay for the decode step (GPU only).

A decode step at batch B is the same kernel sequence every time: only the contents of
`input_ids`, `positions`, `slot_mapping`, `block_tables` and `context_lens` change. So we
capture model forward + `compute_logits` once per batch-size BUCKET into a
`torch.cuda.CUDAGraph` and replay it after copying the step's inputs into static buffers.
That removes the per-layer Python/launch overhead that dominates small-batch decode.

Requirements this places on the rest of the engine:
* backend = `paged_flash`: every tensor the kernels read must be a STATIC buffer, so the
  backend reads `cache_seqlens` from `meta.context_lens_t` and the block table from
  `meta.block_tables_nonneg` (both filled in by this runner), never from Python lists.
* block tables have a fixed width `max_blocks = ceil(max_model_len / block_size)`; rows
  are padded with block 0. flash-attn never dereferences table entries beyond
  `ceil(cache_seqlens / block_size)`, so 0 is safe (any valid id is).
* one RESERVED SCRATCH BLOCK: rows of a bucket beyond the real batch still execute the
  KV write (index_copy_ into the cache), so their slot_mapping points into a block the
  BlockManager never hands out. The engine allocates the cache with `num_blocks` and the
  BlockManager with `num_blocks - 1`; the last block id is the scratch block.
* prefill stays eager (`LLMEngine.step` only calls `run` when `not is_prefill`).

Importable without CUDA; nothing touches the device until `capture()`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from emberserve.attn.base import AttnMetadata
from emberserve.devutil import index_tensor

DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


class CUDAGraphRunner:
    def __init__(self, model, backend, max_model_len: int, block_size: int, scratch_block: int,
                 buckets: tuple[int, ...] = DEFAULT_BUCKETS, max_batch: int | None = None,
                 device: torch.device | str = "cuda") -> None:
        self.model = model
        self.backend = backend
        self.device = torch.device(device)
        self.block_size = block_size
        self.max_blocks = math.ceil(max_model_len / block_size)
        self.scratch_block = scratch_block
        cache = backend.cache
        assert 0 <= scratch_block < cache.num_blocks, (scratch_block, cache.num_blocks)
        bs = sorted(set(b for b in buckets if max_batch is None or b <= max_batch))
        if max_batch is not None and (not bs or bs[-1] < max_batch):
            bs.append(max_batch)
        assert bs, "no buckets"
        self.buckets: list[int] = bs
        self.max_bucket = bs[-1]
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._metas: dict[int, AttnMetadata] = {}
        self._pool = None
        self._captured = False
        self._alloc_static()

    # ---- static buffers ------------------------------------------------------------
    def _alloc_static(self) -> None:
        n, dev = self.max_bucket, self.device
        self.input_ids = torch.zeros(n, dtype=torch.int64, device=dev)
        self.positions = torch.zeros(n, dtype=torch.int64, device=dev)
        self.slot_mapping = torch.zeros(n, dtype=torch.int64, device=dev)
        self.block_tables = torch.zeros((n, self.max_blocks), dtype=torch.int32, device=dev)
        self.context_lens = torch.ones(n, dtype=torch.int32, device=dev)
        self.cu_seqlens_q = torch.arange(n + 1, dtype=torch.int32, device=dev)
        vocab = self.model.config.vocab_size
        dtype = next(self.model.parameters()).dtype
        self.logits = torch.zeros((n, vocab), dtype=dtype, device=dev)
        self._fill_padding(0)

    def _scratch_slots(self, start: int, end: int) -> Tensor:
        """Distinct slots inside the scratch block for rows [start, end)."""
        rows = torch.arange(start, end, device=self.device)
        return self.scratch_block * self.block_size + rows % self.block_size

    def _fill_padding(self, start: int) -> None:
        """Rows [start, max_bucket) become harmless dummies."""
        n = self.max_bucket
        if start >= n:
            return
        self.input_ids[start:].zero_()
        self.positions[start:].zero_()
        self.slot_mapping[start:] = self._scratch_slots(start, n)
        self.block_tables[start:].zero_()
        self.block_tables[start:, 0] = self.scratch_block
        self.context_lens[start:] = 1

    def _meta(self, bucket: int) -> AttnMetadata:
        """Metadata whose tensors are VIEWS of the static buffers (kept for replay)."""
        if bucket not in self._metas:
            self._metas[bucket] = AttnMetadata(
                is_prefill=False, seq_ids=list(range(bucket)), query_lens=[1] * bucket,
                context_lens=[1] * bucket, positions=self.positions[:bucket],
                slot_mapping=self.slot_mapping[:bucket],
                block_tables=self.block_tables[:bucket], block_size=self.block_size,
                cu_seqlens_q=self.cu_seqlens_q[:bucket + 1],
                context_lens_t=self.context_lens[:bucket],
                block_tables_nonneg=self.block_tables[:bucket],
            )
        return self._metas[bucket]

    def _forward(self, bucket: int) -> Tensor:
        meta = self._meta(bucket)
        hidden = self.model(self.input_ids[:bucket], self.backend, meta)
        return self.model.compute_logits(hidden, meta)

    # ---- capture --------------------------------------------------------------------
    @torch.inference_mode()
    def capture(self, warmup_iters: int = 2) -> None:
        """Warm up then capture one graph per bucket, largest first, sharing a mempool."""
        assert self.device.type == "cuda", "CUDA graphs need a CUDA device"
        self._fill_padding(0)
        self._pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for b in reversed(self.buckets):
                for _ in range(warmup_iters):
                    self._forward(b)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        for b in reversed(self.buckets):
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g, pool=self._pool):
                    out = self._forward(b)
                    self.logits[:b].copy_(out)
            except Exception as capture_err:  # noqa: BLE001 - re-raised with diagnostics
                raise RuntimeError(self._diagnose(b, capture_err)) from capture_err
            self.graphs[b] = g
        torch.cuda.synchronize(self.device)
        # Capture ran real KV writes into scratch; wipe anything it may have touched.
        self.backend.reset()
        self._captured = True

    def _diagnose(self, bucket: int, capture_err: Exception) -> str:
        """Capture errors are opaque ("previous error during capture"); re-run the same
        bucket eagerly with a sync after every layer to name the kernel that fails."""
        lines = [f"CUDA graph capture failed for bucket {bucket} "
                 f"(backend {type(self.backend).__name__}): {capture_err}"]
        try:
            torch.cuda.synchronize(self.device)
        except Exception as e:  # noqa: BLE001
            lines.append(f"  device already in error state: {e}")
            return "\n".join(lines)
        meta = self._meta(bucket)
        try:
            hidden = self.model.model.embed_tokens(self.input_ids[:bucket])
            for i, layer in enumerate(self.model.model.layers):
                hidden = layer(hidden, self.backend, meta)
                torch.cuda.synchronize(self.device)
                lines.append(f"  eager layer {i}: ok")
            self.model.compute_logits(self.model.model.norm(hidden), meta)
            torch.cuda.synchronize(self.device)
            lines.append("  eager forward of the same bucket succeeds: the failure is "
                         "capture-specific (an allocation, sync, or host-side op inside "
                         "the captured region). Re-run with CUDA_LAUNCH_BLOCKING=1 via "
                         "scripts/gpu_debug_capture.py to localise it.")
        except Exception as e:  # noqa: BLE001
            lines.append(f"  eager re-run fails too: {type(e).__name__}: {e}")
        return "\n".join(lines)

    # ---- replay ---------------------------------------------------------------------
    def bucket_for(self, batch: int) -> int | None:
        for b in self.buckets:
            if b >= batch:
                return b
        return None

    @torch.inference_mode()
    def release(self) -> None:
        """Destroy the captured graphs (`cudaGraphExecDestroy`). Under tensor parallelism
        this must happen before the process group goes: NCCL will not destroy or abort a
        communicator while a graph that captured its collectives still exists."""
        for g in self.graphs.values():
            g.reset()
        self.graphs.clear()

    def run(self, input_ids: Tensor, meta: AttnMetadata) -> Tensor:
        """Replay the decode graph for `meta`; returns logits `[B, vocab]` (a fresh copy).

        Falls back to eager execution when B exceeds the largest bucket.
        """
        assert self._captured, "call capture() first"
        assert not meta.is_prefill and all(n == 1 for n in meta.query_lens), \
            "CUDAGraphRunner only handles decode steps"
        batch = meta.num_seqs
        bucket = self.bucket_for(batch)
        if bucket is None:
            hidden = self.model(input_ids, self.backend, meta)
            return self.model.compute_logits(hidden, meta)
        assert meta.block_tables is not None and meta.slot_mapping is not None
        nb = meta.block_tables.shape[1]
        assert nb <= self.max_blocks, (nb, self.max_blocks)

        self.input_ids[:batch].copy_(input_ids)
        self.positions[:batch].copy_(meta.positions)
        self.slot_mapping[:batch].copy_(meta.slot_mapping)
        self.block_tables[:batch, :nb].copy_(meta.block_tables.clamp_min(0))
        if nb < self.max_blocks:
            self.block_tables[:batch, nb:].zero_()
        if meta.context_lens_t is not None:
            self.context_lens[:batch].copy_(meta.context_lens_t)
        else:
            self.context_lens[:batch].copy_(
                index_tensor(meta.context_lens, torch.int32, self.device))
        if batch < bucket:
            self._fill_padding(batch)
        self.graphs[bucket].replay()
        return self.logits[:batch].clone()

    def __repr__(self) -> str:
        return (f"CUDAGraphRunner(buckets={self.buckets}, max_blocks={self.max_blocks}, "
                f"scratch_block={self.scratch_block}, captured={self._captured})")
