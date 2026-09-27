"""Piecewise CUDA graphs for prefill and mixed (chunked-prefill) steps.

The full-step graph (`cuda_graphs.py`) needs every shape fixed, which a decode step has. A
mixed step has a variable token count and per-sequence query lengths, and its attention
needs per-step index plans, so it ran eagerly: ~20 Python launches per layer that a decode
step never pays. On Qwen2.5-0.5B that was the 11% the chunked-prefill default cost at
saturation (12,786 vs 14,394 tok/s), because at that size a step is launches, not math.

vLLM v1's answer is piecewise: capture everything *except* attention. Each decoder layer is
three pieces (`model/qwen2.py`, `model/deepseek.py`): `pre` (input norm with the residual
add, the attention projections, RoPE), `attend` (the kernel), `post` (output projection,
post-attention norm, MLP). `pre` and `post` are row-wise, so they run correctly on rows
padded up to a token bucket without touching anything shared; `attend` runs eagerly on the
real rows with the step's real metadata and writes the KV cache for real rows only, so no
scratch block is involved.

Static buffers, allocated once at the largest bucket and sliced per bucket: `hidden`,
`residual`, `positions`, one buffer per `pre` output, `attn_out`. Per layer l and bucket T:

    graph A[l, T]:  pre_l(hidden[:T], residual[:T], positions[:T]) -> pre buffers[:T], residual[:T]
    (eager)         attn_out[:n] = attend_l(pre buffers[:n], backend, meta)
    graph B[l, T]:  post_l(attn_out[:T], residual[:T]) -> hidden[:T], residual[:T]

A step is embed -> that per layer -> final norm -> logits: two replays plus the attention
launches per layer instead of every projection, norm and activation launched by hand. The
graphs share one memory pool; each keeps only what it copies into the static buffers, so
the pool is reused across layers and buckets.

Importable without CUDA. `capture()` needs a CUDA device; tests exercise the orchestration
on the CPU by substituting an eager "replay".
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from pagedserve.attn.base import AttnMetadata

DEFAULT_TOKEN_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def token_buckets(max_tokens: int, step: int = 0) -> tuple[int, ...]:
    """The capture buckets: powers of two (`step` 0), or powers of two up to `step` and then
    every `step` tokens (`step` 256: 16, 32, 64, 128, 256, 512, 768, ...). A mixed step is
    padded up to its bucket, and at 7B that padding is real compute (a 270-token step
    padded to 512 nearly doubles its MLP work), so finer buckets trade capture count and
    memory for less waste; at 0.5B the powers of two are enough because the step is launches."""
    if step <= 0:
        return DEFAULT_TOKEN_BUCKETS
    out = [b for b in DEFAULT_TOKEN_BUCKETS if b <= step]
    out += list(range(2 * step, max_tokens + 1, step))
    return tuple(out)


class _Eager:
    """A replayable that just runs the piece (CPU tests, and the no-CUDA fallback)."""

    def __init__(self, fn: Callable[[], None]) -> None:
        self.fn = fn

    def replay(self) -> None:
        self.fn()


class PiecewiseGraphRunner:
    def __init__(self, model, backend, max_tokens: int,
                 buckets: tuple[int, ...] = DEFAULT_TOKEN_BUCKETS,
                 device: torch.device | str = "cuda") -> None:
        self.model = model
        self.backend = backend
        self.device = torch.device(device)
        self.layers = list(model.model.layers)
        bs = sorted({b for b in buckets if b <= max_tokens} | {max_tokens})
        self.buckets: list[int] = bs
        self.max_bucket = bs[-1]
        self.graphs_pre: dict[tuple[int, int], object] = {}
        self.graphs_post: dict[tuple[int, int], object] = {}
        self._pool = None
        self._captured = False
        self._alloc_static()

    # ---- static buffers ------------------------------------------------------------
    def _alloc_static(self) -> None:
        t, dev = self.max_bucket, self.device
        cfg = self.model.config
        dtype = next(self.model.parameters()).dtype
        self.hidden = torch.zeros((t, cfg.hidden_size), dtype=dtype, device=dev)
        self.residual = torch.zeros((t, cfg.hidden_size), dtype=dtype, device=dev)
        self.positions = torch.zeros(t, dtype=torch.int64, device=dev)
        # Learn the pre-output and attention-output shapes from one eager call on the
        # (zero) buffers; only shapes/dtypes are used.
        with torch.inference_mode():
            pre, _ = self.layers[0].pre(self.hidden[:1], None, self.positions[:1])
        self.pre_bufs = [torch.zeros((t, *p.shape[1:]), dtype=p.dtype, device=dev) for p in pre]
        attn = self.layers[0].self_attn
        self.attn_out = torch.zeros((t, *attn.attn_out_shape(1)[1:]), dtype=dtype, device=dev)

    # ---- the two pieces on the static buffers ------------------------------------------
    def _run_pre(self, layer_idx: int, t: int) -> None:
        layer = self.layers[layer_idx]
        residual = None if layer_idx == 0 else self.residual[:t]
        pre, new_residual = layer.pre(self.hidden[:t], residual, self.positions[:t])
        for buf, out in zip(self.pre_bufs, pre, strict=True):
            buf[:t].copy_(out)
        self.residual[:t].copy_(new_residual)

    def _run_post(self, layer_idx: int, t: int) -> None:
        layer = self.layers[layer_idx]
        hidden, residual = layer.post(self.attn_out[:t], self.residual[:t])
        self.hidden[:t].copy_(hidden)
        self.residual[:t].copy_(residual)

    # ---- capture --------------------------------------------------------------------
    def _capture(self, fn: Callable[[], None]):
        """One graph for `fn` on the shared pool (CUDA); the eager fallback elsewhere."""
        if self.device.type != "cuda":
            return _Eager(fn)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool):
            fn()
        return g

    @torch.inference_mode()
    def capture(self, warmup_iters: int = 1) -> None:
        """Warm every piece up on a side stream (lazy kernel compilation, cuBLAS
        workspaces), then capture A and B per layer per bucket, largest bucket first."""
        if self.device.type == "cuda":
            self._pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for t in reversed(self.buckets):
                    for _ in range(warmup_iters):
                        for i in range(len(self.layers)):
                            self._run_pre(i, t)
                            self._run_post(i, t)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
        for t in reversed(self.buckets):
            for i in range(len(self.layers)):
                self.graphs_pre[i, t] = self._capture(lambda i=i, t=t: self._run_pre(i, t))
                self.graphs_post[i, t] = self._capture(lambda i=i, t=t: self._run_post(i, t))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._captured = True

    # ---- run ------------------------------------------------------------------------
    def bucket_for(self, num_tokens: int) -> int | None:
        for b in self.buckets:
            if b >= num_tokens:
                return b
        return None

    @torch.inference_mode()
    def run(self, input_ids: Tensor, meta: AttnMetadata) -> Tensor:
        """Logits `[num_seqs, vocab]` for any step (prefill, mixed, or decode). Steps larger
        than the largest bucket fall back to the eager forward."""
        assert self._captured, "call capture() first"
        n = meta.num_tokens
        t = self.bucket_for(n)
        if t is None:
            hidden = self.model(input_ids, self.backend, meta)
            return self.model.compute_logits(hidden, meta)
        inner = self.model.model
        self.hidden[:n].copy_(inner.embed_tokens(input_ids))
        self.positions[:n].copy_(meta.positions)
        pre_views = tuple(buf[:n] for buf in self.pre_bufs)
        for i, layer in enumerate(self.layers):
            self.graphs_pre[i, t].replay()
            self.attn_out[:n].copy_(layer.attend(pre_views, self.backend, meta))
            self.graphs_post[i, t].replay()
        normed, _ = inner.norm.forward_with_residual(self.hidden[:n], self.residual[:n])
        return self.model.compute_logits(normed, meta)

    def __repr__(self) -> str:
        return (f"PiecewiseGraphRunner(buckets={self.buckets}, layers={len(self.layers)}, "
                f"captured={self._captured})")
