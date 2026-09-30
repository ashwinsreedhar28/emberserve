"""LLMEngine: the step loop that ties scheduler, block manager, model, attention backend
and sampler together. One `step()` = one scheduler decision + one model forward + one
sampled token per scheduled sequence.

    engine = LLMEngine.from_pretrained("models/Qwen2.5-0.5B-Instruct")
    engine.add_request("r0", "Hello", SamplingParams.greedy(32))
    while engine.has_unfinished_requests():
        for out in engine.step():
            ...

With `EngineConfig.async_scheduling` a `step()` launches the next batch and returns the
outputs of the PREVIOUS one (see `_PendingStep`): same tokens, one step of latency inside
the engine, and the GPU never waits for the scheduler.
"""

from __future__ import annotations

import gc
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch

from pagedserve import dist as tpdist
from pagedserve.attn.base import AttentionBackend, AttnMetadata
from pagedserve.attn.naive import NaiveAttentionBackend
from pagedserve.attn.paged_torch import PagedTorchAttentionBackend
from pagedserve import steptrace
from pagedserve.config import EngineConfig, ModelConfig
from pagedserve.devutil import index_tensor
from pagedserve.kv.block_manager import BlockManager
from pagedserve.kv.cache import PagedKVCache
from pagedserve.model.qwen2 import Qwen2ForCausalLM
from pagedserve.model.weights import load_model
from pagedserve.sampling import Sampler, check_stop
from pagedserve.sched.request import FinishReason, Request, RequestOutput, SamplingParams
from pagedserve.sched.scheduler import Scheduler, SchedulerOutput
from pagedserve.tokenizer import IncrementalDetokenizer, Tokenizer, has_tokenizer


@dataclass
class StepStats:
    step: int
    is_prefill: bool
    num_seqs: int
    num_tokens: int
    num_preempted: int
    forward_ms: float
    sample_ms: float
    kv_utilization: float
    num_free_blocks: int
    # num_prefill_tokens + num_decode_tokens == num_tokens. A decode token is one new slot
    # for a request whose prefill is done; everything else (prompt, recompute after
    # preemption, chunked-prefill chunks) is a prefill token.
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0


# Backends whose decode step reads only static device tensors (meta.context_lens_t /
# meta.block_tables_nonneg) and can therefore be captured into a CUDA graph.
GRAPH_CAPABLE_BACKENDS = ("paged_flash", "paged_triton", "mla_triton")


@dataclass
class _PendingStep:
    """A launched step whose sampled tokens have not been read back yet.

    `sampled_dev[j]` is the token of `sampled_reqs[j]` (the rows with
    `prefill_complete`), still on the device; on CUDA a non-blocking copy of it into a
    pinned host buffer was enqueued right after sampling and `event` marks its completion,
    so reading the tokens waits for that copy only, never for the step launched after it.
    """

    so: SchedulerOutput
    sampled_reqs: list[Request]
    sampled_dev: torch.Tensor
    host: torch.Tensor | None = None
    event: torch.cuda.Event | None = None
    trace: "steptrace.StepRecord | None" = None

    def tokens(self) -> list[int]:
        if self.event is not None:
            self.event.synchronize()
            assert self.host is not None
            return self.host[:self.sampled_dev.shape[0]].tolist()
        return self.sampled_dev.tolist()


@dataclass
class StepPlan:
    """The host-side description of one step's inputs: what `_plan_inputs` computes from
    the scheduler's decision and `_materialize` turns into device tensors. Plain lists so it
    pickles small; under tensor parallelism the driver broadcasts it and every rank
    materializes the same tensors (`dist.py`)."""

    is_prefill: bool
    paged: bool
    seq_ids: list[int]
    query_lens: list[int]
    starts: list[int]
    context_lens: list[int]
    tokens: list[int]
    positions: list[int]
    slots: list[int]
    cu: list[int]
    tables: list[list[int]]
    fill_rows: list[int] = field(default_factory=list)
    fill_src: list[int] = field(default_factory=list)
    logit_rows: list[int] = field(default_factory=list)


KV_WORKSPACE_BYTES = 1 << 30  # graph mempool, allocator slack, sampler scratch, Triton/cuBLAS
MIN_GPU_BLOCKS = 64


def activation_reserve_bytes(model_config: ModelConfig, engine_config: EngineConfig) -> int:
    """Memory the engine needs outside the cache at its busiest step, kept free when the
    cache is sized: KV_WORKSPACE_BYTES plus the two allocations that scale with the
    configuration — the MLP's gate/up activations for one prefill chunk
    (`max_num_batched_tokens x intermediate x 2` in the model dtype) and the logits for the
    largest decode batch (`max_num_seqs x vocab`, fp16 plus the fp32 copy the sampler
    takes). About 1.4 GB for the 7B at the CLI defaults, 1.3 GB for the 0.5B."""
    esize = torch.tensor([], dtype=engine_config.dtype).element_size()
    mlp = engine_config.max_num_batched_tokens * model_config.intermediate_size * 2 * esize
    logits = engine_config.max_num_seqs * model_config.vocab_size * (esize + 4)
    return KV_WORKSPACE_BYTES + mlp + logits


def kv_blocks_for(free_bytes: int, utilization: float, bytes_per_block: int,
                  reserve_bytes: int = KV_WORKSPACE_BYTES) -> int:
    """KV blocks that fit in `free_bytes` of device memory read *with the weights already
    resident*: `free * utilization - reserve`, floored at MIN_GPU_BLOCKS.

    The weights are not subtracted here. They used to be, on top of a `free` that already
    excluded them, and an 80 GB card never noticed (7B: 42 GB of cache instead of 57) while
    a 24 GB card serving the 7B went to the 64-block floor — 16K tokens of cache, about 30
    sequences in flight, 1,118 tok/s on an RTX 4090 where the batch should have been KV-bound
    at three times that. The reserve is `activation_reserve_bytes`: on the same card a
    hand-set 512 blocks (7.0 GiB) left 71 MiB for the bucket-256 graph capture and failed."""
    budget = int(free_bytes * utilization) - reserve_bytes
    return max(budget // bytes_per_block, MIN_GPU_BLOCKS)


def default_num_blocks(model_config: ModelConfig, engine_config: EngineConfig) -> int:
    """How many KV blocks to allocate when the user did not say.

    CUDA: `kv_blocks_for` over the device's free memory, which is read after
    `from_pretrained` / `build_tp_model` have loaded the weights. CPU/MPS: a fixed 2048
    blocks (block_size 16 -> 32K cached tokens), which is ~800 MB at fp32 for the 0.5B
    model and plenty for local correctness work.
    """
    bytes_per_block = model_config.kv_bytes_per_token(engine_config.dtype) * engine_config.block_size
    if engine_config.device.startswith("cuda") and torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info(torch.device(engine_config.device))
        return kv_blocks_for(free, engine_config.gpu_memory_utilization, bytes_per_block,
                             activation_reserve_bytes(model_config, engine_config))
    return 2048


class LLMEngine:
    def __init__(self, model: Qwen2ForCausalLM, model_config: ModelConfig,
                 engine_config: EngineConfig, tokenizer: Tokenizer | None = None) -> None:
        self.model = model.eval()
        self.model_config = model_config
        self.config = engine_config
        self.device = torch.device(engine_config.device)
        self.dtype = engine_config.dtype
        self.tokenizer = tokenizer
        self.detok = IncrementalDetokenizer(tokenizer)
        self.eos_token_id = tokenizer.eos_token_id if tokenizer else model_config.eos_token_id
        # Every id that ends generation: the config's list plus the tokenizer's own eos.
        self.eos_token_ids: frozenset[int] = model_config.all_eos_token_ids | {self.eos_token_id}
        # Tensor parallelism (dist.py): this rank's share of the heads decides the KV cache
        # geometry; `model_config` stays the full model (vocabulary, eos ids) and
        # `local_config` is what the cache and the attention backend are built from.
        self.tp = tpdist.get_tp()
        if engine_config.tensor_parallel_size != self.tp.size:
            raise ValueError(f"tensor_parallel_size={engine_config.tensor_parallel_size} but the "
                             f"process group has {self.tp.size} rank(s); build a tensor-parallel "
                             "engine with LLMEngine.from_pretrained / launch_tp")
        self.local_config = model_config.shard(self.tp.size)
        self._tp_workers: list = []
        # True from the plan broadcast until this rank's forward has been launched: a step
        # that raises in between leaves the workers waiting on collectives that will never
        # come, and `shutdown()` must kill them rather than wait on a broadcast of its own.
        self._tp_step_open = False

        self.boot_phases: dict[str, float] = {}
        self.boot_notes = ""
        t_phase = time.perf_counter()
        num_blocks = engine_config.num_gpu_blocks or default_num_blocks(self.local_config, engine_config)
        num_blocks = tpdist.all_reduce_min(num_blocks)  # every rank's cache has the same shape
        engine_config.num_gpu_blocks = num_blocks
        if engine_config.attn_backend == "naive":
            # The naive backend has no shared physical blocks, so prefix reuse is impossible.
            engine_config.enable_prefix_caching = False
        # CUDA graphs need one scratch KV block the BlockManager never hands out
        # (padding rows of a graph bucket write their k/v there). See attn/cuda_graphs.py.
        use_graphs = (engine_config.enable_cuda_graphs and self.device.type == "cuda"
                      and engine_config.attn_backend in GRAPH_CAPABLE_BACKENDS)
        if engine_config.enable_cuda_graphs and not use_graphs:
            warnings.warn("enable_cuda_graphs ignored: needs device=cuda and attn_backend in "
                          f"{GRAPH_CAPABLE_BACKENDS}", stacklevel=2)
        self.scratch_block: int | None = num_blocks - 1 if use_graphs else None
        self.block_manager = BlockManager(num_blocks - 1 if use_graphs else num_blocks,
                                          engine_config.block_size)
        self.scheduler = Scheduler(engine_config, self.block_manager)
        self.backend: AttentionBackend = self._make_backend(num_blocks)
        self.sampler = Sampler(self.device)
        self.boot_phases["kv_cache_s"] = time.perf_counter() - t_phase  # sizing + allocation
        # Optional CUDA-graph decode runner (attn/cuda_graphs.py); installed on GPU only.
        self.graph_runner = None
        if use_graphs:
            from pagedserve.attn.cuda_graphs import CUDAGraphRunner
            self.graph_runner = CUDAGraphRunner(
                self.model, self.backend, max_model_len=engine_config.max_model_len,
                block_size=engine_config.block_size, scratch_block=self.scratch_block,
                max_batch=engine_config.max_num_seqs, device=self.device)
            t_phase = time.perf_counter()
            self.graph_runner.capture()
            self.boot_phases["capture_full_graphs_s"] = time.perf_counter() - t_phase
        self.piecewise_runner = None
        if use_graphs and engine_config.piecewise_cuda_graphs:
            from pagedserve.attn.piecewise_graphs import PiecewiseGraphRunner, token_buckets
            max_tokens = max(engine_config.max_num_batched_tokens, engine_config.max_num_seqs)
            self.piecewise_runner = PiecewiseGraphRunner(
                self.model, self.backend, max_tokens=max_tokens,
                buckets=token_buckets(max_tokens, engine_config.piecewise_bucket_step),
                device=self.device)
            t_phase = time.perf_counter()
            self.piecewise_runner.capture()
            self.boot_phases["capture_piecewise_graphs_s"] = time.perf_counter() - t_phase
        elif engine_config.piecewise_cuda_graphs:
            warnings.warn("piecewise_cuda_graphs ignored: needs enable_cuda_graphs on CUDA",
                          stacklevel=2)
        self._next_seq_id = 0
        self._final_text: dict[str, str] = {}
        self._step_count = 0
        self.stats: list[StepStats] = []
        self.keep_stats = True
        # Async scheduling state: the launched-but-unread step, and two pinned host
        # buffers its tokens land in (alternating, so a buffer is never overwritten before
        # the step that filled it has been resolved).
        self.async_scheduling = bool(engine_config.async_scheduling)
        self._pending: _PendingStep | None = None
        self._host_bufs: list[torch.Tensor] = []
        self._host_idx = 0
        if self.async_scheduling and self.device.type == "cuda":
            self._host_bufs = [torch.empty(engine_config.max_num_seqs, dtype=torch.int64,
                                           pin_memory=True) for _ in range(2)]
        # Whether the last `step()` call scheduled any work (distinguishes "nothing to do"
        # from "launched, outputs come next call" for the callers' stuck-queue check).
        self.last_step_scheduled = False
        # Speculative decoding (spec.py): n-gram size and draft count; needs the last
        # token on the host before the next schedule, so it turns the async lookahead off.
        self.spec_ngram = int(engine_config.speculative_ngram)
        self.spec_k = int(engine_config.num_speculative_tokens) if self.spec_ngram > 0 else 0
        if self.spec_k > 0 and self.async_scheduling:
            warnings.warn("speculative decoding needs the sampled token on the host before the "
                          "next step is scheduled: async_scheduling turned off", stacklevel=2)
            self.async_scheduling = False
            self._host_bufs = []
        # Speculation counters (drafts proposed / accepted), for /metrics and the tests.
        self.spec_drafted = 0
        self.spec_accepted = 0
        # Per-step trace (steptrace.py), only with PAGEDSERVE_STEP_TRACE set; driver only.
        self._tracer: steptrace.StepTracer | None = None
        self._open_trace()

    def _open_trace(self) -> None:
        path = steptrace.trace_path()
        if path is None or not self.tp.is_driver:
            return
        if self._tracer is not None:
            self._tracer.close()
        self._tracer = steptrace.StepTracer(path, cuda=self.device.type == "cuda",
                                            boot=self.boot_phases)

    def _trace_record(self, so: SchedulerOutput, t_start: float, sched_ms: float) -> "steptrace.StepRecord":
        graph = self.graph_runner is not None and not so.is_prefill
        piecewise = self.piecewise_runner is not None and not graph
        kind, n_dec, n_pre, n_pre_seqs, max_chunk = steptrace.classify(list(so.query_lens), graph, piecewise)
        return steptrace.StepRecord(step=self._step_count, kind=kind, n_seqs=len(so.scheduled),
                                    n_decode=n_dec, n_prefill_tokens=n_pre, n_prefill_seqs=n_pre_seqs,
                                    max_chunk=max_chunk, t_start=t_start, host_sched_ms=sched_ms)

    # ---- construction -----------------------------------------------------------
    def _make_backend(self, num_blocks: int) -> AttentionBackend:
        name = self.config.attn_backend
        cfg = self.local_config
        if self.model_config.mla is not None:
            from pagedserve.kv.cache import PagedLatentCache

            cache = PagedLatentCache(cfg, num_blocks, self.config.block_size,
                                     self.device, self.dtype)
            # The dense backend names map onto their latent-attention counterparts so the
            # CLI defaults (paged_flash / paged_triton on CUDA) work unchanged.
            if name in ("mla_triton", "paged_flash", "paged_triton"):
                from pagedserve.attn.mla_triton import MLATritonBackend

                return MLATritonBackend(cfg, cache)  # type: ignore[return-value]
            if name in ("mla_torch", "naive", "paged_torch"):
                from pagedserve.attn.mla_torch import MLATorchBackend

                return MLATorchBackend(cfg, cache)  # type: ignore[return-value]
            raise ValueError(f"unknown attn_backend {name!r} for a latent-attention model")
        if name == "naive":
            return NaiveAttentionBackend(cfg, self.device, self.dtype)
        if name == "paged_torch":
            cache = PagedKVCache(cfg, num_blocks, self.config.block_size,
                                 self.device, self.dtype)
            return PagedTorchAttentionBackend(cfg, cache)
        if name == "paged_flash":
            from pagedserve.attn.paged_flash import PagedFlashAttentionBackend
            cache = PagedKVCache(cfg, num_blocks, self.config.block_size,
                                 self.device, self.dtype)
            return PagedFlashAttentionBackend(cfg, cache)
        if name == "paged_triton":
            from pagedserve.attn.paged_triton import PagedTritonAttentionBackend
            cache = PagedKVCache(cfg, num_blocks, self.config.block_size,
                                 self.device, self.dtype)
            return PagedTritonAttentionBackend(cfg, cache)
        raise ValueError(f"unknown attn_backend {name!r}")

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, engine_config: EngineConfig | None = None,
                        load_tokenizer: bool = True, **overrides) -> "LLMEngine":
        engine_config = engine_config or EngineConfig()
        for k, v in overrides.items():
            setattr(engine_config, k, v)
        engine_config.model_dir = str(model_dir)
        if engine_config.tensor_parallel_size > 1:
            return cls.launch_tp(tpdist.WorkerSpec(engine_config, model_dir=str(model_dir)),
                                 load_tokenizer=load_tokenizer)
        t0 = time.perf_counter()
        model = load_model(model_dir, device=engine_config.device, dtype=engine_config.dtype)
        if engine_config.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()  # the load's copies are done: the phase is honest
        load_s = time.perf_counter() - t0
        quant_s = 0.0
        if engine_config.quantization:
            from pagedserve.model.quant import quantize_model

            t0 = time.perf_counter()
            quantize_model(model, engine_config.quantization)
            quant_s = time.perf_counter() - t0
        model_config = ModelConfig.from_hf_dir(model_dir)
        tokenizer = None
        if load_tokenizer and has_tokenizer(model_dir):
            try:
                tokenizer = Tokenizer(model_dir)
            except ImportError:
                tokenizer = None
        engine = cls(model, model_config, engine_config, tokenizer)
        stats = getattr(model, "load_stats", None)  # the streaming loader's (model/fastload.py)
        read = {"weights_read_s": stats.seconds} if stats is not None else {}
        engine.boot_phases = {"load_weights_s": load_s, **read,
                              **({"quantize_s": quant_s} if quant_s else {}), **engine.boot_phases}
        if stats is not None:
            engine.boot_notes = (f"weights {stats.bytes / 1e9:.2f} GB in {stats.seconds:.2f} s = "
                                 f"{stats.gb_per_s:.2f} GB/s ({stats.threads} readers, "
                                 f"{stats.buffer_mb:g} MB buffers"
                                 + (f", {stats.wait_seconds:.2f} s waiting for the download"
                                    if stats.wait_seconds else "") + ")")
        engine._open_trace()  # (re-)writes the trace header with the load phase included
        return engine

    @classmethod
    def launch_tp(cls, spec: "tpdist.WorkerSpec", load_tokenizer: bool = True) -> "LLMEngine":
        """Build the driver (rank 0) of a tensor-parallel engine in this process: start the
        worker processes, join the group, load this rank's shard, construct. The workers
        build the same engine from the same spec and then run `worker_loop`; `shutdown()`
        (or garbage collection of the driver) stops them."""
        ecfg = spec.engine_config
        world = ecfg.tensor_parallel_size
        init_method = f"tcp://127.0.0.1:{tpdist.free_port()}"
        procs = tpdist.spawn_workers(spec, world, init_method)
        ecfg.device = tpdist.rank_device(ecfg.device, 0)
        try:
            tpdist.init_tp(0, world, init_method, ecfg.device)
            model, model_config = tpdist.build_tp_model(spec, ecfg.device, ecfg.dtype)
            tokenizer = None
            if spec.model_dir and load_tokenizer and has_tokenizer(spec.model_dir):
                try:
                    tokenizer = Tokenizer(spec.model_dir)
                except ImportError:
                    tokenizer = None
            engine = cls(model, model_config, ecfg, tokenizer)
        except BaseException:
            for p in procs:
                p.kill()
            tpdist.destroy_tp()
            raise
        engine._tp_workers = procs
        return engine

    def shutdown(self) -> None:
        """Leave the tensor-parallel group (any rank; the driver stops its workers first).
        Idempotent."""
        driver = self.tp.size > 1 and self.tp.is_driver and tpdist.is_initialized()
        failed = False
        if driver:
            if self._tp_step_open:
                # A step raised after its plan went out: the workers are (or will be) blocked
                # in collectives nothing pairs with, so a "stop" broadcast would hang here
                # and hide the exception. Kill them and tear the group down with a timeout.
                failed = True
                for p in self._tp_workers:
                    p.kill()
            else:
                tpdist.broadcast_object(("stop",))
        if tpdist.is_initialized():
            # NCCL will neither destroy nor abort a communicator while a CUDA graph that
            # captured its collectives exists: drop the graphs first, on every rank, then
            # leave the group *before* the driver waits for the worker processes.
            self.release_graphs()
            tpdist.destroy_tp(timeout_s=15.0 if failed else 60.0)
        if driver:
            tpdist.stop_workers(self._tp_workers)
            self._tp_workers = []
        self.tp = tpdist.get_tp()

    def release_graphs(self) -> None:
        """Destroy the captured CUDA graphs (the engine falls back to eager steps)."""
        for runner in (self.graph_runner, self.piecewise_runner):
            if runner is not None:
                runner.release()
        self.graph_runner = self.piecewise_runner = None
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.synchronize(self.device)

    def __del__(self) -> None:
        try:
            if getattr(self, "_tp_workers", None):
                self.shutdown()
        except Exception:  # noqa: BLE001
            pass

    def worker_loop(self) -> None:
        """Rank > 0: follow the driver's steps until it says stop. Each message is a
        `StepPlan`; the worker materializes it, receives the driver's input ids and runs
        the same forward (whose collectives pair with the driver's)."""
        assert self.tp.size > 1 and not self.tp.is_driver
        with torch.inference_mode():
            while True:
                msg = tpdist.broadcast_object(None)
                if msg[0] == "stop":
                    return
                if msg[0] == "reset":
                    self.backend.reset()
                    continue
                plan: StepPlan = msg[1]
                input_ids, meta = self._materialize(plan)
                tpdist.broadcast_tensor(input_ids)
                self._forward(input_ids, meta, plan.is_prefill)

    # ---- request API ----------------------------------------------------------------
    def add_request(self, request_id: str, prompt: str | list[int],
                    sampling_params: SamplingParams | None = None,
                    arrival_time: float | None = None, metadata: dict | None = None) -> Request:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("engine has no tokenizer; pass prompt token ids")
            prompt_ids = self.tokenizer.encode(prompt)
        else:
            prompt_ids = list(prompt)
        if prompt_ids and (max(prompt_ids) >= self.model_config.vocab_size or min(prompt_ids) < 0):
            raise ValueError(f"prompt token id out of vocabulary (vocab_size "
                             f"{self.model_config.vocab_size})")
        req = Request(request_id=request_id, prompt_token_ids=prompt_ids,
                      sampling_params=sampling_params or SamplingParams(),
                      seq_id=self._next_seq_id, metadata=metadata or {})
        if arrival_time is not None:
            req.arrival_time = arrival_time
        self._next_seq_id += 1
        self.scheduler.add_request(req)
        return req

    def abort_request(self, request_id: str) -> None:
        req = self.scheduler.abort_request(request_id)
        if req is not None:
            self.backend.free_sequence(req.seq_id)
            self.detok.reset(request_id)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests() or self._pending is not None

    # ---- the step ---------------------------------------------------------------------
    @torch.inference_mode()
    def step(self) -> list[RequestOutput]:
        tr = self._tracer
        if tr is None:
            return self._step_async() if self.async_scheduling else self._step_sync()
        tr.begin_step()
        self._trace_current = None
        try:
            return self._step_async() if self.async_scheduling else self._step_sync()
        finally:
            tr.end_step(self._trace_current)

    def _step_sync(self) -> list[RequestOutput]:
        tr = self._tracer
        t_sched = time.perf_counter()
        sched_out = self.scheduler.schedule()
        for req in sched_out.preempted:
            # Recompute-on-readmit: the backend must drop whatever it held for this seq.
            self.backend.free_sequence(req.seq_id)
        self.last_step_scheduled = not sched_out.is_empty
        if sched_out.is_empty:
            if tr is not None:
                tr.idle()
            return []
        self._step_count += 1
        rec = None
        if tr is not None:
            t_built = time.perf_counter()
            rec = self._trace_current = self._trace_record(sched_out, t_sched, (t_built - t_sched) * 1e3)

        input_ids, meta = self._build_inputs(sched_out)
        t0 = time.perf_counter()
        if rec is not None:
            rec.host_build_ms = (t0 - t_built) * 1e3
            rec.ev_start = tr.event()
        logits = self._forward(input_ids, meta, sched_out)
        t1 = time.perf_counter()
        # Only rows whose prefill completes this step get a token; a partial prefill
        # chunk's last-token logits are meaningless (its next chunk continues the prompt).
        complete = sched_out.prefill_complete
        if meta.logit_indices is not None:
            # speculative step: one logits row per query position of a draft sequence
            sampled = self._sample_draft_rows(logits, sched_out)
        elif all(complete):
            sampled = self.sampler.sample(logits, sched_out.scheduled)
        else:
            idx = [i for i, c in enumerate(complete) if c]
            reqs = [sched_out.scheduled[i] for i in idx]
            sampled_rows = (self.sampler.sample(logits.index_select(0, index_tensor(idx, torch.long, logits.device)), reqs)
                            if idx else [])
            sampled = [None] * len(complete)
            for i, tok in zip(idx, sampled_rows, strict=True):
                sampled[i] = tok
        t2 = time.perf_counter()
        if rec is not None:
            rec.ev_end = tr.event()
            rec.host_launch_ms = (t1 - t0) * 1e3

        self._advance(sched_out)
        outputs = self._postprocess(sched_out, sampled)
        if self.spec_k:
            self._propose_drafts(sched_out)
        if rec is not None:
            rec.host_resolve_ms = (time.perf_counter() - t1) * 1e3  # sample (a sync) + postprocess
            tr.finish(rec)
        if self.keep_stats:
            st = self.block_manager.stats()
            self.stats.append(StepStats(
                step=self._step_count, is_prefill=sched_out.is_prefill,
                num_seqs=len(sched_out.scheduled), num_tokens=sched_out.num_tokens,
                num_preempted=len(sched_out.preempted), forward_ms=(t1 - t0) * 1e3,
                sample_ms=(t2 - t1) * 1e3, kv_utilization=st.utilization,
                num_free_blocks=st.num_free,
                num_prefill_tokens=sched_out.num_prefill_tokens,
                num_decode_tokens=sched_out.num_decode_tokens))
        return outputs

    # ---- speculative decoding ------------------------------------------------------------
    def _sample_draft_rows(self, logits: torch.Tensor, so: SchedulerOutput) -> list:
        """`logits` holds, per scheduled request, either its last row or (draft rows) one
        row per query position. Returns per request: an int, a list of ints (draft rows,
        `query_len` of them), or None (partial chunk)."""
        reqs: list[Request] = []
        keep: list[int] = []  # logits rows that belong to completed rows
        pos = 0
        for req, qlen, done in zip(so.scheduled, so.query_lens, so.prefill_complete, strict=True):
            count = qlen if req.draft_tokens else 1
            if done:
                reqs.extend([req] * count)
                keep.extend(range(pos, pos + count))
            pos += count
        assert pos == logits.shape[0], (pos, logits.shape)
        flat = self.sampler.sample(logits[keep] if len(keep) < pos else logits, reqs) if reqs else []
        out: list = []
        pos = 0
        for req, qlen, done in zip(so.scheduled, so.query_lens, so.prefill_complete, strict=True):
            if not done:
                out.append(None)
            elif req.draft_tokens:
                out.append(flat[pos:pos + qlen])
                pos += qlen
            else:
                out.append(flat[pos])
                pos += 1
        return out

    def _propose_drafts(self, so: SchedulerOutput) -> None:
        """Guess the next tokens of every greedy request that will decode next step."""
        from pagedserve.spec import propose_ngram

        for req in so.scheduled:
            req.draft_tokens = []
            if req.is_finished or not req.sampling_params.is_greedy or req.is_prefill:
                continue
            room = min(req.sampling_params.max_tokens - req.num_output_tokens,
                       self.config.max_model_len - req.num_tokens) - 1
            k = min(self.spec_k, room)
            if k <= 0:
                continue
            req.draft_tokens = propose_ngram(req.all_token_ids, self.spec_ngram, k)

    def _forward(self, input_ids: torch.Tensor, meta: AttnMetadata,
                 so: SchedulerOutput | bool) -> torch.Tensor:
        is_prefill = so if isinstance(so, bool) else so.is_prefill
        if self.graph_runner is not None and not is_prefill:
            logits = self.graph_runner.run(input_ids, meta)
        elif self.piecewise_runner is not None:
            logits = self.piecewise_runner.run(input_ids, meta)
        else:
            hidden = self.model(input_ids, self.backend, meta)
            logits = self.model.compute_logits(hidden, meta)
        self._tp_step_open = False  # every collective of this step is enqueued
        return logits

    @staticmethod
    def _advance(so: SchedulerOutput) -> None:
        """Account the K/V this step writes. Independent of the sampled tokens, so under
        async scheduling it happens at launch, before the next schedule() looks."""
        for req, qlen in zip(so.scheduled, so.query_lens, strict=True):
            req.num_computed_tokens += qlen

    def _step_async(self) -> list[RequestOutput]:
        """Launch this step, then resolve the previous one.

        Order matters: `_build_inputs` reads the previous step's sampled tensor for the
        decode rows whose token is still on the device (`Request.pending_row`), so the
        launch goes first, and only then are the previous tokens read back (their
        device->host copy was enqueued before this step's kernels, so the wait is short)
        and turned into outputs. Requests that finished meanwhile (aborted, or ended by
        the previous step's token) are skipped when this step resolves.
        """
        tr = self._tracer
        t_sched = time.perf_counter()
        sched_out = self.scheduler.schedule()
        for req in sched_out.preempted:
            self.backend.free_sequence(req.seq_id)
        self.last_step_scheduled = not sched_out.is_empty
        launched: _PendingStep | None = None
        rec = None
        if not sched_out.is_empty:
            self._step_count += 1
            if tr is not None:
                t_built = time.perf_counter()
                rec = self._trace_current = self._trace_record(sched_out, t_sched,
                                                               (t_built - t_sched) * 1e3)
            input_ids, meta = self._build_inputs(sched_out)
            t0 = time.perf_counter()
            if rec is not None:
                rec.host_build_ms = (t0 - t_built) * 1e3
                rec.ev_start = tr.event()
            logits = self._forward(input_ids, meta, sched_out)
            complete = sched_out.prefill_complete
            if all(complete):
                reqs = list(sched_out.scheduled)
                sampled_dev = self.sampler.sample_tensor(logits, reqs)
            else:
                idx = [i for i, c in enumerate(complete) if c]
                reqs = [sched_out.scheduled[i] for i in idx]
                # (an index tensor from a Python list would be a synchronizing copy)
                sampled_dev = (self.sampler.sample_tensor(
                    logits.index_select(0, index_tensor(idx, torch.long, logits.device)), reqs)
                    if idx else torch.empty(0, dtype=torch.int64, device=logits.device))
            host = event = None
            if self._host_bufs:
                host = self._host_bufs[self._host_idx]
                self._host_idx ^= 1
                host[:sampled_dev.shape[0]].copy_(sampled_dev, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
            t1 = time.perf_counter()
            if rec is not None:
                rec.ev_end = tr.event()
                rec.host_launch_ms = (t1 - t0) * 1e3
            self._advance(sched_out)
            # The rows sampled here own the pending tokens now; every other scheduled row
            # (a partial chunk) has nothing outstanding.
            for req in sched_out.scheduled:
                req.pending_row = None
            for j, req in enumerate(reqs):
                req.pending_row = j
            launched = _PendingStep(sched_out, reqs, sampled_dev, host, event, trace=rec)
            if self.keep_stats:
                st = self.block_manager.stats()
                self.stats.append(StepStats(
                    step=self._step_count, is_prefill=sched_out.is_prefill,
                    num_seqs=len(sched_out.scheduled), num_tokens=sched_out.num_tokens,
                    num_preempted=len(sched_out.preempted), forward_ms=(t1 - t0) * 1e3,
                    sample_ms=0.0, kv_utilization=st.utilization, num_free_blocks=st.num_free,
                    num_prefill_tokens=sched_out.num_prefill_tokens,
                    num_decode_tokens=sched_out.num_decode_tokens))
        outputs: list[RequestOutput] = []
        if self._pending is not None:
            # Rows re-sampled by the step just launched keep their (new) pending_row.
            keep = {id(r) for r in launched.sampled_reqs} if launched is not None else set()
            t_res = time.perf_counter()
            outputs = self._resolve(self._pending, keep)
            prev = self._pending.trace
            if prev is not None and tr is not None:
                prev.host_resolve_ms = (time.perf_counter() - t_res) * 1e3
                tr.finish(prev)
        if launched is None and tr is not None:
            tr.idle()  # nothing queued behind: the next step's GPU gap is not host lag
        self._pending = launched
        return outputs

    def _resolve(self, pending: _PendingStep, keep: set[int]) -> list[RequestOutput]:
        tokens = pending.tokens()
        so = pending.so
        sampled: list[int | None] = [None] * len(so.scheduled)
        by_id = {id(r): i for i, r in enumerate(so.scheduled)}
        for req, tok in zip(pending.sampled_reqs, tokens, strict=True):
            sampled[by_id[id(req)]] = tok
            if id(req) not in keep:
                req.pending_row = None
        return self._postprocess(so, sampled)

    def step_logits(self, so: SchedulerOutput, all_positions: bool = False) -> torch.Tensor:
        """The scheduled step's logits without sampling or bookkeeping (the golden checks).
        `all_positions` projects every token's row, not only each sequence's last one. The
        choice travels in the plan, so under tensor parallelism every rank gathers the same
        logits shape: calling the model directly on the driver with a different row set
        while the workers ran the plan's is a collective mismatch, i.e. a hang."""
        plan = self._plan_inputs(so)
        if all_positions:
            plan.logit_rows = list(range(len(plan.tokens)))
        if self.tp.size > 1:
            self._tp_step_open = True
            tpdist.broadcast_object(("step", plan))
        input_ids, meta = self._materialize(plan)
        if self.tp.size > 1:
            tpdist.broadcast_tensor(input_ids)
        logits = self.model.compute_logits(self.model(input_ids, self.backend, meta), meta)
        self._tp_step_open = False
        return logits

    def _build_inputs(self, so: SchedulerOutput) -> tuple[torch.Tensor, AttnMetadata]:
        """Per-step tensors in two host->device copies (one int64, one int32) instead of one
        per field: at 200 running sequences the original six `torch.tensor(..., device=cuda)`
        calls plus a per-row block-table build were ~2 ms of a ~8 ms step. Under tensor
        parallelism the plan goes to the workers first, and the finished `input_ids` (which
        may hold tokens gathered on the device) after."""
        plan = self._plan_inputs(so)
        if self.tp.size > 1:
            self._tp_step_open = True
            tpdist.broadcast_object(("step", plan))
        input_ids, meta = self._materialize(plan)
        if self.tp.size > 1:
            tpdist.broadcast_tensor(input_ids)
        return input_ids, meta

    def _plan_inputs(self, so: SchedulerOutput) -> StepPlan:
        reqs = so.scheduled
        bm = self.block_manager
        bs = self.config.block_size
        paged = self.config.attn_backend != "naive" or self.model_config.mla is not None
        seq_ids = [r.seq_id for r in reqs]
        starts = [r.num_computed_tokens for r in reqs]
        tokens: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        context_lens: list[int] = []
        cu = [0]
        tables: list[list[int]] = []
        fill_rows: list[int] = []  # packed rows whose token is still on the device
        fill_src: list[int] = []  # ... and its row in the pending step's sampled tensor
        logit_rows: list[int] = []  # speculative: every row of a draft sequence
        any_drafts = False
        for r, start, qlen in zip(reqs, starts, so.query_lens, strict=True):
            end = start + qlen
            known = r.all_token_ids
            if r.draft_tokens:  # the real last token followed by the guesses
                known = known + r.draft_tokens
                any_drafts = True
                logit_rows.extend(range(len(tokens), len(tokens) + qlen))
            else:
                logit_rows.append(len(tokens) + qlen - 1)
            if start == len(known):  # async: the token at `start` was sampled last step
                assert qlen == 1 and r.pending_row is not None, (qlen, r.pending_row, r.request_id)
                fill_rows.append(len(tokens))
                fill_src.append(r.pending_row)
                tokens.append(0)
            else:
                ids = known[start:end]
                assert len(ids) == qlen, (len(ids), qlen, r.request_id)
                tokens.extend(ids)
            context_lens.append(end)
            cu.append(cu[-1] + qlen)
            if qlen == 1:
                positions.append(start)
            else:
                positions.extend(range(start, end))
            if paged:
                table = bm.get_block_table(r.seq_id)
                tables.append(table)
                if qlen == 1:
                    slots.append(table[start // bs] * bs + start % bs)
                else:
                    slots.extend(table[p // bs] * bs + p % bs for p in range(start, end))
        if not any_drafts:
            logit_rows = []  # the default (last row per sequence) is computed on the device
        return StepPlan(is_prefill=so.is_prefill, paged=paged, seq_ids=seq_ids,
                        query_lens=list(so.query_lens), starts=starts, context_lens=context_lens,
                        tokens=tokens, positions=positions, slots=slots, cu=cu, tables=tables,
                        fill_rows=fill_rows, fill_src=fill_src, logit_rows=logit_rows)

    def _materialize(self, plan: StepPlan) -> tuple[torch.Tensor, AttnMetadata]:
        """The plan's device tensors. Rows whose token is still on the device (`fill_rows`)
        are gathered from the pending step's sampled tensor on the driver; a worker leaves
        them and receives the driver's finished `input_ids` instead."""
        tokens, positions, slots = plan.tokens, plan.positions, plan.slots
        fill_rows, fill_src, logit_rows = plan.fill_rows, plan.fill_src, plan.logit_rows
        n, b = len(tokens), len(plan.seq_ids)
        dev = self.device
        pin = dev.type == "cuda"
        # int64 block: [tokens | positions | slots | fill rows | fill sources | logit rows]
        i64 = torch.tensor(tokens + positions + slots + fill_rows + fill_src + logit_rows,
                           dtype=torch.int64, pin_memory=pin)
        i64 = i64.to(dev, non_blocking=pin)
        input_ids, pos = i64[:n], i64[n:2 * n]
        o = 2 * n + len(slots)
        if fill_rows:
            m = len(fill_rows)
            if self.tp.is_driver:
                assert self._pending is not None
                input_ids.index_copy_(0, i64[o:o + m],
                                      self._pending.sampled_dev.index_select(0, i64[o + m:o + 2 * m]))
            o += 2 * m
        logit_indices = i64[o:o + len(logit_rows)] if logit_rows else None
        # int32 block: [context_lens | cu_seqlens | block tables (padded with -1)]
        paged, tables, bs = plan.paged, plan.tables, self.config.block_size
        max_blocks = max((len(t) for t in tables), default=0) if paged else 0
        flat32 = plan.context_lens + plan.cu
        if paged and max_blocks:
            pad = [-1] * max_blocks
            for t in tables:
                flat32 += t if len(t) == max_blocks else t + pad[:max_blocks - len(t)]
        i32 = torch.tensor(flat32, dtype=torch.int32, pin_memory=pin).to(dev, non_blocking=pin)
        meta = AttnMetadata(
            is_prefill=plan.is_prefill, seq_ids=plan.seq_ids, query_lens=list(plan.query_lens),
            context_lens=plan.context_lens, positions=pos,
            num_cached_tokens=plan.starts if plan.is_prefill else [],
            cu_seqlens_q=i32[b:2 * b + 1],
        )
        meta.context_lens_t = i32[:b]
        meta.logit_indices = logit_indices
        if paged:
            meta.slot_mapping = i64[2 * n:3 * n]
            meta.block_tables = i32[2 * b + 1:2 * b + 1 + b * max_blocks].view(b, max_blocks)
            meta.block_size = bs
        return input_ids, meta

    def _postprocess(self, so: SchedulerOutput, sampled: list[int | None]) -> list[RequestOutput]:
        """Append/stop-check the sampled rows (`_advance` has already accounted the K/V).
        `sampled[i]` is None exactly where `so.prefill_complete[i]` is False. Detokenization
        for all sampled rows happens in one batched call."""
        now = time.perf_counter()
        emitted: list[tuple[Request, list[int], FinishReason | None]] = []
        for req, tok, qlen, done in zip(so.scheduled, sampled, so.query_lens, so.prefill_complete,
                                        strict=True):
            if not done:
                assert tok is None
                continue  # partial prefill chunk: K/V written, nothing to emit yet
            assert tok is not None
            if req.is_finished:
                continue  # async: aborted, or ended by its previous token, after this launch
            if isinstance(tok, list):  # speculative: verify the drafts against the model's rows
                new_tokens = self._verify_drafts(req, tok, qlen)
            else:
                new_tokens = [tok]
            reason = None
            kept: list[int] = []
            for t in new_tokens:
                req.append_output(t)
                kept.append(t)
                reason = check_stop(req, t, self.eos_token_ids, self.config.max_model_len)
                if reason is not None:
                    break
            if req.first_token_time is None:
                req.first_token_time = now
            emitted.append((req, kept, reason))
        if not emitted:
            return []
        deltas = self.detok.update_batch(
            [r.request_id for r, _, _ in emitted], [r.output_token_ids for r, _, _ in emitted],
            [r.sampling_params.stop for r, _, _ in emitted],
            finals=[reason is not None for _, _, reason in emitted])
        outputs: list[RequestOutput] = []
        for (req, toks, reason), (text_delta, matched_stop) in zip(emitted, deltas, strict=True):
            if matched_stop is not None and reason is None:
                reason = FinishReason.STOP
            if reason is not None:
                self.scheduler.finish_request(req, reason)
                self.backend.free_sequence(req.seq_id)
                self._final_text[req.request_id] = self.detok.text(req.request_id)
                self.detok.reset(req.request_id)
            outputs.append(RequestOutput(
                request_id=req.request_id, new_token_ids=toks,
                output_token_ids=list(req.output_token_ids), finished=reason is not None,
                finish_reason=reason, text_delta=text_delta,
                metrics=self._metrics(req) if reason is not None else {}))
        return outputs

    def _verify_drafts(self, req: Request, rows: list[int], qlen: int) -> list[int]:
        """`rows[j]` is the model's greedy token at query position j (input: the real last
        token for j = 0, draft j-1 after). Keep the drafts the model agreed with plus its
        own token after them; give the rejected positions' K/V slots back."""
        from pagedserve.spec import accepted_prefix

        drafts = req.draft_tokens
        assert qlen == 1 + len(drafts) == len(rows), (qlen, len(drafts), len(rows))
        a = accepted_prefix(drafts, rows)
        self.spec_drafted += len(drafts)
        self.spec_accepted += a
        # `_advance` counted all qlen positions as computed; only the first a + 1 hold
        # tokens the sequence actually has (last real token + accepted drafts).
        req.num_computed_tokens -= len(drafts) - a
        self.block_manager.truncate(req.seq_id, req.num_computed_tokens)
        req.draft_tokens = []
        return drafts[:a] + [rows[a]]

    @staticmethod
    def _metrics(req: Request) -> dict:
        m = {"num_prompt_tokens": req.num_prompt_tokens,
             "num_output_tokens": req.num_output_tokens}
        if req.first_token_time and req.arrival_time:
            m["ttft_s"] = req.first_token_time - req.arrival_time
        if req.finished_time and req.first_token_time and req.num_output_tokens > 1:
            m["tpot_s"] = (req.finished_time - req.first_token_time) / (req.num_output_tokens - 1)
        if req.finished_time and req.arrival_time:
            m["e2e_s"] = req.finished_time - req.arrival_time
        return m

    # ---- convenience -------------------------------------------------------------------
    def output_text(self, request_id: str) -> str:
        """Text so far for a live request, or the full text of a finished one (popped)."""
        if request_id in self._final_text:
            return self._final_text.pop(request_id)
        return self.detok.text(request_id)

    def reset(self) -> None:
        """Drop all requests and cache state (used between benchmark runs)."""
        self._pending = None
        if self.tp.size > 1 and self.tp.is_driver:
            tpdist.broadcast_object(("reset",))
        for rid in [r.request_id for r in list(self.scheduler.running) + list(self.scheduler.waiting)]:
            self.abort_request(rid)
        self.backend.reset()
        self.block_manager.reset()
        self.stats.clear()
