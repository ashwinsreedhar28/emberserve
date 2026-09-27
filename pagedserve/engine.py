"""LLMEngine: the step loop that ties scheduler, block manager, model, attention backend
and sampler together. One `step()` = one scheduler decision + one model forward + one
sampled token per scheduled sequence.

    engine = LLMEngine.from_pretrained("models/Qwen2.5-0.5B-Instruct")
    engine.add_request("r0", "Hello", SamplingParams.greedy(32))
    while engine.has_unfinished_requests():
        for out in engine.step():
            ...
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from pagedserve.attn.base import AttentionBackend, AttnMetadata
from pagedserve.attn.naive import NaiveAttentionBackend
from pagedserve.attn.paged_torch import (PagedTorchAttentionBackend, build_block_tables_tensor,
                                        build_slot_mapping)
from pagedserve.config import EngineConfig, ModelConfig
from pagedserve.kv.block_manager import BlockManager
from pagedserve.kv.cache import PagedKVCache
from pagedserve.model.qwen2 import Qwen2ForCausalLM
from pagedserve.model.weights import load_model
from pagedserve.sampling import Sampler, check_stop
from pagedserve.sched.request import FinishReason, Request, RequestOutput, SamplingParams
from pagedserve.sched.scheduler import Scheduler, SchedulerOutput
from pagedserve.tokenizer import IncrementalDetokenizer, Tokenizer


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


# Backends whose decode step reads only static device tensors (meta.context_lens_t /
# meta.block_tables_nonneg) and can therefore be captured into a CUDA graph.
GRAPH_CAPABLE_BACKENDS = ("paged_flash", "paged_triton")


def default_num_blocks(model_config: ModelConfig, engine_config: EngineConfig,
                       model_bytes: int) -> int:
    """How many KV blocks to allocate when the user did not say.

    CUDA: (free memory * utilization - model weights - 512 MiB workspace) / bytes per block.
    CPU/MPS: a fixed 2048 blocks (block_size 16 -> 32K cached tokens), which is ~800 MB at
    fp32 for the 0.5B model and plenty for local correctness work.
    """
    bytes_per_block = model_config.kv_bytes_per_token(engine_config.dtype) * engine_config.block_size
    if engine_config.device.startswith("cuda") and torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info(torch.device(engine_config.device))
        budget = int(free * engine_config.gpu_memory_utilization) - model_bytes - (512 << 20)
        return max(budget // bytes_per_block, 64)
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

        model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        num_blocks = engine_config.num_gpu_blocks or default_num_blocks(
            model_config, engine_config, model_bytes)
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
        # Optional CUDA-graph decode runner (attn/cuda_graphs.py); installed on GPU only.
        self.graph_runner = None
        if use_graphs:
            from pagedserve.attn.cuda_graphs import CUDAGraphRunner
            self.graph_runner = CUDAGraphRunner(
                self.model, self.backend, max_model_len=engine_config.max_model_len,
                block_size=engine_config.block_size, scratch_block=self.scratch_block,
                max_batch=engine_config.max_num_seqs, device=self.device)
            self.graph_runner.capture()
        self._next_seq_id = 0
        self._final_text: dict[str, str] = {}
        self._step_count = 0
        self.stats: list[StepStats] = []
        self.keep_stats = True

    # ---- construction -----------------------------------------------------------
    def _make_backend(self, num_blocks: int) -> AttentionBackend:
        name = self.config.attn_backend
        if name == "naive":
            return NaiveAttentionBackend(self.model_config, self.device, self.dtype)
        if name == "paged_torch":
            cache = PagedKVCache(self.model_config, num_blocks, self.config.block_size,
                                 self.device, self.dtype)
            return PagedTorchAttentionBackend(self.model_config, cache)
        if name == "paged_flash":
            from pagedserve.attn.paged_flash import PagedFlashAttentionBackend
            cache = PagedKVCache(self.model_config, num_blocks, self.config.block_size,
                                 self.device, self.dtype)
            return PagedFlashAttentionBackend(self.model_config, cache)
        if name == "paged_triton":
            from pagedserve.attn.paged_triton import PagedTritonAttentionBackend
            cache = PagedKVCache(self.model_config, num_blocks, self.config.block_size,
                                 self.device, self.dtype)
            return PagedTritonAttentionBackend(self.model_config, cache)
        raise ValueError(f"unknown attn_backend {name!r}")

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, engine_config: EngineConfig | None = None,
                        **overrides) -> "LLMEngine":
        engine_config = engine_config or EngineConfig()
        for k, v in overrides.items():
            setattr(engine_config, k, v)
        engine_config.model_dir = str(model_dir)
        model = load_model(model_dir, device=engine_config.device, dtype=engine_config.dtype)
        model_config = ModelConfig.from_hf_dir(model_dir)
        tokenizer = None
        if (Path(model_dir) / "tokenizer.json").exists():
            try:
                tokenizer = Tokenizer(model_dir)
            except ImportError:
                tokenizer = None
        return cls(model, model_config, engine_config, tokenizer)

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
        return self.scheduler.has_unfinished_requests()

    # ---- the step ---------------------------------------------------------------------
    @torch.inference_mode()
    def step(self) -> list[RequestOutput]:
        sched_out = self.scheduler.schedule()
        for req in sched_out.preempted:
            # Recompute-on-readmit: the backend must drop whatever it held for this seq.
            self.backend.free_sequence(req.seq_id)
        if sched_out.is_empty:
            return []
        self._step_count += 1

        input_ids, meta = self._build_inputs(sched_out)
        t0 = time.perf_counter()
        if self.graph_runner is not None and not sched_out.is_prefill:
            logits = self.graph_runner.run(input_ids, meta)
        else:
            hidden = self.model(input_ids, self.backend, meta)
            logits = self.model.compute_logits(hidden, meta)
        t1 = time.perf_counter()
        sampled = self.sampler.sample(logits, sched_out.scheduled)
        t2 = time.perf_counter()

        outputs = self._postprocess(sched_out, sampled)
        if self.keep_stats:
            st = self.block_manager.stats()
            self.stats.append(StepStats(
                step=self._step_count, is_prefill=sched_out.is_prefill,
                num_seqs=len(sched_out.scheduled), num_tokens=sched_out.num_tokens,
                num_preempted=len(sched_out.preempted), forward_ms=(t1 - t0) * 1e3,
                sample_ms=(t2 - t1) * 1e3, kv_utilization=st.utilization,
                num_free_blocks=st.num_free))
        return outputs

    def _build_inputs(self, so: SchedulerOutput) -> tuple[torch.Tensor, AttnMetadata]:
        reqs = so.scheduled
        seq_ids = [r.seq_id for r in reqs]
        starts = [r.num_computed_tokens for r in reqs]
        tokens: list[int] = []
        positions: list[int] = []
        context_lens: list[int] = []
        for r, start, qlen in zip(reqs, starts, so.query_lens, strict=True):
            ids = r.all_token_ids[start:start + qlen]
            assert len(ids) == qlen, (len(ids), qlen, r.request_id)
            tokens.extend(ids)
            positions.extend(range(start, start + qlen))
            context_lens.append(start + qlen)
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=self.device)
        pos = torch.tensor(positions, dtype=torch.int64, device=self.device)
        meta = AttnMetadata(
            is_prefill=so.is_prefill, seq_ids=seq_ids, query_lens=list(so.query_lens),
            context_lens=context_lens, positions=pos,
            num_cached_tokens=starts if so.is_prefill else [],
        )
        if self.config.attn_backend != "naive":
            meta.slot_mapping = build_slot_mapping(self.block_manager, seq_ids, starts,
                                                  list(so.query_lens), self.device)
            meta.block_tables = build_block_tables_tensor(
                [self.block_manager.get_block_table(s) for s in seq_ids], self.device)
            meta.block_size = self.config.block_size
        return input_ids, meta

    def _postprocess(self, so: SchedulerOutput, sampled: list[int]) -> list[RequestOutput]:
        now = time.perf_counter()
        outputs: list[RequestOutput] = []
        for req, qlen, tok in zip(so.scheduled, so.query_lens, sampled, strict=True):
            req.num_computed_tokens += qlen
            req.append_output(tok)
            if req.first_token_time is None:
                req.first_token_time = now
            reason = check_stop(req, tok, self.eos_token_id, self.config.max_model_len)
            text_delta, matched_stop = self.detok.update(
                req.request_id, req.output_token_ids, req.sampling_params.stop,
                final=reason is not None)
            if matched_stop is not None and reason is None:
                reason = FinishReason.STOP
            if reason is not None:
                self.scheduler.finish_request(req, reason)
                self.backend.free_sequence(req.seq_id)
                self._final_text[req.request_id] = self.detok.text(req.request_id)
                self.detok.reset(req.request_id)
            outputs.append(RequestOutput(
                request_id=req.request_id, new_token_ids=[tok],
                output_token_ids=list(req.output_token_ids), finished=reason is not None,
                finish_reason=reason, text_delta=text_delta,
                metrics=self._metrics(req) if reason is not None else {}))
        return outputs

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
        for rid in [r.request_id for r in list(self.scheduler.running) + list(self.scheduler.waiting)]:
            self.abort_request(rid)
        self.backend.reset()
        self.block_manager.reset()
        self.stats.clear()
