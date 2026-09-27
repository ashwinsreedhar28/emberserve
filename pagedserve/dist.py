"""Tensor parallelism: one process per GPU, Megatron-style sharding of the dense block.

Each decoder layer is cut along the dimension that needs no communication *inside* the
block: attention heads (q/k/v rows of `qkv_proj` are column-parallel, each rank owns
`num_heads / tp` query heads and `num_kv_heads / tp` KV heads and the matching slice of
the paged KV cache; `o_proj` is row-parallel) and the MLP's intermediate width
(`gate_up_proj` column-parallel, `down_proj` row-parallel). The two row-parallel
projections each end in one all-reduce, so a layer costs two all-reduces of
`[num_tokens, hidden]` and nothing else; embeddings and norms are replicated, and the
`lm_head` is vocabulary-parallel with one all-gather of the logits at the end. Every rank
therefore holds `1/tp` of the weights and of the KV cache, and the batch-1 decode step,
which is the weight read, gets `1/tp` of the bytes to stream.

Process model (vLLM's driver + workers): rank 0 is the engine (scheduler, block manager,
sampler, the API's process); ranks 1.. run `LLMEngine.worker_loop`, which has no scheduler
of its own. Per step the driver broadcasts the step plan (the host-side lists
`_plan_inputs` produces: tokens, positions, slots, block tables...) over a gloo group,
every rank materializes the same device tensors from it, and the driver's `input_ids`
(which under async scheduling holds tokens gathered on the device from the previous
step's logits, so no rank but the driver could build them) go out over the device group;
then all ranks run the identical forward, collectives included, and only the driver
samples. The host-side broadcast never touches the device, so async scheduling's overlap
of one step's CPU work with the previous step's GPU work is kept. CUDA graphs capture the
collectives with the rest of the forward (NCCL in graph capture, torch >= 2.2).

Workers are `subprocess.Popen(python -m pagedserve.dist ...)` rather than
`multiprocessing` children because the engine core is itself a daemonic process. A worker
that loses its parent exits on its own (a watchdog thread), so a crashed driver never
leaves a GPU's memory held.
"""

from __future__ import annotations

import argparse
import base64
import os
import pickle
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class TPState:
    rank: int = 0
    world_size: int = 1

    @property
    def size(self) -> int:
        return self.world_size

    @property
    def is_driver(self) -> bool:
        return self.rank == 0


_STATE = TPState()
_DEVICE_GROUP: Any = None  # NCCL on CUDA (gloo on CPU): the collectives inside the forward
_CPU_GROUP: Any = None  # gloo: the per-step plan and control messages (never touches the device)
# The driver's idle time between requests is unbounded, so a blocked worker must never time out.
_TIMEOUT = timedelta(days=30)


def get_tp() -> TPState:
    return _STATE


def is_initialized() -> bool:
    return _STATE.world_size > 1


def init_tp(rank: int, world_size: int, init_method: str, device: torch.device | str) -> TPState:
    """Join the tensor-parallel group as `rank`. `device` picks the collective backend:
    NCCL on CUDA (each rank owns its own device), gloo otherwise (the CPU tests)."""
    global _STATE, _DEVICE_GROUP, _CPU_GROUP
    if world_size <= 1:
        _STATE = TPState()
        return _STATE
    device = torch.device(device)
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world_size,
                            timeout=_TIMEOUT)
    _DEVICE_GROUP = dist.group.WORLD
    _CPU_GROUP = dist.new_group(backend="gloo", timeout=_TIMEOUT) if backend != "gloo" else _DEVICE_GROUP
    _STATE = TPState(rank, world_size)
    return _STATE


def destroy_tp() -> None:
    global _STATE, _DEVICE_GROUP, _CPU_GROUP
    if dist.is_initialized():
        dist.destroy_process_group()
    _STATE = TPState()
    _DEVICE_GROUP = _CPU_GROUP = None


# torch 2.9 renamed all_gather_into_tensor (and deprecates the old name); the pod runs 2.8.
_all_gather = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor


# ---- collectives (no-ops at world size 1, so the model code calls them unconditionally) --------
def all_reduce(x: Tensor) -> Tensor:
    """Sum `x` across ranks, in place; the row-parallel projections' epilogue."""
    if _STATE.world_size > 1:
        dist.all_reduce(x, group=_DEVICE_GROUP)
    return x


def all_gather_cols(x: Tensor) -> Tensor:
    """`[n, c]` per rank -> `[n, world * c]`, rank-major along the last dim (the
    vocabulary-parallel lm_head's logits)."""
    if _STATE.world_size == 1:
        return x
    x = x.contiguous()
    w = _STATE.world_size
    out = torch.empty((w * x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    _all_gather(out, x, group=_DEVICE_GROUP)  # rank-major along dim 0
    return out.view((w,) + tuple(x.shape)).movedim(0, -2).reshape(*x.shape[:-1], w * x.shape[-1])


def broadcast_tensor(x: Tensor, src: int = 0) -> Tensor:
    """In place, over the device group (stream-ordered on CUDA: no host sync)."""
    if _STATE.world_size > 1:
        dist.broadcast(x, src=src, group=_DEVICE_GROUP)
    return x


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Pickle `obj` on `src` and hand it to every rank, over the CPU group."""
    if _STATE.world_size == 1:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src, group=_CPU_GROUP)
    return box[0]


def all_reduce_min(value: int) -> int:
    """The smallest of every rank's `value` (KV block counts: each GPU measures its own
    free memory, the cache must be the same size everywhere)."""
    if _STATE.world_size == 1:
        return value
    t = torch.tensor([value], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=_CPU_GROUP)
    return int(t.item())


def barrier() -> None:
    if _STATE.world_size > 1:
        dist.barrier(group=_CPU_GROUP)


# ---- checkpoint sharding -------------------------------------------------------------------------
# HF parameter name suffix -> how a rank's slice is cut out of the full tensor.
_ROW_PARALLEL_INPUT = ("self_attn.o_proj.weight", "mlp.down_proj.weight")  # split columns (dim 1)
_COLUMN_PARALLEL = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "mlp.gate_proj", "mlp.up_proj", "lm_head")  # split rows (dim 0), weight and bias


def shard_tensor(hf_name: str, full: Tensor, rank: int, world_size: int) -> Tensor:
    """The slice of checkpoint tensor `hf_name` that `rank` holds (a view of `full`).
    Anything not column- or row-parallel (embeddings, norms) is replicated."""
    if world_size == 1:
        return full
    head = hf_name.rpartition(".")[0]
    if hf_name.endswith(_ROW_PARALLEL_INPUT):
        return full.chunk(world_size, dim=1)[rank]
    if head.endswith(_COLUMN_PARALLEL):
        return full.chunk(world_size, dim=0)[rank]
    return full


def vocab_shard(vocab_size: int, world_size: int) -> int | None:
    """Rows of the lm_head one rank holds, or None when the vocabulary does not split evenly
    (then the lm_head is replicated and no logits gather happens)."""
    if world_size > 1 and vocab_size % world_size == 0:
        return vocab_size // world_size
    return None


# ---- worker processes -----------------------------------------------------------------------
@dataclass(frozen=True)
class WorkerSpec:
    """Everything a rank needs to build its shard of the engine: the snapshot (or the seeded
    tiny model the tests use) and the engine config. Mirrors `engine_core.EngineSpec`."""

    engine_config: Any  # EngineConfig (typed loosely: config.py must not import this module)
    model_dir: str | None = None
    tiny: bool = False
    tiny_seed: int = 0
    tiny_overrides: dict = field(default_factory=dict)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rank_device(device: str, rank: int) -> str:
    """Rank r of a CUDA group owns `cuda:r` (plus an explicit base offset if one was given)."""
    if not device.startswith("cuda"):
        return device
    base = int(device.split(":")[1]) if ":" in device else 0
    return f"cuda:{base + rank}"


def spawn_workers(spec: WorkerSpec, world_size: int, init_method: str) -> list[subprocess.Popen]:
    """Start ranks 1..world_size-1 as `python -m pagedserve.dist` processes."""
    blob = base64.b64encode(pickle.dumps(spec)).decode()
    procs = []
    log_dir = os.environ.get("PAGEDSERVE_TP_LOG_DIR")  # per-rank worker logs, else inherited stderr
    for rank in range(1, world_size):
        cmd = [sys.executable, "-m", "pagedserve.dist", "--rank", str(rank), "--world-size",
               str(world_size), "--init-method", init_method, "--spec", blob]
        out = open(os.path.join(log_dir, f"tp_worker_{rank}.log"), "w") if log_dir else None
        procs.append(subprocess.Popen(cmd, env=os.environ.copy(), stdout=out, stderr=out))
    return procs


def stop_workers(procs: list[subprocess.Popen], timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    for p in procs:
        try:
            p.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(5.0)


def _parent_watchdog(parent: int) -> None:
    """Exit when the driver process is gone; a worker blocked in a broadcast would otherwise
    wait for it until the group timeout, holding its GPU."""
    while True:
        time.sleep(1.0)
        if os.getppid() != parent:
            os._exit(0)


def build_tp_model(spec: WorkerSpec, device: str, dtype: torch.dtype):
    """This rank's shard of the model, weights loaded: `(model, full ModelConfig)`."""
    from pagedserve.config import ModelConfig
    from pagedserve.model import weights as W

    tp = get_tp()
    if spec.tiny:
        from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic

        full_cfg = ModelConfig.tiny(**spec.tiny_overrides)
        full = Qwen2ForCausalLM(full_cfg, tp_size=1)
        reset_parameters_deterministic(full, spec.tiny_seed)
        if tp.size == 1:
            return full.to(device, dtype).eval(), full_cfg
        with torch.device(device):
            model = Qwen2ForCausalLM(full_cfg.shard(tp.size))
        W.load_hf_state_dict(model, W.hf_state_dict(full), dtype=dtype, device=device)
        return model.eval(), full_cfg
    if spec.model_dir is None:
        raise ValueError("WorkerSpec needs model_dir or tiny=True")
    model = W.load_model(spec.model_dir, device=device, dtype=dtype)
    if spec.engine_config.quantization:
        from pagedserve.model.quant import quantize_model

        quantize_model(model, spec.engine_config.quantization)
    return model, ModelConfig.from_hf_dir(spec.model_dir)


def worker_main(argv: list[str] | None = None) -> None:
    from pagedserve.engine import LLMEngine

    ap = argparse.ArgumentParser(description="pagedserve tensor-parallel worker (rank > 0)")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--init-method", required=True)
    ap.add_argument("--spec", required=True, help="base64 pickle of a WorkerSpec")
    args = ap.parse_args(argv)
    spec: WorkerSpec = pickle.loads(base64.b64decode(args.spec))
    threading.Thread(target=_parent_watchdog, args=(os.getppid(),), daemon=True).start()
    ecfg = spec.engine_config
    ecfg.device = rank_device(ecfg.device, args.rank)
    init_tp(args.rank, args.world_size, args.init_method, ecfg.device)
    try:
        model, full_cfg = build_tp_model(spec, ecfg.device, ecfg.dtype)
        engine = LLMEngine(model, full_cfg, ecfg, tokenizer=None)
        engine.worker_loop()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        destroy_tp()


if __name__ == "__main__":
    # `python -m pagedserve.dist` runs this file as `__main__`; the model imports the real
    # `pagedserve.dist`, whose state must be the one `init_tp` sets, so delegate to it.
    from pagedserve.dist import worker_main as _main

    _main()
