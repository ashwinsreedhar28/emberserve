"""Stream safetensors checkpoints straight into the model's parameters.

Why: the 7B's boot on an A100 pod was ~33 s, 29 s of it in `load_hf_weights` — 15 GB at
~0.5 GB/s, and the same with a warm page cache (five server starts in one pod, 28.3-29.0 s
each), so the time was not the disk. The reference loader does, per tensor and one at a
time on one thread: `safe_open(...).get_tensor` (a copy out of the mmap into a pageable CPU
tensor), then `.to(cuda, dtype)` (a synchronous copy from pageable memory, staged by the
driver), then `copy_` into the parameter. Nothing overlaps.

This loader reads the files itself:

  * the safetensors header (8-byte length + JSON) says where every tensor's bytes are, so
    the plan - which bytes go to which parameter slice (fused qkv / gate_up rows, stacked
    experts), with every shape and name checked - is made before a byte of data is read;
  * reader threads `preadv` fixed-size batches of those bytes into a ring of pinned host
    buffers (the syscall releases the GIL, so the reads run in parallel);
  * the main thread issues one non-blocking host->device copy per piece on a side stream
    and records an event per batch; a buffer is refilled only after its event has passed,
    so reading, the PCIe transfer and the next reads overlap;
  * bytes land directly in the parameter when the checkpoint dtype is the parameter's
    dtype; otherwise (Qwen2.5 ships bf16, the engine runs fp16) they land in a device
    staging tensor that is cast into the parameter on the GPU when its last piece arrives.

The result is bounded by the slower of the disk (or page cache) and PCIe instead of by one
Python thread. `load_hf_weights` uses it whenever it can (one rank, safetensors files);
`PAGEDSERVE_LOADER=safetensors` forces the reference path. On CPU the same plan runs with
plain buffers and synchronous copies (the tests compare both loaders there).
"""

from __future__ import annotations

import json
import os
import struct
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
if hasattr(torch, "float8_e4m3fn"):
    _DTYPES["F8_E4M3"] = torch.float8_e4m3fn
    _DTYPES["F8_E5M2"] = torch.float8_e5m2


def read_header(path: str | os.PathLike) -> tuple[dict, int]:
    """(tensor entries, byte offset where the data section starts) of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


@dataclass
class _Tensor:
    name: str
    fd: int
    offset: int  # absolute byte offset in its file
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    target: torch.Tensor  # the parameter (slice) it loads into
    direct: bool  # bytes can land in `target` as they are
    key: tuple = ()  # (state_dict key, shard) it fills
    staging: torch.Tensor | None = None
    filled: int = 0


@dataclass
class LoadStats:
    bytes: int = 0
    tensors: int = 0
    seconds: float = 0.0
    threads: int = 0
    buffer_mb: float = 0
    direct_bytes: int = 0
    files: list[str] = field(default_factory=list)

    @property
    def gb_per_s(self) -> float:
        return self.bytes / max(self.seconds, 1e-9) / 1e9


def plan_tensors(model: nn.Module, files: list[Path], fds: list[int]) -> list[_Tensor]:
    """Every checkpoint tensor that maps into `model`, in file order, validated the way
    `weights._load_tensors` validates: unknown keys, duplicates and shape mismatches raise
    here, before anything is read; parameters nothing loads raise after (`missing`)."""
    from pagedserve.model.weights import _shard_view, hf_to_local

    config = model.config
    state = model.state_dict()
    out: list[_Tensor] = []
    seen: set[tuple] = set()
    for path, fd in zip(files, fds):
        header, data_start = read_header(path)
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            m = hf_to_local(name, getattr(config, "model_type", "qwen2"))
            if m is None:
                continue
            local, shard = m
            if local not in state:
                raise KeyError(f"unexpected checkpoint key {name!r} in {path.name}")
            if (local, shard) in seen:
                raise KeyError(f"duplicate checkpoint key {name!r} in {path.name}")
            seen.add((local, shard))
            if info["dtype"] not in _DTYPES:
                raise ValueError(f"{name!r}: unsupported safetensors dtype {info['dtype']}")
            dtype = _DTYPES[info["dtype"]]
            shape = tuple(info["shape"])
            begin, end = info["data_offsets"]
            target = _shard_view(config, state[local], shard)
            if shape != tuple(target.shape):
                raise ValueError(f"shape mismatch for {name!r}: checkpoint {shape} vs model "
                                 f"{tuple(target.shape)}")
            itemsize = torch.empty((), dtype=dtype).element_size()
            numel = 1
            for d in shape:
                numel *= d
            if end - begin != numel * itemsize:
                raise ValueError(f"{name!r}: {end - begin} bytes on disk for {numel} x {itemsize}")
            out.append(_Tensor(name, fd, data_start + begin, end - begin, dtype, shape, target,
                               direct=dtype == target.dtype and target.is_contiguous(),
                               key=(local, shard)))
    return out


def _batches(tensors: list[_Tensor], cap: int) -> list[list[tuple[_Tensor, int, int, int]]]:
    """Pack (tensor, offset in tensor, length, offset in buffer) pieces into batches of at
    most `cap` bytes, splitting tensors larger than a buffer."""
    batches: list[list[tuple[_Tensor, int, int, int]]] = []
    cur: list[tuple[_Tensor, int, int, int]] = []
    used = 0
    for t in tensors:
        done = 0
        while done < t.nbytes:
            if used == cap:
                batches.append(cur)
                cur, used = [], 0
            n = min(t.nbytes - done, cap - used)
            cur.append((t, done, n, used))
            used += n
            done += n
    if cur:
        batches.append(cur)
    return batches


def stream_weights(model: nn.Module, model_dir: str | os.PathLike, device: torch.device | str,
                   threads: int | None = None, buffer_mb: float = 64) -> LoadStats:
    """Load `model_dir/*.safetensors` into `model` (single rank). See the module docstring."""
    from pagedserve.model.weights import _expected_shards, _hf_name

    t0 = time.perf_counter()
    device = torch.device(device)
    cuda = device.type == "cuda"
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {model_dir}")
    threads = threads or max(2, min(8, os.cpu_count() or 2))
    fds = [os.open(str(p), os.O_RDONLY) for p in files]
    stats = LoadStats(threads=threads, buffer_mb=buffer_mb, files=[p.name for p in files])
    try:
        with torch.no_grad():
            tensors = plan_tensors(model, files, fds)
            cap = max(1, int(buffer_mb * (1 << 20)))
            batches = _batches(tensors, cap)
            nbuf = min(len(batches), 2 * threads) or 1
            bufs = [torch.empty(cap, dtype=torch.uint8, pin_memory=cuda) for _ in range(nbuf)]
            views = [b.numpy() for b in bufs]
            stream = torch.cuda.Stream(device) if cuda else None

            def read(i: int, wait: "torch.cuda.Event | None") -> None:
                if wait is not None:
                    wait.synchronize()  # the batch that used this buffer has been copied out
                view = views[i % nbuf]
                for t, _toff, n, boff in batches[i]:
                    got = 0
                    while got < n:  # preadv may return short on some filesystems
                        k = os.preadv(t.fd, [memoryview(view[boff + got:boff + n])],
                                      t.offset + _toff + got)
                        if k <= 0:
                            raise OSError(f"short read in {t.name}")
                        got += k

            pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="pagedserve-load")
            futures: dict[int, Future] = {}
            for i in range(min(nbuf, len(batches))):
                futures[i] = pool.submit(read, i, None)
            try:
                ctx = torch.cuda.stream(stream) if cuda else _Null()
                with ctx:
                    for i, batch in enumerate(batches):
                        futures.pop(i).result()
                        buf = bufs[i % nbuf]
                        for t, toff, n, boff in batch:
                            src = buf[boff:boff + n]
                            if t.direct:
                                dst = t.target.view(-1).view(torch.uint8)[toff:toff + n]
                                dst.copy_(src, non_blocking=cuda)
                                stats.direct_bytes += n
                            else:
                                if t.staging is None:
                                    t.staging = torch.empty(t.nbytes, dtype=torch.uint8, device=device)
                                t.staging[toff:toff + n].copy_(src, non_blocking=cuda)
                            t.filled += n
                            if t.filled == t.nbytes and not t.direct:
                                t.target.copy_(t.staging.view(t.dtype).view(t.shape))
                                t.staging = None
                        ev = None
                        if cuda:
                            ev = torch.cuda.Event()
                            ev.record(stream)
                        nxt = i + nbuf
                        if nxt < len(batches):
                            futures[nxt] = pool.submit(read, nxt, ev)
                if cuda:
                    stream.synchronize()
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
            stats.bytes = sum(t.nbytes for t in tensors)
            stats.tensors = len(tensors)
            loaded = {t.key for t in tensors}
    finally:
        for fd in fds:
            os.close(fd)

    config = model.config
    tied = getattr(config, "tie_word_embeddings", False)
    state = model.state_dict()
    missing = [_hf_name(k, sh, config) for k, v in state.items() for sh in _expected_shards(k, v)
               if not (tied and k == "lm_head.weight") and (k, sh) not in loaded]
    if missing:
        raise KeyError(f"parameters never loaded from {model_dir}: {missing}")
    stats.seconds = time.perf_counter() - t0
    return stats


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
