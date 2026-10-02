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
    dtype; otherwise (Qwen2.5 ships bf16, the engine runs fp16) each piece lands in one
    buffer-sized device staging tensor and is cast into its slice of the parameter on the
    GPU, in stream order. A piece that does not start and end on an element boundary (only
    possible with a buffer size that is not a multiple of the element size) or a
    non-contiguous target falls back to staging the whole tensor and casting it when its
    last piece arrives. Staging whole tensors cost a 1.24 GB device buffer for Qwen3-8B's
    embedding, which the caching allocator then kept, so the KV cache sized after the load
    came out that much smaller, and a KV cache sized *before* the load (the engine can
    capture its CUDA graphs while the weights download) would have had to leave room for it.

The result is bounded by the slower of the disk (or page cache) and PCIe instead of by one
Python thread.

Files are streamed one after another, each planned when it is opened, so the loader can
also run *while the checkpoint is still downloading* (`wait_s`, or
`EMBERSERVE_WAIT_WEIGHTS_S` through `load_hf_weights`): the expected file list comes from
`model.safetensors.index.json`, and each file is loaded as soon as it appears at its final
path (downloaders write to a temporary name and rename, so a file that exists is
complete). A Serverless worker with a small image starts the engine while the weights are
still arriving; the engine's own startup and the loading of shard i overlap the download
of the shards after it. `load_hf_weights` uses it whenever it can (one rank, safetensors files);
`EMBERSERVE_LOADER=safetensors` forces the reference path. On CPU the same plan runs with
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
    whole_staged_bytes: int = 0  # cast bytes that needed a whole-tensor staging buffer
    wait_seconds: float = 0.0  # time spent waiting for files still downloading
    files: list[str] = field(default_factory=list)

    @property
    def gb_per_s(self) -> float:
        """Read rate, excluding time spent waiting for files to arrive."""
        return self.bytes / max(self.seconds - self.wait_seconds, 1e-9) / 1e9


def plan_tensors(model: nn.Module, files: list[Path], fds: list[int],
                 seen: set[tuple] | None = None) -> list[_Tensor]:
    """Every checkpoint tensor that maps into `model`, in file order, validated the way
    `weights._load_tensors` validates: unknown keys, duplicates and shape mismatches raise
    here, before anything is read; parameters nothing loads raise after (`missing`)."""
    from emberserve.model.weights import _shard_view, hf_to_local

    config = model.config
    state = model.state_dict()
    out: list[_Tensor] = []
    seen = set() if seen is None else seen
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


def expected_files(model_dir: str | os.PathLike) -> list[Path]:
    """The checkpoint's safetensors files, in load order: the index's `weight_map` values
    when there is an index (they need not exist yet), else the files present."""
    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [d / n for n in names]
    if (d / "model.safetensors").exists() or not any(d.glob("*.safetensors")):
        return [d / "model.safetensors"]
    return sorted(d.glob("*.safetensors"))


def wait_for_file(path: Path, deadline: float, poll_s: float = 0.02) -> float:
    """Block until `path` exists; seconds waited. TimeoutError at `deadline` (monotonic)."""
    t0 = time.monotonic()
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"{path} did not appear (weights still downloading?)")
        time.sleep(poll_s)
    return time.monotonic() - t0


def stream_weights(model: nn.Module, model_dir: str | os.PathLike, device: torch.device | str,
                   threads: int | None = None, buffer_mb: float = 64,
                   wait_s: float | None = None) -> LoadStats:
    """Load `model_dir/*.safetensors` into `model` (single rank). See the module docstring.
    `wait_s`: wait up to this long in total for files that are still downloading."""
    from emberserve.model.weights import _expected_shards, _hf_name

    t0 = time.perf_counter()
    device = torch.device(device)
    cuda = device.type == "cuda"
    if wait_s is not None:
        files = expected_files(model_dir)
        deadline = time.monotonic() + wait_s
    else:
        files = sorted(Path(model_dir).glob("*.safetensors"))
        deadline = None
        if not files:
            raise FileNotFoundError(f"no *.safetensors files in {model_dir}")
    threads = threads or max(2, min(8, os.cpu_count() or 2))
    stats = LoadStats(threads=threads, buffer_mb=buffer_mb, files=[p.name for p in files])
    cap = max(1, int(buffer_mb * (1 << 20)))
    bufs: list[torch.Tensor] = []
    views: list = []
    stream = torch.cuda.Stream(device) if cuda else None
    seen: set[tuple] = set()
    loaded: set[tuple] = set()
    pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="emberserve-load")
    try:
        with torch.no_grad():
            for path in files:
                if deadline is not None:
                    stats.wait_seconds += wait_for_file(path, deadline)
                fd = os.open(str(path), os.O_RDONLY)
                try:
                    tensors = plan_tensors(model, [path], [fd], seen)
                    batches = _batches(tensors, cap)
                    nbuf = min(len(batches), 2 * threads) or 1
                    while len(bufs) < nbuf:  # the ring grows to the largest file's need
                        bufs.append(torch.empty(cap, dtype=torch.uint8, pin_memory=cuda))
                        views.append(bufs[-1].numpy())
                    _stream_file(batches, bufs[:nbuf], views[:nbuf], pool, stream, device, cuda, stats)
                    stats.bytes += sum(t.nbytes for t in tensors)
                    stats.tensors += len(tensors)
                    loaded.update(t.key for t in tensors)
                finally:
                    os.close(fd)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    config = model.config
    tied = getattr(config, "tie_word_embeddings", False)
    state = model.state_dict()
    missing = [_hf_name(k, sh, config) for k, v in state.items() for sh in _expected_shards(k, v)
               if not (tied and k == "lm_head.weight") and (k, sh) not in loaded]
    if missing:
        raise KeyError(f"parameters never loaded from {model_dir}: {missing}")
    stats.seconds = time.perf_counter() - t0
    return stats


def _stream_file(batches, bufs, views, pool, stream, device, cuda, stats) -> None:
    """Read one file's batches into the pinned ring and copy them out (the pipeline in the
    module docstring). Returns once every copy of this file has completed."""
    nbuf = len(bufs)
    whole = _needs_whole_staging(batches)
    chunk: torch.Tensor | None = None  # the per-piece cast buffer, stream-ordered reuse

    def read(i: int, wait: "torch.cuda.Event | None") -> None:
        if wait is not None:
            wait.synchronize()  # the batch that used this buffer has been copied out
        view = views[i % nbuf]
        for t, toff, n, boff in batches[i]:
            got = 0
            while got < n:  # preadv may return short on some filesystems
                k = os.preadv(t.fd, [memoryview(view[boff + got:boff + n])], t.offset + toff + got)
                if k <= 0:
                    raise OSError(f"short read in {t.name}")
                got += k

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
                    elif id(t) not in whole:
                        # One piece: bytes into the small staging buffer, then cast into the
                        # matching elements of the parameter. Both run on `stream`, so the
                        # next piece's copy into `chunk` waits for this cast.
                        if chunk is None:
                            chunk = torch.empty(len(buf), dtype=torch.uint8, device=device)
                        chunk[:n].copy_(src, non_blocking=cuda)
                        s = t.dtype.itemsize
                        t.target.view(-1)[toff // s:(toff + n) // s].copy_(chunk[:n].view(t.dtype))
                    else:
                        stats.whole_staged_bytes += n
                        if t.staging is None:
                            t.staging = torch.empty(t.nbytes, dtype=torch.uint8, device=device)
                        t.staging[toff:toff + n].copy_(src, non_blocking=cuda)
                    t.filled += n
                    if t.filled == t.nbytes and t.staging is not None:
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
    finally:  # on an exception mid-file: drop queued reads, let running ones finish
        pending = [f for f in futures.values() if not f.cancel()]
        for f in pending:
            try:
                f.result()
            except Exception:  # noqa: BLE001 - the original exception is the one raised
                pass


def _needs_whole_staging(batches) -> set[int]:
    """ids of the tensors to cast that cannot be cast piece by piece: a piece that splits an
    element, or a target whose elements are not laid out in checkpoint order."""
    out: set[int] = set()
    for batch in batches:
        for t, toff, n, _ in batch:
            if t.direct or id(t) in out:
                continue
            s = t.dtype.itemsize
            if toff % s or n % s or not t.target.is_contiguous():
                out.add(id(t))
    return out


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
