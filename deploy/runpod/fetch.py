"""Fetch a checkpoint at cold start while the engine is already starting.

A worker image with the weights baked in boots fast on a host that has the image and
slowly on one that doesn't: Runpod pulled a 27 GB image at ~85 MB/s, while Hugging Face
served the same 16 GB of weights at ~760 MB/s. So the small image fetches the weights at
start, and does it without making the engine wait for the whole download:

  1. the small files (config, tokenizer, the safetensors index) come first, synchronously
     (a second or two), because both the API process and the engine read them at startup;
  2. the safetensors shards then download in the background, several at a time, each into
     a staging directory and renamed into place when complete, so a shard that exists at
     its final path is whole;
  3. meanwhile `emberserve serve` starts with EMBERSERVE_WAIT_WEIGHTS_S set, and its
     streaming loader (emberserve/model/fastload.py) loads each shard the moment it
     appears. The engine's own startup (imports, CUDA context, building the model) and the
     loading of shard i overlap the download of the shards after it.

Hugging Face's Xet backend does the transfer; HF_XET_HIGH_PERFORMANCE=1 (set in the image)
lets it use more connections. `HF_TOKEN`, if set on the endpoint, raises the rate limits.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import timeline

SMALL_PATTERNS = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.py", "merges.txt", "vocab.json"]


def _default_download(repo: str, filename: str, local_dir: str, revision: str | None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, local_dir=local_dir, revision=revision)


def _default_list(repo: str, revision: str | None) -> list[str]:
    from huggingface_hub import list_repo_files

    return list_repo_files(repo, revision=revision)


class Fetcher:
    """Download `repo` into `model_dir`: small files now, shards in the background."""

    def __init__(self, repo: str, model_dir: str | os.PathLike, revision: str | None = None,
                 workers: int = 8,
                 download: Callable[[str, str, str, str | None], str] = _default_download,
                 list_files: Callable[[str, str | None], list[str]] = _default_list) -> None:
        self.repo, self.dir, self.revision = repo, Path(model_dir), revision
        self.workers = workers
        self._download, self._list = download, list_files
        self.error: BaseException | None = None
        self.bytes = 0
        self.done = threading.Event()
        self._thread: threading.Thread | None = None

    def _get(self, name: str, staging: Path) -> Path:
        """One file: download into `staging`, then rename into place (atomic on one fs)."""
        got = Path(self._download(self.repo, name, str(staging), self.revision))
        dst = self.dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, dst)
        return dst

    def fetch_small(self) -> list[str]:
        """Everything except the shards, synchronously. Returns the shard names."""
        self.dir.mkdir(parents=True, exist_ok=True)
        staging = self.dir / ".incoming"
        names = self._list(self.repo, self.revision)
        shards = sorted(n for n in names if n.endswith(".safetensors"))
        from fnmatch import fnmatch

        small = [n for n in names if not n.endswith(".safetensors")
                 and any(fnmatch(Path(n).name, p) for p in SMALL_PATTERNS)]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for p in pool.map(lambda n: self._get(n, staging), small):
                self.bytes += p.stat().st_size
        timeline.mark("weights_small_done")
        return shards

    def start_shards(self, shards: list[str]) -> None:
        """Download the shards in the background, in index order, `workers` at a time."""
        staging = self.dir / ".incoming"
        t0 = time.monotonic()

        def run() -> None:
            try:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    for p in pool.map(lambda n: self._get(n, staging), shards):
                        self.bytes += p.stat().st_size
                secs = time.monotonic() - t0
                timeline.mark("weights_downloaded")
                timeline.note("weights_downloaded",
                              f"{self.bytes / 1e9:.2f} GB in {secs:.1f} s = {self.bytes / 1e9 / max(secs, 1e-9):.2f} GB/s "
                              f"({len(shards)} shards, {self.workers} at a time)")
                print(f"[worker] weights downloaded: {timeline.snapshot()['notes']['weights_downloaded']}",
                      flush=True)
            except BaseException as exc:  # noqa: BLE001 - reported, and the engine times out
                self.error = exc
                print(f"[worker] weight download failed: {type(exc).__name__}: {exc}", flush=True)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
                self.done.set()

        self._thread = threading.Thread(target=run, name="weights-download", daemon=True)
        self._thread.start()
