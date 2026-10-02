"""`emberserve serve --engine-process` spawns the engine core first, from the raw argv
(server/early.py), and attaches to it: the server serves, the CLI module imports without
torch, and stopping the server leaves no core process behind."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from tests.test_model import tiny_model
from tests.test_weights import _dump_snapshot

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_cli_imports_without_torch() -> None:
    out = subprocess.run([sys.executable, "-c",
                          "import sys, emberserve.cli, emberserve.server.early; print('torch' in sys.modules)"],
                         capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == "False"


def test_serve_with_early_core_and_clean_shutdown(tmp_path: Path) -> None:
    _dump_snapshot(tiny_model(seed=7), tmp_path / "m")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "emberserve.cli", "serve", "--model", str(tmp_path / "m"), "--engine-process",
         "--port", str(port), "--num-blocks", "256", "--block-size", "4"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 120
        while True:
            assert proc.poll() is None, proc.stdout.read().decode(errors="replace")[-2000:]
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() < deadline, "server did not come up"
            time.sleep(0.2)
        r = httpx.post(f"{base}/v1/completions", timeout=60, json={
            "model": str(tmp_path / "m"), "prompt": [5, 6, 7, 8], "max_tokens": 6, "temperature": 0,
            "ignore_eos": True})
        assert r.status_code == 200, r.text
        assert r.json()["usage"]["completion_tokens"] == 6
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            raise
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:  # the whole group (server + core) is gone
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    raise AssertionError("engine core outlived the server")


def test_checkpoint_bytes_uses_the_index_while_shards_download(tmp_path) -> None:
    """A Serverless worker fetching weights at start: the shards are not there yet, but the
    index says 16 GB, so the size-based defaults (piecewise graphs only below 4 GB) must not
    treat the checkpoint as small."""
    import json

    from emberserve.cli import SMALL_CHECKPOINT_BYTES, checkpoint_bytes

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 16_381_470_720}, "weight_map": {"a": "model-1.safetensors"}}))
    assert checkpoint_bytes(str(tmp_path)) == 16_381_470_720 > SMALL_CHECKPOINT_BYTES
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (tmp_path / "model.safetensors").write_bytes(b"x" * 10)
    assert checkpoint_bytes(str(tmp_path)) == 10
