"""Several API processes in front of one engine core (`emberserve serve --api-workers N`).

Why: at 0.5B the engine core steps 100-200 sequences in ~2.5 ms (60-80k tok/s of
capacity) while one API process - detokenizer, stop strings, SSE encoding and socket
writes on one Python event loop - tops out at 26-29k tok/s on the A100 pod's CPU, and the
core spends a quarter of its time blocked sending to it. The per-token work is
embarrassingly parallel across requests, so it is split across processes:

    supervisor (this module)
      |- engine core process        (one; `engine_core._run_engine_core`, N pipe pairs)
      |- API worker 0 .. N-1        (uvicorn + FastAPI + `AsyncEngineCoreClient` over an
                                     `AttachedCore`, all accepting on one listening socket
                                     the supervisor bound: whichever worker's event loop is
                                     free takes the next connection)

One shared socket rather than SO_REUSEPORT because it balances on macOS too (BSD's
SO_REUSEPORT hands every TCP connection to the last socket bound) and because a busy
worker accepts less, which is the balancing wanted here. It is uvicorn's own `--workers`
layout; uvicorn's flag cannot be used because each of its workers would build its own app
and so start its own engine core.

A request belongs to the worker whose connection carried it; the core sends each worker
only its own rows. `/metrics` on any worker reports server-wide totals (`SharedCounters`).
The supervisor holds no pipes and serves nothing: it starts the processes, waits, and
takes them down together when one dies or it is told to stop.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import socket
import sys
import time
from typing import Any

from emberserve.server.engine_core import AttachedCore, EngineSpec, spawn_core


def _listen_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(2048)
    sock.set_inheritable(True)
    return sock


def _worker_main(spec: EngineSpec, tokenizer: Any, model_name: str, sock: socket.socket,
                 chan: tuple, core_pid: int, shared: Any, index: int, log_level: str) -> None:
    import uvicorn

    from emberserve.server.app import create_app
    from emberserve.server.async_engine import AsyncEngineCoreClient

    if isinstance(tokenizer, str):  # a model dir: each worker loads its own tokenizer
        from emberserve.tokenizer import Tokenizer, has_tokenizer

        tokenizer = Tokenizer(tokenizer) if has_tokenizer(tokenizer) else None
    core = AttachedCore(chan[0], chan[1], core_pid)
    client = AsyncEngineCoreClient(spec, tokenizer, core=core, shared=(shared, index))
    app = create_app(client, model_name)
    uvicorn.Server(uvicorn.Config(app, log_level=log_level)).run(sockets=[sock])


class MultiServer:
    """Start / watch / stop the core and N API workers. `run()` blocks (the CLI);
    `start()` + `stop()` are for callers that run their own loop (tests, benchmarks)."""

    def __init__(self, spec: EngineSpec, tokenizer: Any, model_name: str, host: str, port: int,
                 n_workers: int, log_level: str = "info") -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self.spec, self.tokenizer, self.model_name = spec, tokenizer, model_name
        self.host, self.port, self.n_workers, self.log_level = host, port, n_workers, log_level
        self.core: Any = None
        self.workers: list[Any] = []
        self.bound_port = port  # the real port once started (port 0 picks one)

    def start(self) -> None:
        from emberserve.server.async_engine import SharedCounters

        ctx = mp.get_context("spawn")
        sock = _listen_socket(self.host, self.port)
        self.bound_port = sock.getsockname()[1]
        self.core, chans = spawn_core(self.spec, self.n_workers)
        shared = SharedCounters(self.n_workers, ctx)
        for i, chan in enumerate(chans):
            p = ctx.Process(target=_worker_main,
                            args=(self.spec, self.tokenizer, self.model_name, sock,
                                  chan, self.core.pid, shared, i, self.log_level),
                            name=f"emberserve-api-{i}")
            p.start()
            self.workers.append(p)
        for chan in chans:  # the workers own them now
            for c in chan:
                c.close()
        sock.close()  # each worker holds its own duplicate

    def alive(self) -> bool:
        return (self.core is not None and self.core.is_alive()
                and all(p.is_alive() for p in self.workers))

    def stop(self, timeout_s: float = 10.0) -> None:
        for p in self.workers:
            if p.is_alive():
                p.terminate()  # SIGTERM: uvicorn shuts down, the client closes its pipes
        deadline = time.monotonic() + timeout_s
        for p in self.workers:
            p.join(max(0.1, deadline - time.monotonic()))
            if p.is_alive():
                p.kill()
                p.join(5.0)
        if self.core is not None:
            self.core.join(max(0.5, deadline - time.monotonic()))  # exits once every pipe closed
            if self.core.is_alive():
                self.core.terminate()
                self.core.join(5.0)
            if self.core.is_alive():
                self.core.kill()
                self.core.join(5.0)
        self.workers = []
        self.core = None

    def run(self) -> int:
        stop = {"flag": False}

        def _on_signal(signum, frame):  # noqa: ARG001
            stop["flag"] = True

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)
        self.start()
        print(f"[serve] {self.n_workers} API workers on {self.host}:{self.bound_port}, engine core "
              f"pid {self.core.pid}, supervisor pid {os.getpid()}", file=sys.stderr, flush=True)
        code = 0
        while not stop["flag"]:
            if not self.alive():
                dead = [p.name for p in self.workers if not p.is_alive()]
                if not self.core.is_alive():
                    dead.append("engine core")
                print(f"[serve] {', '.join(dead)} exited; stopping", file=sys.stderr, flush=True)
                code = 1
                break
            time.sleep(0.5)
        self.stop()
        return code
