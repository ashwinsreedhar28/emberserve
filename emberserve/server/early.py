"""Start the engine core before the API process imports anything heavy.

`emberserve serve` on CUDA used to do, in order: import torch and the server (1.6 s on the
A100 pod), load the tokenizer through `transformers` (2.45 s), and only then spawn the
engine core, which imported torch again (1.35 s), created the CUDA context (0.6 s) and
booted (~4 s: weights, graph capture). Every step waited for the one before it; a 7B took
10.3 s from process start to first token (results/coldstart/).

This module imports nothing but the standard library, so `cmd_serve` can spawn the core as
its very first act, from the raw command-line arguments; the core parses them itself. The
API process's imports and tokenizer load then run while the core boots, and the server
attaches to the core through `engine_core.AttachedCore` (the same handle the multi-worker
server uses). The core exits when the API process closes its pipe or dies.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any


def _core_main(argv: list[str], cmd_recvs: list, out_sends: list) -> None:
    from emberserve.cli import build_parser, engine_config_from_args
    from emberserve.server.engine_core import EngineSpec, _run_engine_core

    args = build_parser().parse_args(argv)
    _run_engine_core(EngineSpec(engine_config_from_args(args), model_dir=args.model), cmd_recvs, out_sends)


def spawn_core_from_argv(argv: list[str], n_workers: int = 1) -> tuple[Any, list[tuple[Any, Any]]]:
    """Spawn the engine core for `emberserve <argv>`; per worker, (command sender, output
    receiver). Same pipes and protocol as `engine_core.spawn_core`."""
    ctx = mp.get_context("spawn")
    cmd_recvs, out_sends, chans = [], [], []
    for _ in range(n_workers):
        cmd_recv, cmd_send = ctx.Pipe(duplex=False)
        out_recv, out_send = ctx.Pipe(duplex=False)
        cmd_recvs.append(cmd_recv)
        out_sends.append(out_send)
        chans.append((cmd_send, out_recv))
    proc = ctx.Process(target=_core_main, args=(list(argv), cmd_recvs, out_sends),
                       name="emberserve-engine-core", daemon=True)
    proc.start()
    for c in cmd_recvs + out_sends:
        c.close()
    return proc, chans
