"""The stall diagnostics: step/GC logging through the engine-core process and the report."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pagedserve.config import EngineConfig
from pagedserve.sched.request import SamplingParams
from pagedserve.server.async_engine import AsyncEngineCoreClient
from pagedserve.server.engine_core import EngineSpec
from tests.stub_tokenizer import StubTokenizer
from tests.test_engine import CFG, prompts

TINY = dict(num_hidden_layers=CFG.num_hidden_layers, num_attention_heads=CFG.num_attention_heads,
            num_key_value_heads=CFG.num_key_value_heads, hidden_size=CFG.hidden_size,
            intermediate_size=CFG.intermediate_size, vocab_size=CFG.vocab_size)


def test_step_log_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "steps.tsv"
    monkeypatch.setenv("PAGEDSERVE_STEP_LOG", str(log))
    monkeypatch.setenv("PAGEDSERVE_GC", "tune")
    ecfg = EngineConfig(device="cpu", dtype=torch.float32, block_size=4, num_gpu_blocks=256,
                        max_num_seqs=64, max_num_batched_tokens=512, max_model_len=256)
    spec = EngineSpec(ecfg, tiny=True, tiny_seed=0, tiny_overrides=TINY)

    async def run():
        c = AsyncEngineCoreClient(spec, StubTokenizer())
        c.start()
        try:
            outs = await asyncio.gather(*(
                _collect(c, f"r{i}", p, SamplingParams.greedy(6, ignore_eos=True))
                for i, p in enumerate(prompts(4, seed=2))))
        finally:
            c.stop()
        return outs

    outs = asyncio.run(run())
    assert all(o[-1].finished for o in outs)
    rows = log.read_text().splitlines()
    assert rows[0].split("\t") == ["t_start", "duration_s", "num_seqs", "num_tokens"]
    assert len(rows) >= 7  # a prefill step + 5 decode steps at least
    assert sum(int(r.split("\t")[3]) for r in rows[1:]) == 4 * 6
    # the GC pause log of the core exists (gc.callbacks were installed there)
    assert Path(f"{log}.gc-core").exists()
    rep = subprocess.run([sys.executable, "scripts/stall_report.py", str(log), "--threshold-ms", "0.01"],
                         capture_output=True, text=True, check=True).stdout
    assert "steps over" in rep and "step duration ms" in rep and "stalls over" in rep


async def _collect(c, rid, prompt, sp):
    return [out async for out in c.generate(rid, prompt, sp)]


def test_tune_gc_is_idempotent_and_freezes() -> None:
    import gc

    from pagedserve import diag

    before = gc.get_threshold()
    try:
        diag.tune_gc()
        assert gc.get_threshold() == (50_000, 20, 25)
        assert gc.get_freeze_count() > 0
        diag.tune_gc()
    finally:
        gc.set_threshold(*before)
        gc.unfreeze()
