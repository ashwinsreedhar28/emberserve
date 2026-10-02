"""The per-step trace (EMBERSERVE_STEP_TRACE), the report that reads it, the engine's boot
phases, and the non-synchronizing index helper — all on the tiny CPU model."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from emberserve.devutil import index_tensor
from emberserve.steptrace import classify
from tests.test_engine import CFG, make_engine, prompts
from emberserve.sched.request import SamplingParams

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location("step_trace_report", ROOT / "scripts/step_trace_report.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step_trace_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_index_tensor_matches_torch_tensor_on_cpu() -> None:
    for vals, dt in (([3, 1, 2], torch.long), ([0, 5, 9], torch.int32), ([0.5, 1.0], torch.float32)):
        a = index_tensor(vals, dt, "cpu")
        assert a.dtype == dt and torch.equal(a, torch.tensor(vals, dtype=dt))


def test_classify_kinds() -> None:
    assert classify([1, 1, 1], graph=True, piecewise=False)[0] == "decode_graph"
    assert classify([1, 1], graph=False, piecewise=False)[0] == "decode_eager"
    assert classify([7, 3], graph=False, piecewise=False) == ("prefill", 0, 10, 2, 7)
    assert classify([1, 1, 12], graph=False, piecewise=True) == ("mixed_piecewise", 2, 12, 1, 12)


def _run_traced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, async_sched: bool) -> Path:
    path = tmp_path / f"trace_{int(async_sched)}.jsonl"
    monkeypatch.setenv("EMBERSERVE_STEP_TRACE", str(path))
    eng = make_engine("paged_torch", max_batched=8, enable_chunked_prefill=True)
    eng.async_scheduling = async_sched
    sp = SamplingParams.greedy(12, ignore_eos=True)
    for i, p in enumerate(prompts(4, seed=3)):
        eng.add_request(f"r{i}", p[:2], sp)
    eng.step()
    eng.step()
    g = torch.Generator().manual_seed(4)
    eng.add_request("long", torch.randint(2, CFG.vocab_size, (30,), generator=g).tolist(),
                    SamplingParams.greedy(4, ignore_eos=True))
    while eng.has_unfinished_requests():
        eng.step()
    eng._tracer.close()
    return path


@pytest.mark.parametrize("async_sched", [False, True])
def test_trace_records_every_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, async_sched: bool) -> None:
    path = _run_traced(tmp_path, monkeypatch, async_sched)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert "boot" in lines[0] and "kv_cache_s" in lines[0]["boot"]
    steps = lines[1:]
    assert [d["step"] for d in steps] == list(range(1, len(steps) + 1))  # none lost, in order
    kinds = {d["kind"] for d in steps}
    assert {"prefill", "mixed", "decode_eager"} <= kinds
    for d in steps:
        assert d["n_seqs"] == d["n_decode"] + d["n_prefill_seqs"]
        assert d["gpu_ms"] is None  # CPU
        assert d["host_launch_ms"] >= 0 and d["host_resolve_ms"] >= 0
    mixed = [d for d in steps if d["kind"] == "mixed"]
    assert all(d["n_decode"] == 4 and 1 <= d["n_prefill_tokens"] <= 4 for d in mixed)


def test_report_decomposes_a_cpu_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture) -> None:
    path = _run_traced(tmp_path, monkeypatch, True)
    rep = _report()
    boot, steps = rep.load(path)
    a = rep.analyse(steps)
    assert a["decode_tokens"] == sum(d["n_decode"] for d in steps)
    dcmp = a["decomposition_ms_per_token"]
    parts = sum(dcmp.values())
    assert a["engine_tpot_ms"] == pytest.approx(parts, rel=0.35)  # floor is a per-batch median
    assert a["prompt_chunks"] >= 1 and "mixed_step_excess_ms" in a
    monkeypatch.setattr(sys, "argv", ["x", str(path), "--json", str(tmp_path / "a.json")])
    rep.main()
    out = capsys.readouterr().out
    assert "engine TPOT" in out and "mixed" in out
    assert json.loads((tmp_path / "a.json").read_text())["steps"] == len(steps)


def test_report_synthetic_gpu_trace() -> None:
    """Hand-built GPU records: 8 decode steps of 10 ms at batch 32, 2 mixed steps of 25 ms
    with 5 ms idle before each. TPOT inside the engine = (8*10 + 2*30)/10 = 14 ms: floor 10,
    GPU excess 3 (15 of 25 above the floor, over 5 of 10 steps' tokens... per token 2*15/10),
    gaps 1."""
    rep = _report()
    steps = [{"kind": "decode_graph", "n_decode": 32, "n_prefill_tokens": 0, "n_prefill_seqs": 0,
              "gpu_ms": 10.0, "gpu_gap_ms": 0.0} for _ in range(8)]
    steps += [{"kind": "mixed", "n_decode": 32, "n_prefill_tokens": 270, "n_prefill_seqs": 1,
               "gpu_ms": 25.0, "gpu_gap_ms": 5.0} for _ in range(2)]
    a = rep.analyse(steps)
    assert a["engine_tpot_ms"] == pytest.approx(14.0)
    d = a["decomposition_ms_per_token"]
    assert d["decode_floor"] == pytest.approx(10.0)
    assert d["prompt_steps_gpu_excess"] == pytest.approx(3.0)
    assert d["prompt_steps_gpu_gap"] == pytest.approx(1.0)
    assert d["decode_steps_gpu_gap"] == pytest.approx(0.0)
    assert a["excess_per_prompt_chunk_ms"] == pytest.approx(20.0)
