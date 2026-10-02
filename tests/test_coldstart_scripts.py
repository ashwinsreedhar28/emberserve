"""The cold-start benchmark's log parsing, on the lines vLLM 0.30.0 and emberserve print."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("bench_coldstart", ROOT / "scripts/bench_coldstart.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["bench_coldstart"] = m
    spec.loader.exec_module(m)
    return m


def test_log_facts_vllm_and_emberserve() -> None:
    m = _mod()
    vllm = ("(EngineCore pid=5391) INFO 09-29 20:30:05 [monitor.py:53] torch.compile took 15.56 s in total\n"
            "(EngineCore pid=5391) INFO 09-29 20:31:45 [core.py:372] init engine (profile, create kv cache, "
            "warmup model) took 116.99 s (compilation: 15.56 s)\n")
    f = m.log_facts(vllm)
    assert f["vllm_init_s"] == 116.99 and f["vllm_compile_s"] == 15.56 and f["vllm_torch_compile_s"] == 15.56
    ps = "[boot] load_weights 3.10 s · weights_read 2.80 s · kv_cache 0.05 s (weights 15.23 GB in 2.80 s = 5.44 GB/s)\n"
    assert m.log_facts(ps)["emberserve_boot"].startswith("load_weights 3.10 s")


def test_commands_shape() -> None:
    import argparse

    m = _mod()
    a = argparse.Namespace(model="M", port=8000, dtype="float16", max_model_len=4096, vllm_bin="vllm",
                           emberserve_args="", runai=True)
    cmd, env = m.commands("vllm_tuned", a)
    assert cmd[:3] == ["vllm", "serve", "M"] and "--load-format" in cmd and env["HF_HUB_OFFLINE"] == "1"
    assert "--enforce-eager" in m.commands("vllm_eager", a)[0]
    assert "emberserve.cli" in " ".join(m.commands("emberserve", a)[0])
