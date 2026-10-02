"""`LLMEngine.from_pretrained` building the engine (KV cache, CUDA graphs) before the weights
load (`graphs_before_weights`), on CPU: when it applies, that it gives the same engine and
output as the usual order, that it really runs before the shards exist when they are still
downloading, and the guard that captures the graphs again if a parameter's storage moved.
The GPU side (graphs captured on the empty model reproduce the usual order exactly) is in
test_graphs_before_weights_gpu.py."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pytest
import torch

from emberserve import engine as engine_mod
from emberserve.config import EngineConfig
from emberserve.engine import GRAPHS_BEFORE_WEIGHTS_ENV, LLMEngine, graphs_before_weights
from emberserve.llm import LLM
from emberserve.sched.request import SamplingParams
from tests.test_fastload import _hold_back_shards
from tests.test_model import tiny_model
from tests.test_weights import _dump_snapshot

WAIT = "EMBERSERVE_WAIT_WEIGHTS_S"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GRAPHS_BEFORE_WEIGHTS_ENV, raising=False)
    monkeypatch.delenv(WAIT, raising=False)


def _cfg(model_dir: Path | None = None, **kw) -> EngineConfig:
    c = EngineConfig(num_gpu_blocks=256, max_model_len=256, **kw)
    c.model_dir = str(model_dir) if model_dir else None
    return c


def _gen(eng: LLMEngine) -> list[list[int]]:
    g = torch.Generator().manual_seed(7)
    ps = [torch.randint(2, eng.model_config.vocab_size, (n,), generator=g).tolist() for n in (3, 11, 25)]
    return [r.output_token_ids for r in
            LLM.from_engine(eng).generate(ps, SamplingParams.greedy(12, ignore_eos=True))]


def test_when_it_applies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _dump_snapshot(tiny_model(seed=1), tmp_path)
    assert not graphs_before_weights(_cfg(tmp_path))  # weights on disk: the usual order
    monkeypatch.setenv(WAIT, "600")  # a worker still downloading: on by default
    assert graphs_before_weights(_cfg(tmp_path))
    assert not graphs_before_weights(_cfg(tmp_path, quantization="int8"))
    assert not graphs_before_weights(_cfg(tmp_path, tensor_parallel_size=2))
    assert not graphs_before_weights(_cfg(None))
    monkeypatch.setenv(GRAPHS_BEFORE_WEIGHTS_ENV, "0")
    assert not graphs_before_weights(_cfg(tmp_path))
    monkeypatch.delenv(WAIT)
    monkeypatch.setenv(GRAPHS_BEFORE_WEIGHTS_ENV, "1")  # forced, e.g. for the golden gate
    assert graphs_before_weights(_cfg(tmp_path))


def test_not_for_latent_attention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading a DeepSeek model folds a rope permutation into the weights and flips the
    flag its forward branches on; a graph captured before would record the other path."""
    _dump_snapshot(tiny_model(seed=1), tmp_path)
    monkeypatch.setenv(GRAPHS_BEFORE_WEIGHTS_ENV, "1")
    from emberserve.config import ModelConfig

    monkeypatch.setattr(ModelConfig, "from_hf_dir",
                        classmethod(lambda cls, d: type("C", (), {"mla": object()})()))
    assert not graphs_before_weights(_cfg(tmp_path))


@pytest.mark.parametrize("backend", ["paged_torch", "naive"])
def test_same_engine_and_output_as_the_usual_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   backend: str) -> None:
    _dump_snapshot(tiny_model(seed=2), tmp_path, split=True)
    usual = LLMEngine.from_pretrained(tmp_path, _cfg(attn_backend=backend), load_tokenizer=False)
    monkeypatch.setenv(GRAPHS_BEFORE_WEIGHTS_ENV, "1")
    first = LLMEngine.from_pretrained(tmp_path, _cfg(attn_backend=backend), load_tokenizer=False)
    su, sf = usual.model.state_dict(), first.model.state_dict()
    assert set(su) == set(sf) and all(torch.equal(su[k], sf[k]) for k in su)
    assert first.config.num_gpu_blocks == usual.config.num_gpu_blocks
    assert _gen(first) == _gen(usual)
    phases = list(first.boot_phases)
    assert phases[0] == "build_model_s" and phases.index("kv_cache_s") < phases.index("load_weights_s")
    assert "engine built before the weights" in first.boot_notes
    assert "before the weights" not in usual.boot_notes


def test_engine_is_built_while_the_shards_are_still_downloading(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _dump_snapshot(tiny_model(seed=3), src, split=True)
    dst = tmp_path / "dst"
    shards = _hold_back_shards(src, dst)
    monkeypatch.setenv(WAIT, "10")  # the worker's setting; graphs-first is then the default
    shards_at_init: list[int] = []
    real_init = LLMEngine.__init__

    def init(self, *a, **kw):
        shards_at_init.append(sum((dst / s.name).exists() for s in shards))
        real_init(self, *a, **kw)

    monkeypatch.setattr(LLMEngine, "__init__", init)
    released = threading.Event()

    def download() -> None:
        while not shards_at_init:  # hold the shards until the engine has been constructed
            time.sleep(0.01)
        time.sleep(0.1)
        for sh in shards:
            tmp = dst / (sh.name + ".incomplete")
            shutil.copy(sh, tmp)
            tmp.rename(dst / sh.name)
        released.set()

    t = threading.Thread(target=download)
    t.start()
    eng = LLMEngine.from_pretrained(dst, _cfg(), load_tokenizer=False)
    t.join()
    assert shards_at_init == [0] and released.is_set()
    assert "waiting for the download" in eng.boot_notes
    monkeypatch.setattr(LLMEngine, "__init__", real_init)
    monkeypatch.delenv(WAIT)
    ref = LLMEngine.from_pretrained(src, _cfg(), load_tokenizer=False)
    assert _gen(eng) == _gen(ref)


class _StubRunner:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def release(self) -> None:
        self.log.append("release")


def test_graphs_captured_again_if_a_parameter_moved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A loader that rebinds a parameter instead of copying into it would leave the graphs
    pointing at the old storage; the addresses are compared and the graphs re-captured."""
    from emberserve.model import weights as W

    _dump_snapshot(tiny_model(seed=4), tmp_path)
    monkeypatch.setenv(GRAPHS_BEFORE_WEIGHTS_ENV, "1")
    log: list[str] = []
    real_init = LLMEngine.__init__

    def init(self, *a, **kw):  # pretend the constructor captured graphs (CPU has none)
        real_init(self, *a, **kw)
        self.graph_runner = _StubRunner(log)

    def recapture(self) -> None:
        log.append("capture")
        self.graph_runner = _StubRunner(log)

    real_load = W.load_weights_into

    def moving_load(model, *a, **kw):
        out = real_load(model, *a, **kw)
        p = model.model.norm.weight
        p.data = p.data.clone()  # new storage
        return out

    monkeypatch.setattr(LLMEngine, "__init__", init)
    monkeypatch.setattr(LLMEngine, "_capture_graphs", recapture)
    monkeypatch.setattr(W, "load_weights_into", moving_load)
    with pytest.warns(UserWarning, match="capturing the CUDA graphs again"):
        eng = LLMEngine.from_pretrained(tmp_path, _cfg(), load_tokenizer=False)
    assert log == ["release", "capture"] and "graphs captured again" in eng.boot_notes

    log.clear()
    monkeypatch.setattr(W, "load_weights_into", real_load)
    eng = LLMEngine.from_pretrained(tmp_path, _cfg(), load_tokenizer=False)
    assert log == [] and "again" not in eng.boot_notes


def test_addresses_cover_parameters_and_buffers() -> None:
    m = tiny_model(seed=5)
    a = engine_mod._tensor_addresses(m)
    assert any(k.startswith("p:") for k in a)
    m.model.norm.weight.data = m.model.norm.weight.data.clone()
    assert engine_mod._tensor_addresses(m) != a
