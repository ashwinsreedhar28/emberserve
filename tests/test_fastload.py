"""The streaming safetensors loader (model/fastload.py) against the reference loader, on
CPU: identical parameters for fp32 and bf16 checkpoints, split files, and buffers so small
that tensors straddle buffers and the ring wraps many times; the same errors for bad keys,
shapes and missing parameters."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pagedserve.model.fastload import read_header, stream_weights
from pagedserve.model.qwen2 import Qwen2ForCausalLM
from pagedserve.model.weights import load_model
from tests.test_weights import _dump_snapshot
from tests.test_model import tiny_model


def _reference(tmp: Path, monkeypatch: pytest.MonkeyPatch, dtype=torch.float32) -> Qwen2ForCausalLM:
    monkeypatch.setenv("PAGEDSERVE_LOADER", "safetensors")
    m = load_model(tmp, dtype=dtype)
    monkeypatch.delenv("PAGEDSERVE_LOADER")
    return m


def _fresh(tmp: Path, dtype=torch.float32) -> Qwen2ForCausalLM:
    from pagedserve.config import ModelConfig

    cfg = ModelConfig.from_hf_dir(tmp)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        return Qwen2ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev)


def _same(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("buffer_mb", [64, 0.001])
def test_stream_matches_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: bool,
                                  buffer_mb: float) -> None:
    _dump_snapshot(tiny_model(seed=3), tmp_path, split=split)
    ref = _reference(tmp_path, monkeypatch)
    m = _fresh(tmp_path)
    st = stream_weights(m, tmp_path, "cpu", threads=3, buffer_mb=buffer_mb)
    _same(m, ref)
    assert st.tensors > 0 and st.bytes == st.direct_bytes  # fp32 into fp32: all direct


def test_bf16_checkpoint_is_cast_like_the_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _dump_snapshot(tiny_model(seed=4), tmp_path)
    f = tmp_path / "model.safetensors"
    from safetensors.torch import load_file

    save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in load_file(f).items()}, str(f))
    assert read_header(f)[0]["model.embed_tokens.weight"]["dtype"] == "BF16"
    for dt in (torch.float32, torch.float16):
        ref = _reference(tmp_path, monkeypatch, dtype=dt)
        m = _fresh(tmp_path, dtype=dt)
        st = stream_weights(m, tmp_path, "cpu", threads=2, buffer_mb=0.002)
        _same(m, ref)
        assert st.direct_bytes == 0  # every tensor went through staging + cast


def test_errors_match_reference(tmp_path: Path) -> None:
    model = tiny_model(seed=5)
    _dump_snapshot(model, tmp_path, drop={"model.layers.0.mlp.up_proj.weight"})
    with pytest.raises(KeyError, match="never loaded.*up_proj"):
        stream_weights(_fresh(tmp_path), tmp_path, "cpu")
    bad = tmp_path / "bad"
    _dump_snapshot(model, bad)
    from safetensors.torch import load_file

    t = load_file(bad / "model.safetensors")
    t["model.layers.0.self_attn.o_proj.weight"] = torch.zeros(3, 3)
    save_file(t, str(bad / "model.safetensors"))
    with pytest.raises(ValueError, match="shape mismatch"):
        stream_weights(_fresh(bad), bad, "cpu")
    t["model.layers.0.self_attn.o_proj.weight"] = model.state_dict()["model.layers.0.self_attn.o_proj.weight"].clone()
    t["model.layers.0.not_a_param.weight"] = torch.zeros(2)
    save_file(t, str(bad / "model.safetensors"))
    with pytest.raises(KeyError, match="unexpected checkpoint key"):
        stream_weights(_fresh(bad), bad, "cpu")


def test_load_model_uses_stream_and_records_stats(tmp_path: Path) -> None:
    _dump_snapshot(tiny_model(seed=6), tmp_path)
    m = load_model(tmp_path)
    assert m.load_stats.bytes > 0 and m.load_stats.seconds > 0
