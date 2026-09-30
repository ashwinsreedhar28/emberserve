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


@pytest.mark.parametrize("buffer_mb,whole", [(64, False), (0.001, False), (0.002, True)])
def test_cast_goes_piece_by_piece_unless_a_piece_splits_an_element(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, buffer_mb: float, whole: bool) -> None:
    """bf16 into fp16 is cast one buffer-sized piece at a time (no whole-tensor staging
    buffer: 1.24 GB for Qwen3-8B's embedding). A 2,097-byte buffer (0.002 MB) splits bf16
    elements across pieces, so those tensors fall back to whole-tensor staging; either way
    the parameters equal the reference loader's."""
    _dump_snapshot(tiny_model(seed=8), tmp_path)
    f = tmp_path / "model.safetensors"
    from safetensors.torch import load_file

    save_file({k: v.to(torch.bfloat16) for k, v in load_file(f).items()}, str(f))
    ref = _reference(tmp_path, monkeypatch, dtype=torch.float16)
    m = _fresh(tmp_path, dtype=torch.float16)
    st = stream_weights(m, tmp_path, "cpu", threads=2, buffer_mb=buffer_mb)
    _same(m, ref)
    assert st.direct_bytes == 0
    assert (st.whole_staged_bytes > 0) == whole
    assert st.whole_staged_bytes < st.bytes  # even the odd buffer leaves some pieces aligned


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


def _hold_back_shards(src: Path, dst: Path) -> list[Path]:
    """Copy a split snapshot's small files to `dst` plus an index; return the shard paths
    still to "download"."""
    import json
    import shutil

    from safetensors import safe_open

    dst.mkdir()
    shards = sorted(src.glob("*.safetensors"))
    weight_map = {}
    for sh in shards:
        with safe_open(str(sh), framework="pt") as f:
            weight_map.update({k: sh.name for k in f.keys()})
    (dst / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for p in src.iterdir():
        if p.suffix != ".safetensors":
            shutil.copy(p, dst / p.name)
    return shards


def test_stream_waits_for_shards_still_downloading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    import threading
    import time

    src = tmp_path / "src"
    src.mkdir()
    _dump_snapshot(tiny_model(seed=5), src, split=True)
    ref = _reference(src, monkeypatch)
    dst = tmp_path / "dst"
    shards = _hold_back_shards(src, dst)

    def download() -> None:  # like hf_hub_download: temp name, then rename into place
        for sh in shards:
            time.sleep(0.15)
            tmp = dst / (sh.name + ".incomplete")
            shutil.copy(sh, tmp)
            tmp.rename(dst / sh.name)

    t = threading.Thread(target=download)
    t.start()
    m = _fresh(dst)
    st = stream_weights(m, dst, "cpu", threads=2, buffer_mb=0.001, wait_s=10)
    t.join()
    _same(m, ref)
    assert st.wait_seconds > 0.1 and st.files == [p.name for p in shards]


def test_stream_wait_times_out(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _dump_snapshot(tiny_model(seed=6), src, split=True)
    dst = tmp_path / "dst"
    _hold_back_shards(src, dst)
    with pytest.raises(TimeoutError):
        stream_weights(_fresh(dst), dst, "cpu", wait_s=0.2)


def test_load_model_waits_via_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    import threading
    import time

    src = tmp_path / "src"
    src.mkdir()
    _dump_snapshot(tiny_model(seed=7), src, split=True)
    ref = _reference(src, monkeypatch)
    dst = tmp_path / "dst"
    shards = _hold_back_shards(src, dst)

    def download() -> None:
        time.sleep(0.2)
        for sh in shards:
            shutil.copy(sh, dst / (sh.name + ".part"))
            (dst / (sh.name + ".part")).rename(dst / sh.name)

    threading.Thread(target=download).start()
    monkeypatch.setenv("PAGEDSERVE_WAIT_WEIGHTS_S", "10")
    m = load_model(dst, dtype=torch.float32)
    _same(m, ref)
    assert m.load_stats.wait_seconds > 0.1
