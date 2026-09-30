"""CUDA graphs captured on the empty model, weights loaded after (`graphs_before_weights`),
must reproduce the usual order exactly (GPU only): same parameters and the same greedy
tokens through the full and piecewise graphs. Also the streaming loader's piece-by-piece bf16 -> fp16 cast on the device against the reference loader.

Run: `python -m pytest -m gpu -q tests/test_graphs_before_weights_gpu.py`. The real-model
check is the golden gate with the flag forced:
`PAGEDSERVE_GRAPHS_BEFORE_WEIGHTS=1 python scripts/check_golden.py --device cuda
--dtype float16 --backends paged_flash,paged_triton --block-size 256 --cuda-graphs`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pagedserve.config import EngineConfig
from pagedserve.engine import GRAPHS_BEFORE_WEIGHTS_ENV, LLMEngine
from pagedserve.llm import LLM
from pagedserve.sched.request import SamplingParams
from tests.test_model import tiny_model
from tests.test_weights import _dump_snapshot

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("flash_attn")


def _engine(model_dir: Path, piecewise: bool) -> LLMEngine:
    cfg = EngineConfig(device="cuda", dtype=torch.float16, attn_backend="paged_flash",
                       block_size=256, max_model_len=512, max_num_seqs=64,
                       max_num_batched_tokens=1024, enable_cuda_graphs=True,
                       piecewise_cuda_graphs=piecewise, num_gpu_blocks=64)
    return LLMEngine.from_pretrained(model_dir, cfg, load_tokenizer=False)


def _gen(eng: LLMEngine) -> list[list[int]]:
    g = torch.Generator().manual_seed(11)
    ps = [torch.randint(2, eng.model_config.vocab_size, (int(n),), generator=g).tolist()
          for n in torch.randint(3, 60, (13,), generator=g)]
    return [r.output_token_ids for r in
            LLM.from_engine(eng).generate(ps, SamplingParams.greedy(24, ignore_eos=True))]


@pytest.mark.parametrize("piecewise", [False, True])
def test_graphs_first_matches_the_usual_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              piecewise: bool) -> None:
    monkeypatch.delenv("PAGEDSERVE_WAIT_WEIGHTS_S", raising=False)
    _dump_snapshot(tiny_model(seed=21), tmp_path, split=True)
    monkeypatch.setenv(GRAPHS_BEFORE_WEIGHTS_ENV, "0")
    usual = _engine(tmp_path, piecewise)
    want = _gen(usual)
    usual_state = {k: v.clone() for k, v in usual.model.state_dict().items()}
    usual.release_graphs()
    del usual
    torch.cuda.empty_cache()

    monkeypatch.setenv(GRAPHS_BEFORE_WEIGHTS_ENV, "1")
    first = _engine(tmp_path, piecewise)
    assert "engine built before the weights" in first.boot_notes and "again" not in first.boot_notes
    assert first.graph_runner is not None and (first.piecewise_runner is not None) == piecewise
    st = first.model.state_dict()
    assert all(torch.equal(st[k], usual_state[k]) for k in usual_state)
    assert _gen(first) == want


def test_bf16_checkpoint_cast_piecewise_on_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from safetensors.torch import load_file, save_file

    from pagedserve.model.fastload import stream_weights
    from pagedserve.model.weights import build_empty_model, load_model

    _dump_snapshot(tiny_model(seed=22), tmp_path)
    f = tmp_path / "model.safetensors"
    save_file({k: v.to(torch.bfloat16) for k, v in load_file(f).items()}, str(f))
    monkeypatch.setenv("PAGEDSERVE_LOADER", "safetensors")
    ref = load_model(tmp_path, device="cuda", dtype=torch.float16)
    monkeypatch.delenv("PAGEDSERVE_LOADER")
    for buffer_mb in (64, 0.001):
        m = build_empty_model(tmp_path, device="cuda", dtype=torch.float16)
        stats = stream_weights(m, tmp_path, "cuda", buffer_mb=buffer_mb)
        assert stats.whole_staged_bytes == 0 and stats.direct_bytes == 0
        sr, sm = ref.state_dict(), m.state_dict()
        assert all(torch.equal(sr[k], sm[k]) for k in sr)
