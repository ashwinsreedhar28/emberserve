"""Golden gate as a pytest (skips unless the model and golden files are present).

    python scripts/download_model.py && python scripts/dump_golden.py && pytest -m hf -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
GOLDEN = ROOT / "golden" / "greedy.pt"

pytestmark = pytest.mark.hf
if not (MODEL.exists() and GOLDEN.exists()):
    pytest.skip("model or golden files missing", allow_module_level=True)
pytest.importorskip("transformers")
sys.path.insert(0, str(ROOT / "scripts"))
from check_golden import run_golden_check  # noqa: E402


@pytest.mark.parametrize("backend,prefix", [("naive", False), ("paged_torch", False),
                                            ("paged_torch", True)])
def test_matches_hf_golden(backend: str, prefix: bool) -> None:
    assert run_golden_check(str(MODEL), backend, "cpu", torch.float32, str(GOLDEN.parent),
                            atol=1e-3, prefix_caching=prefix)
