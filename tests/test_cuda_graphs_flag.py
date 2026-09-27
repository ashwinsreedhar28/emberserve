"""enable_cuda_graphs on CPU is ignored with a warning (runs everywhere)."""

import pytest

from pagedserve.config import EngineConfig, ModelConfig
from pagedserve.engine import LLMEngine
from pagedserve.model.qwen2 import Qwen2ForCausalLM

CFG = ModelConfig.tiny()


def test_flag_ignored_on_cpu():
    model = Qwen2ForCausalLM(CFG)
    ecfg = EngineConfig(device="cpu", num_gpu_blocks=8, block_size=4, enable_cuda_graphs=True)
    with pytest.warns(UserWarning, match="enable_cuda_graphs ignored"):
        eng = LLMEngine(model, CFG, ecfg)
    assert eng.graph_runner is None and eng.block_manager.num_blocks == 8
