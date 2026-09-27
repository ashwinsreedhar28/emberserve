"""Int8 dequant GEMM on CUDA at 7B-like shapes, and a quantized tiny engine: the kernel
path and the torch path give the same tokens (both read the same int8 weights)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from pagedserve.config import EngineConfig, ModelConfig  # noqa: E402
from pagedserve.engine import LLMEngine  # noqa: E402
from pagedserve.llm import LLM  # noqa: E402
from pagedserve.model.quant import int8_gemm, int8_gemm_torch, quantize_int8_weight, quantize_model  # noqa: E402
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic  # noqa: E402
from pagedserve.sched.request import SamplingParams  # noqa: E402

DEV = "cuda"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 8, 200, 2048])
@pytest.mark.parametrize("n,k", [(4608, 3584), (3584, 18944)])
def test_kernel_matches_reference(dtype, m, n, k):
    g = torch.Generator().manual_seed(m)
    w = (torch.randn(n, k, generator=g) * 0.02)
    q, s = quantize_int8_weight(w)
    q, s = q.to(DEV), s.to(DEV)
    x = (torch.randn(m, k, generator=g) * 0.5).to(DEV, dtype)
    b = (torch.randn(n, generator=g) * 0.1).to(DEV, dtype)
    got = int8_gemm(x, q, s, b).float()
    want = int8_gemm_torch(x, q, s, b).float()
    tol = 2e-2 if dtype == torch.float16 else 8e-2
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)


def test_quantized_engine_kernel_equals_torch_path(monkeypatch):
    cfg = ModelConfig.tiny(num_attention_heads=4, num_key_value_heads=2, hidden_size=256)
    model = Qwen2ForCausalLM(cfg)
    reset_parameters_deterministic(model, 0)
    model = model.to(DEV, torch.float16)
    quantize_model(model)
    g = torch.Generator().manual_seed(1)
    ps = [torch.randint(2, cfg.vocab_size, (int(torch.randint(3, 40, (1,), generator=g)),), generator=g).tolist()
          for _ in range(8)]
    sp = SamplingParams.greedy(16, ignore_eos=True)

    def run(graphs):
        ecfg = EngineConfig(device=DEV, dtype=torch.float16, block_size=256, num_gpu_blocks=16,
                            max_num_seqs=64, max_num_batched_tokens=4096, max_model_len=512,
                            attn_backend="paged_flash", enable_cuda_graphs=graphs)
        return [r.output_token_ids for r in LLM.from_engine(LLMEngine(model, cfg, ecfg, tokenizer=None)).generate(ps, sp)]

    kernel = run(True)
    monkeypatch.setenv("PAGEDSERVE_INT8_KERNEL", "0")
    torch_path = run(False)
    assert kernel == torch_path
