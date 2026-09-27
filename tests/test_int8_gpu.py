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
@pytest.mark.parametrize("m", [1, 8, 33, 200, 2048])
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


def test_fixed_configs_match_autotuned(monkeypatch):
    """Every tile config the autotuner can pick computes the same thing (the [BK, BN]
    weight-tile read is exercised by each), and the load-time warmup runs every bucket."""
    from pagedserve.model import quant

    g = torch.Generator().manual_seed(3)
    q, s = quantize_int8_weight(torch.randn(3584, 4608, generator=g) * 0.02)
    q, s = q.to(DEV), s.to(DEV)
    for m in (1, 16, 17, 64, 300):
        x = (torch.randn(m, 4608, generator=g) * 0.5).to(DEV, torch.float16)
        want = int8_gemm_torch(x, q, s).float()
        torch.testing.assert_close(int8_gemm(x, q, s).float(), want, atol=2e-2, rtol=2e-2)
        cfgs = quant._CONFIGS_SMALL_M if m <= 16 else quant._CONFIGS_LARGE_M
        for cfg in cfgs:
            import triton

            out = torch.empty((m, 3584), dtype=torch.float16, device=DEV)
            grid = (triton.cdiv(m, cfg["BM"]), triton.cdiv(3584, cfg["BN"]))
            quant._kernel()[grid](x, q, s, s, out, out, m, 3584, 4608, x.stride(0), q.stride(0), out.stride(0),
                                  M_BUCKET=quant._m_bucket(m), SPLIT_K=1, HAS_BIAS=False, BM=cfg["BM"],
                                  BN=cfg["BN"], BK=cfg["BK"], num_warps=cfg["num_warps"],
                                  num_stages=cfg["num_stages"])
            torch.testing.assert_close(out.float(), want, atol=2e-2, rtol=2e-2), cfg
        # split-K off must give the same answer as the host's choice of splits
        monkeypatch.setenv("PAGEDSERVE_INT8_SPLITK", "0")
        torch.testing.assert_close(int8_gemm(x, q, s).float(), want, atol=2e-2, rtol=2e-2)
        monkeypatch.delenv("PAGEDSERVE_INT8_SPLITK")
    lin = torch.nn.Linear(4608, 3584, bias=False).to(DEV, torch.float16)
    model = torch.nn.Sequential(lin)
    quantize_model(model)
    assert quant.warm_int8_kernels(model, max_m=1024) == 11  # 1, 2, ..., 1024


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
