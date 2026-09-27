"""Reproduce a CUDA-graph capture failure with CUDA_LAUNCH_BLOCKING=1 so the failing
kernel is named instead of "operation failed due to a previous error during capture".

    python scripts/gpu_debug_capture.py --backend paged_triton --block-size 16
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pagedserve.config import EngineConfig, ModelConfig  # noqa: E402
from pagedserve.engine import LLMEngine  # noqa: E402
from pagedserve.model.qwen2 import Qwen2ForCausalLM, reset_parameters_deterministic  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="paged_triton")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA")
        return 0
    cfg = ModelConfig.tiny(num_attention_heads=args.heads, num_key_value_heads=args.kv_heads,
                           hidden_size=args.hidden, num_hidden_layers=args.layers)
    model = Qwen2ForCausalLM(cfg)
    reset_parameters_deterministic(model, 0)
    model = model.to("cuda", torch.float16)
    ecfg = EngineConfig(device="cuda", dtype=torch.float16, block_size=args.block_size,
                        num_gpu_blocks=args.num_blocks, max_num_seqs=args.max_num_seqs,
                        max_num_batched_tokens=4096, max_model_len=args.max_model_len,
                        attn_backend=args.backend, enable_cuda_graphs=True)
    print(f"capturing {args.backend} block={args.block_size} groups={args.heads // args.kv_heads} "
          f"CUDA_LAUNCH_BLOCKING={os.environ['CUDA_LAUNCH_BLOCKING']}")
    eng = LLMEngine(model, cfg, ecfg, tokenizer=None)
    print("capture ok:", eng.graph_runner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
