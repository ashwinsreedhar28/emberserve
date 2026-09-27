"""GPU gate: every backend produces the same greedy tokens; then decode throughput.

    python scripts/gpu_smoke.py [--model models/Qwen2.5-0.5B-Instruct] [--batches 1,8,32,128]

Runs naive / paged_torch / paged_flash / paged_flash+graphs / paged_triton /
paged_triton+graphs on one real prompt and asserts identical tokens, then times batch
decode (random ~256-token prompts, 128 output tokens) per backend. Skips (exit 0)
without CUDA, flash-attn or the model directory; the paged_triton rows are dropped when
triton is not importable. paged_flash uses block_size 256 (flash-attn constraint); the
torch backends and paged_triton use 16.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedserve.config import EngineConfig  # noqa: E402
from pagedserve.engine import LLMEngine  # noqa: E402
from pagedserve.llm import LLM  # noqa: E402
from pagedserve.sched.request import SamplingParams  # noqa: E402

BACKENDS = [("naive", 16, False), ("paged_torch", 16, False),
            ("paged_flash", 256, False), ("paged_flash+graphs", 256, True),
            ("paged_triton", 16, False), ("paged_triton+graphs", 16, True)]


def make(model_dir: str, name: str, block: int, graphs: bool, max_seqs: int) -> LLMEngine:
    cfg = EngineConfig(device="cuda", dtype=torch.float16, block_size=block,
                       attn_backend=name.split("+")[0], enable_cuda_graphs=graphs,
                       max_num_seqs=max_seqs, max_num_batched_tokens=64 * 1024,
                       max_model_len=1024, gpu_memory_utilization=0.6)
    return LLMEngine.from_pretrained(model_dir, cfg)


def top2_margin(eng: LLMEngine, ids: list[int]) -> float:
    """Gap between the two largest next-token logits after `ids`, under `eng` (fp32)."""
    eng.reset()
    req = eng.add_request("margin", ids, SamplingParams.greedy(1))
    so = eng.scheduler.schedule()
    input_ids, meta = eng._build_inputs(so)
    with torch.inference_mode():
        hidden = eng.model(input_ids, eng.backend, meta)
        logits = eng.model.compute_logits(hidden, meta)[0].float()
    eng.abort_request(req.request_id)
    eng.reset()
    top = torch.topk(logits, 2).values
    return float(top[0] - top[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--tie-margin", type=float, default=2e-2,
                    help="a divergence whose top-2 logit gap (under the reference backend) is "
                         "below this is reported as a numeric tie-break, not a failure")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA: skipping")
        return 0
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("flash_attn not installed: skipping")
        return 0
    if not Path(args.model, "config.json").exists():
        print(f"model {args.model} missing (python scripts/download_model.py): skipping")
        return 0
    try:
        import triton  # noqa: F401
        backends = list(BACKENDS)
    except ImportError:
        print("triton not installed: skipping paged_triton rows", file=sys.stderr)
        backends = [b for b in BACKENDS if not b[0].startswith("paged_triton")]
    batches = [int(b) for b in args.batches.split(",")]
    max_seqs = max(batches)

    prompt = "The three laws of thermodynamics, explained simply, are:"
    outputs: dict[str, list[int]] = {}
    engines: dict[str, LLMEngine] = {}
    for name, block, graphs in backends:
        eng = make(args.model, name, block, graphs, max_seqs)
        engines[name] = eng
        res = LLM.from_engine(eng).generate([prompt], SamplingParams.greedy(64))[0]
        outputs[name] = res.output_token_ids
        print(f"[{name:20s}] {res.text[:80]!r}")
    ref = outputs["naive"]
    prompt_ids = engines["naive"].tokenizer.encode(prompt)
    clean = True
    for name, toks in outputs.items():
        if toks == ref:
            continue
        pos = next(i for i, (a, b) in enumerate(zip(toks, ref)) if a != b)
        margin = top2_margin(engines["naive"], prompt_ids + ref[:pos])
        if margin < args.tie_margin:
            print(f"[{name:20s}] tie-break at token {pos} (top-2 logit margin {margin:.2e} "
                  f"< {args.tie_margin:.0e}) - numerics, not a bug")
        else:
            clean = False
            print(f"[{name:20s}] DIVERGED at token {pos}: got {toks[pos]} want {ref[pos]} "
                  f"(margin {margin:.2e})")
    if not clean:
        return 1
    print("all backends agree on greedy tokens (up to sub-margin tie-breaks)\n")

    g = torch.Generator().manual_seed(0)
    vocab = engines["naive"].model_config.vocab_size
    print(f"{'backend':20s} " + " ".join(f"B={b:>4d}" for b in batches) + "   (decode tok/s)")
    for name, _, _ in backends:
        eng = engines[name]
        row = []
        for b in batches:
            ps = [torch.randint(2, vocab // 2, (args.prompt_len,), generator=g).tolist()
                  for _ in range(b)]
            eng.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            LLM.from_engine(eng).generate(ps, SamplingParams.greedy(args.out_tokens, ignore_eos=True))
            torch.cuda.synchronize()
            dec = [s for s in eng.stats if not s.is_prefill]
            dec_s = sum(s.forward_ms + s.sample_ms for s in dec) / 1e3
            row.append(b * args.out_tokens / dec_s if dec_s else float("nan"))
            print(f"  {name} B={b}: total {time.perf_counter()-t0:.2f}s", file=sys.stderr)
        print(f"{name:20s} " + " ".join(f"{r:6.0f}" for r in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
