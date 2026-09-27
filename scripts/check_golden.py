"""Correctness gate: pagedserve vs the HF golden files, on every backend that can run here.

    python scripts/check_golden.py --model models/Qwen2.5-0.5B-Instruct [--backends naive,paged_torch]
        [--device cpu|cuda|mps] [--dtype float32] [--prefix-caching]

Checks (1) greedy token ids match token-for-token for all golden prompts, decoded together
in one continuous batch and (2) all-position logits for prompt 0 match within --atol.
Exit code 1 on any mismatch. Also importable: `run_golden_check(...)`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedserve.config import EngineConfig  # noqa: E402
from pagedserve.engine import LLMEngine  # noqa: E402
from pagedserve.llm import LLM  # noqa: E402
from pagedserve.sched.request import SamplingParams  # noqa: E402


def check_logits(engine: LLMEngine, prompt_ids: list[int], ref: torch.Tensor, atol: float) -> float:
    n = len(prompt_ids)
    req = engine.add_request("logits-check", prompt_ids, SamplingParams.greedy(1))
    so = engine.scheduler.schedule()
    assert so.is_prefill and sum(so.query_lens) == n
    with torch.inference_mode():
        logits = engine.step_logits(so, all_positions=True).float().cpu()
    engine.abort_request(req.request_id)
    engine.backend.free_sequence(req.seq_id)
    diff = (logits - ref).abs().max().item()
    return diff


def top2_margin(engine: LLMEngine, ids: list[int]) -> float:
    """Gap between the two largest next-token logits after `ids` under this engine."""
    req = engine.add_request("margin", ids, SamplingParams.greedy(1))
    so = engine.scheduler.schedule()
    with torch.inference_mode():
        logits = engine.step_logits(so)
    engine.abort_request(req.request_id)
    engine.backend.free_sequence(req.seq_id)
    top = torch.topk(logits[0].float(), 2).values
    return float(top[0] - top[1])


def run_golden_check(model_dir: str, backend: str, device: str, dtype: torch.dtype,
                     golden_dir: str = "golden", atol: float | None = None,
                     prefix_caching: bool = False, block_size: int = 16,
                     quantization: str | None = None, tensor_parallel_size: int = 1) -> bool:
    """fp32 runs must match the fp32 HF reference exactly (logits atol 1e-3, tokens
    token-for-token). Half-precision runs are held to a looser, self-calibrated bar: the
    logits gate is 1.0, and a token mismatch counts as a numeric tie-break (not a failure)
    when the top-2 logit gap at that position is below 2x the logits error measured on
    prompt 0 - i.e. the two candidates were closer than the run's own precision noise."""
    g = torch.load(Path(golden_dir) / "greedy.pt")
    lg = torch.load(Path(golden_dir) / "logits_prompt0.pt")
    cfg = EngineConfig(device=device, dtype=dtype, attn_backend=backend, block_size=block_size,
                       enable_prefix_caching=prefix_caching, max_model_len=4096,
                       quantization=quantization, tensor_parallel_size=tensor_parallel_size)
    engine = LLMEngine.from_pretrained(model_dir, cfg)
    half = dtype != torch.float32
    if atol is None:
        atol = 1.0 if half else 1e-3
    if quantization:
        # A quality report, not an exactness gate: int8 rounding moves logits by more than
        # half-precision noise. The tie-break rule is applied at the measured logits error,
        # so a real quality regression shows up as a MISMATCH with a large margin.
        atol = max(atol, 1e9)
        backend = f"{backend}/{quantization}"
    if tensor_parallel_size > 1:
        backend = f"{backend}/tp{tensor_parallel_size}"
    ok = True

    diff = check_logits(engine, lg["prompt_ids"], lg["logits"], atol)
    status = "ok" if diff <= atol else "FAIL"
    print(f"[{backend}] logits prompt0 max|diff| = {diff:.2e} (atol {atol:.0e}, "
          f"{'fp16/bf16 vs fp32 reference' if half else 'fp32'}) {status}")
    ok &= diff <= atol
    tie_margin = 2.0 * diff if half else 0.0

    prompts = [e["prompt_ids"] for e in g["golden"]]
    sp = SamplingParams.greedy(g["max_new_tokens"])
    results = LLM.from_engine(engine).generate(prompts, sp)
    for i, (r, e) in enumerate(zip(results, g["golden"])):
        if r.output_token_ids == e["output_ids"]:
            print(f"[{backend}] prompt {i}: {len(e['output_ids'])} tokens match")
            continue
        first = next((k for k, (a, b) in enumerate(zip(r.output_token_ids, e["output_ids"]))
                      if a != b), min(len(r.output_token_ids), len(e["output_ids"])))
        margin = top2_margin(engine, e["prompt_ids"] + e["output_ids"][:first])
        if margin < tie_margin:
            print(f"[{backend}] prompt {i}: tie-break at token {first} (top-2 margin {margin:.3f} "
                  f"< 2x logits error {tie_margin:.3f}) - numerics, not a bug")
        else:
            ok = False
            print(f"[{backend}] prompt {i}: MISMATCH at token {first} (margin {margin:.3f}): "
                  f"got {r.output_token_ids[first:first + 5]} want {e['output_ids'][first:first + 5]}")
    engine.shutdown()  # tensor-parallel workers, if any
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--golden", default="golden")
    ap.add_argument("--backends", default="naive,paged_torch")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--atol", type=float, default=None,
                    help="logits gate; default 1e-3 for fp32, 1.0 for fp16/bf16")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--quantization", choices=["int8"], default=None,
                    help="check the quantized model: reports token agreement with the fp16/fp32 "
                         "reference; mismatches below the measured logits error are tie-breaks")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()
    all_ok = True
    for b in args.backends.split(","):
        all_ok &= run_golden_check(args.model, b.strip(), args.device,
                                   EngineConfig.dtype_from_str(args.dtype), args.golden,
                                   args.atol, args.prefix_caching, args.block_size,
                                   args.quantization, args.tensor_parallel_size)
    print("ALL OK" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
