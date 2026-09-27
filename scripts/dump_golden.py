"""Dump HF-transformers golden outputs for the correctness gate.

Runs the reference implementation (HF Qwen2ForCausalLM, fp32, CPU, greedy) on a fixed
prompt set and saves:
  golden/prompts.json         prompt texts + token ids
  golden/greedy.pt            {"prompt_ids": [...], "output_ids": [...]} per prompt (64 new tokens)
  golden/logits_prompt0.pt    all-position logits for prompt 0 (fp32)

pagedserve must reproduce output_ids token-for-token and logits within atol.

    python scripts/dump_golden.py --model models/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Q: What is the boiling point of water in Celsius?\nA:",
    "Once upon a time, in a small village by the sea,",
    "Explain PagedAttention in one paragraph:",
]
CHAT_PROMPTS = [
    [{"role": "user", "content": "Give me three facts about the Moon."}],
    [{"role": "system", "content": "You are a terse assistant."},
     {"role": "user", "content": "Why is the sky blue?"}],
]
MAX_NEW_TOKENS = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="golden")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda; fp32 either way")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="for checkpoints that ship their own modeling code (DeepSeek-V2/V3, Moonlight)")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 trust_remote_code=args.trust_remote_code).eval().to(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    entries = []
    for text in PROMPTS:
        entries.append({"kind": "text", "text": text, "prompt_ids": tok.encode(text)})
    for msgs in CHAT_PROMPTS:
        # tokenize=False + encode: transformers 5.x returns an Encoding object from tokenize=True
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(text, add_special_tokens=False)
        entries.append({"kind": "chat", "messages": msgs, "prompt_ids": ids})

    golden = []
    with torch.inference_mode():
        for i, e in enumerate(entries):
            ids = torch.tensor([e["prompt_ids"]], device=args.device)
            # Pure greedy. Qwen2.5's generation_config.json sets repetition_penalty=1.05,
            # top_k=20, top_p=0.8, temperature=0.7 and do_sample=False does NOT clear the
            # penalty, so override every sampling knob explicitly.
            gen = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.0, temperature=None, top_k=None, top_p=None,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
            out_ids = gen[0, ids.shape[1]:].cpu().tolist()
            golden.append({"prompt_ids": e["prompt_ids"], "output_ids": out_ids})
            print(f"[{i}] {len(e['prompt_ids'])} -> {len(out_ids)} tokens: "
                  f"{tok.decode(out_ids)[:80]!r}")
            if i == 0:
                logits = model(ids).logits[0].float().cpu()
                torch.save({"prompt_ids": e["prompt_ids"], "logits": logits},
                           out_dir / "logits_prompt0.pt")

    (out_dir / "prompts.json").write_text(json.dumps(entries, indent=1))
    torch.save({"model": args.model, "max_new_tokens": args.max_new_tokens,
                "eos_token_id": tok.eos_token_id, "golden": golden}, out_dir / "greedy.pt")
    print(f"wrote {out_dir}/prompts.json, greedy.pt, logits_prompt0.pt")


if __name__ == "__main__":
    main()
