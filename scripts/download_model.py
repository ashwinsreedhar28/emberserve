"""Download an HF snapshot into models/ (weights, config, tokenizer only). Any Qwen2 / Llama / Mistral checkpoint.

    python scripts/download_model.py [--repo Qwen/Qwen2.5-0.5B-Instruct] [--out models/Qwen2.5-0.5B-Instruct]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"models/{args.repo.split('/')[-1]}")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        args.repo, local_dir=str(out),
        allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json", "*.txt"],
    )
    print(f"downloaded {args.repo} -> {path}")


if __name__ == "__main__":
    main()
