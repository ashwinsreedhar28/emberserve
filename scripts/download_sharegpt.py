"""Fetch the ShareGPT dump the text-trace benchmarks sample from (the same file vLLM's
`benchmark_serving.py` uses): ~670 MB, once.

    python scripts/download_sharegpt.py            # -> data/ShareGPT_V3_unfiltered_cleaned_split.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(REPO, FILE, repo_type="dataset", local_dir=str(out))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
