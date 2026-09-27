#!/usr/bin/env bash
# One-shot setup for a fresh Runpod PyTorch pod (no volume). ~10 min; vLLM is the slow part.
# Usage: bash scripts/pod_setup.sh [--no-vllm]
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip install flash-attn --no-build-isolation            # fetches the prebuilt wheel for the pod's torch
pip install -e '.[hf,server,dev]'
python scripts/download_model.py
python scripts/dump_golden.py --device cuda            # on the GPU: the pod CPUs take 10+ min for this
if [[ "${1:-}" != "--no-vllm" ]]; then
  # vLLM pins its own torch; keep it in a venv so it cannot break flash-attn.
  python -m venv /opt/vllm && /opt/vllm/bin/pip install -q vllm
  echo "vLLM: /opt/vllm/bin/vllm (pass --vllm-bin /opt/vllm/bin/vllm to run_vllm_baseline)"
fi
git config --global user.email "ashwin.sreedhar2003@gmail.com"
git config --global user.name "Ashwin Sreedhar"
python -c "import torch, flash_attn, triton; print('ok: flash', flash_attn.__version__, 'triton', triton.__version__)"
echo "next: python scripts/gpu_smoke.py && python -m pytest -q -m gpu"
