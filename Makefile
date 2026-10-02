.PHONY: test lint golden gpu-smoke serve
test:        ; python -m pytest -q -m "not gpu and not hf"
lint:        ; ruff check emberserve tests scripts
model:       ; python scripts/download_model.py
golden:      ; python scripts/dump_golden.py && python scripts/check_golden.py
gpu-smoke:   ; python scripts/gpu_smoke.py
serve:       ; python -m emberserve.cli serve --model models/Qwen2.5-0.5B-Instruct
