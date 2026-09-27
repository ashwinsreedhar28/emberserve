# pagedserve on Runpod Serverless.
#
#   docker build -f deploy/runpod/Dockerfile --build-arg MODEL_REPO=Qwen/Qwen2.5-0.5B-Instruct -t <you>/pagedserve-worker:0.5b .
#   docker push <you>/pagedserve-worker:0.5b
#
# then create a Serverless endpoint from the image (GPU: 24 GB is plenty for the 0.5B / 7B fp16
# models with a 4k context; A100 80 GB for Moonlight). Leave MODEL_REPO unset at build time to
# get a model-free image that downloads MODEL_REPO at cold start instead (slower cold starts).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ARG MODEL_REPO=""
ENV MODEL_DIR=/models/model \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY pagedserve ./pagedserve
COPY scripts/download_model.py ./scripts/download_model.py
COPY deploy/runpod/handler.py ./deploy/runpod/handler.py

# flash-attn fetches the prebuilt wheel matching this torch; runpod is the serverless SDK.
RUN pip install --no-cache-dir -e '.[hf,server]' runpod hf_transfer \
 && pip install --no-cache-dir flash-attn --no-build-isolation

RUN if [ -n "$MODEL_REPO" ]; then \
      python scripts/download_model.py --repo "$MODEL_REPO" --out "$MODEL_DIR"; \
    fi

CMD ["python", "-u", "deploy/runpod/handler.py"]
