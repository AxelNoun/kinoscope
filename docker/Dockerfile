# syntax=docker/dockerfile:1.7
# Multi-stage : la cible `serve` ne contient ni CUDA, ni compilateur, ni jeu de tests.

# --------------------------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1 UV_SYSTEM_PYTHON=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*
RUN pip install uv==0.4.9
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache ".[serve]" \
 && uv pip install --system --no-cache torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# --------------------------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS serve
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 kinoscope
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
WORKDIR /app
COPY --chown=kinoscope:kinoscope src ./src
COPY --chown=kinoscope:kinoscope configs ./configs
USER kinoscope
ENV PYTHONPATH=/app/src
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "kinoscope.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# --------------------------------------------------------------------------- #
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS train
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip git \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip3 install --no-cache-dir ".[dev]"
COPY . .
ENV PYTHONPATH=/app/src
ENTRYPOINT ["python3", "scripts/train.py"]
