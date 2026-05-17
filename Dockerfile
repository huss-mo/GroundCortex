# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# GroundCortex - memory consolidation engine
#
# CPU build (default):
#   docker build .
#
# GPU build (CUDA 12.4 - requires NVIDIA Container Toolkit on the host):
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 .
#
# The base image is always python:3.12-slim. On GPU hosts, PyTorch is pulled
# from the CUDA wheel index instead of PyPI. Model inference and training
# auto-detect CUDA → MPS → CPU at runtime so no code change is needed.
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.12
ARG TORCH_INDEX=""

# ---------------------------------------------------------------------------
# Stage 1: builder - installs gcc + all Python deps into an isolated venv.
# The compiler is only needed here; the runtime stage stays small.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TORCH_INDEX

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY pyproject.toml ./
COPY groundcortex/ ./groundcortex/

# When TORCH_INDEX is set, override the PyTorch source with the CUDA index.
# When empty, pip resolves the CPU-only wheel from PyPI (the default).
RUN if [ -n "${TORCH_INDEX}" ]; then \
        pip install --no-cache-dir . \
            --extra-index-url "${TORCH_INDEX}"; \
    else \
        pip install --no-cache-dir .; \
    fi

# ---------------------------------------------------------------------------
# Stage 2: runtime - clean image, no compiler, no build cache.
# Only the installed venv is copied from the builder stage.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

WORKDIR /app

COPY --from=builder /venv /venv

# PYTHONUTF8=1 - force UTF-8 file I/O on all platforms.
#
# TRL 1.4.0 reads deepseekv3.jinja without an explicit encoding= argument.
# On any system whose default locale is not UTF-8 (Windows cp1252, some
# minimal Linux containers), this causes UnicodeDecodeError: 'cp1252' codec
# can't decode byte 0x81 at position 932. Setting PYTHONUTF8=1 before any
# Python process starts makes the default encoding UTF-8 everywhere, which is
# what TRL (and every other modern library) assumes.
#
# HF_HOME=/models - redirect the Hugging Face model cache out of the home
# directory so it lands on the /models volume mount (see docker-compose.yml).
# All model weights downloaded from the Hub are stored here. Without this,
# they land in /root/.cache/huggingface/ inside the ephemeral container layer
# and are re-downloaded on every container restart.
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    HF_HOME=/models \
    GROUNDCORTEX_OUTPUT_DIR=/app/adapters \
    GROUNDCORTEX_BUFFER_DB=/app/data/groundcortex.db

EXPOSE 4343 4344

# Declare the three host-mapped directories as volumes.
# Actual bind mounts are configured in docker-compose.yml.
#   /models       - Hugging Face model weights (large, persisted on host)
#   /app/adapters - trained LoRA adapters
#   /app/data     - SQLite database
VOLUME ["/models", "/app/adapters", "/app/data"]

CMD ["groundcortex"]
