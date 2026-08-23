# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# ffmpeg: used for all container/codec decoding (mp3, m4a, ogg/opus, webm, ...)
# so we never need fragile python codec bindings for compressed formats.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Install CPU-only torch first from its dedicated wheel index (much smaller
# than the default CUDA wheels pulled from PyPI), then the rest of the
# requirements from PyPI as usual.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY eval ./eval

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /srv /home/appuser
USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker by default: the model is CPU-bound and each worker would
# load its own copy of the ~1.2GB weights. Scale out with multiple
# *containers* behind a load balancer instead (see README > Scaling to
# 1,000 concurrent calls) rather than multiple uvicorn workers in one box.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
