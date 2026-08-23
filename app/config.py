"""Centralized runtime configuration, all overridable via environment variables
so the same image behaves correctly in docker-compose, CI, and local dev.
"""
from __future__ import annotations

import os


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


class Settings:
    # --- Model backend ---
    AGE_GENDER_MODEL_NAME: str = os.getenv(
        "AGE_GENDER_MODEL_NAME", "audeering/wav2vec2-large-robust-24-ft-age-gender"
    )
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "cpu")  # "cpu" or "cuda"
    HF_HOME: str = os.getenv("HF_HOME", "/root/.cache/huggingface")
    # If the primary transformer model fails to load (no network on first
    # boot, corrupt cache, OOM, etc.) fall back to the pure-DSP heuristic
    # classifier instead of hard-failing every request.
    ENABLE_MODEL_FALLBACK: bool = _bool("ENABLE_MODEL_FALLBACK", True)
    # Preload the model at process startup rather than on first request, so
    # the (slow, one-time) weight load doesn't count against a caller's
    # latency budget. Adds ~seconds to container start.
    PRELOAD_MODEL: bool = _bool("PRELOAD_MODEL", True)

    # --- Bonus: language ID ---
    ENABLE_LANGUAGE_ID: bool = _bool("ENABLE_LANGUAGE_ID", False)
    WHISPER_LANGID_MODEL: str = os.getenv("WHISPER_LANGID_MODEL", "tiny")

    # --- Audio constraints ---
    TARGET_SAMPLE_RATE: int = 16000
    MAX_AUDIO_SECONDS: float = _float("MAX_AUDIO_SECONDS", 30.0)
    MIN_AUDIO_SECONDS_FOR_INFERENCE: float = _float("MIN_AUDIO_SECONDS_FOR_INFERENCE", 1.0)
    MAX_UPLOAD_BYTES: int = _int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)  # 25 MB
    FFMPEG_TIMEOUT_SECONDS: float = _float("FFMPEG_TIMEOUT_SECONDS", 10.0)

    # --- Streaming (WebSocket) ---
    WS_INFERENCE_WINDOW_SECONDS: float = _float("WS_INFERENCE_WINDOW_SECONDS", 2.0)
    WS_MAX_SESSION_SECONDS: float = _float("WS_MAX_SESSION_SECONDS", 120.0)

    # --- Observability ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_JSON: bool = _bool("LOG_JSON", True)

    # --- Concurrency ---
    # torch CPU inference releases the GIL during the heavy matmuls but is
    # still expensive; cap how many requests can be doing model forward
    # passes at once so we degrade to queueing instead of thrashing.
    INFERENCE_CONCURRENCY: int = _int("INFERENCE_CONCURRENCY", max(1, os.cpu_count() or 2))


settings = Settings()
