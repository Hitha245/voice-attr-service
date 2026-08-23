"""End-to-end orchestration: decoded audio -> quality gate -> model
inference (with graceful fallback) -> API-shaped result.

This is the one place that knows how all the pieces fit together, so
main.py's HTTP/WS handlers stay thin and the same logic can be reused by
the eval harness and by tests.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.inference.age_gender_model import AgeGenderModel, ModelLoadError
from app.inference.fallback_heuristic import classify as heuristic_classify
from app.inference.language_id import identify_language_safe
from app.quality import QualityReport, assess_quality

logger = logging.getLogger("voiceattr.pipeline")


@dataclass
class AnalysisResult:
    contact_id: str
    gender_prediction: str
    gender_confidence: float
    age_prediction: str
    age_confidence: float
    processing_ms: int
    audio_quality: str
    quality_reasons: list[str]
    model_backend: str
    language_prediction: str | None
    language_confidence: float | None
    warnings: list[str]


def _unknown_result(contact_id: str, quality: QualityReport, elapsed_ms: int, warnings: list[str]) -> AnalysisResult:
    return AnalysisResult(
        contact_id=contact_id,
        gender_prediction="unknown",
        gender_confidence=0.0,
        age_prediction="unknown",
        age_confidence=0.0,
        processing_ms=elapsed_ms,
        audio_quality=quality.label,
        quality_reasons=quality.reasons,
        model_backend="none",
        language_prediction=None,
        language_confidence=None,
        warnings=warnings,
    )


def analyze_samples(
    samples: np.ndarray,
    sample_rate: int,
    *,
    contact_id: str | None = None,
) -> AnalysisResult:
    """Run the full pipeline on already-decoded PCM audio.

    Note: `samples` is never written to disk anywhere in this call chain,
    and is dropped (garbage collected) as soon as this function returns —
    the caller (main.py) does not retain a reference either.
    """
    start = time.monotonic()
    contact_id = contact_id or str(uuid.uuid4())
    warnings: list[str] = []

    quality = assess_quality(samples, sample_rate)

    if quality.label == "insufficient":
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "audio quality insufficient, skipping inference",
            extra={"request_id": contact_id, "processing_ms": elapsed_ms, "reasons": quality.reasons},
        )
        return _unknown_result(
            contact_id, quality, elapsed_ms,
            warnings=["audio quality insufficient for reliable inference; predictions withheld"],
        )

    model = AgeGenderModel.get()
    backend = "wav2vec2-age-gender"
    try:
        prediction = model.predict(samples, sample_rate)
        gender_pred, gender_conf = prediction.gender_prediction, prediction.gender_confidence
        age_pred, age_conf = prediction.age_prediction, prediction.age_confidence
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any model failure
        # (load failure, OOM mid-forward-pass, a malformed input the
        # processor chokes on, ...) should degrade gracefully rather than
        # 500 the request, per Task 4's reliability requirement.
        if not settings.ENABLE_MODEL_FALLBACK:
            raise
        kind = "load" if isinstance(exc, ModelLoadError) else "inference"
        logger.warning(
            "primary model %s failure (%s); using DSP heuristic fallback", kind, exc,
            extra={"request_id": contact_id},
        )
        warnings.append(f"primary model {kind} failure; used low-confidence acoustic fallback")
        backend = "heuristic-fallback"
        heuristic = heuristic_classify(samples, sample_rate)
        gender_pred, gender_conf = heuristic.gender_prediction, heuristic.gender_confidence
        age_pred, age_conf = heuristic.age_prediction, heuristic.age_confidence

    if quality.label == "degraded":
        # Degraded audio makes any model's confidence less trustworthy;
        # discount rather than silently pass through a falsely-high score.
        gender_conf = round(gender_conf * 0.7, 4)
        age_conf = round(age_conf * 0.7, 4)
        warnings.append("audio quality degraded; confidence scores discounted")

    language_pred, language_conf = identify_language_safe(samples, sample_rate)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "analysis complete",
        extra={
            "request_id": contact_id,
            "processing_ms": elapsed_ms,
            "backend": backend,
            "audio_quality": quality.label,
            "gender": gender_pred,
            "age_bracket": age_pred,
        },
    )

    return AnalysisResult(
        contact_id=contact_id,
        gender_prediction=gender_pred,
        gender_confidence=gender_conf,
        age_prediction=age_pred,
        age_confidence=age_conf,
        processing_ms=elapsed_ms,
        audio_quality=quality.label,
        quality_reasons=quality.reasons,
        model_backend=backend,
        language_prediction=language_pred,
        language_confidence=language_conf,
        warnings=warnings,
    )
