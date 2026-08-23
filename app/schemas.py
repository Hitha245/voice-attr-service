"""Pydantic response/request models. `AnalyzeResponse` matches the API
contract in the assignment exactly (contact_id, gender, age_bracket,
processing_ms, audio_quality) with a few additive, optional fields for
observability and the bonus language-ID feature — additive fields don't
break clients that only read the required contract keys.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

GenderLabel = Literal["male", "female", "unknown"]
AgeBracketLabel = Literal["18-30", "31-45", "46-60", "60+", "unknown"]
AudioQualityLabel = Literal["good", "degraded", "insufficient"]


class GenderResult(BaseModel):
    prediction: GenderLabel
    confidence: float = Field(ge=0.0, le=1.0)


class AgeBracketResult(BaseModel):
    prediction: AgeBracketLabel
    confidence: float = Field(ge=0.0, le=1.0)


class LanguageResult(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    contact_id: str
    gender: GenderResult
    age_bracket: AgeBracketResult
    processing_ms: int
    audio_quality: AudioQualityLabel

    # --- Additive fields beyond the required contract ---
    language: Optional[LanguageResult] = Field(
        default=None, description="Bonus best-effort language ID; null when disabled/unavailable."
    )
    model_backend: str = Field(
        description="Which inference backend produced this result: "
        "'wav2vec2-age-gender' (primary) or 'heuristic-fallback' (degraded mode)."
    )
    quality_reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons behind the audio_quality label, for debugging/observability.",
    )
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: str
    contact_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_backend: str
    model_load_error: Optional[str] = None
