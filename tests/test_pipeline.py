"""Pipeline-level tests that exercise decode -> quality -> inference without
going through the HTTP layer (so they still run in environments without
fastapi installed, e.g. this repo's CI-less sandbox). See test_api_smoke.py
for the full HTTP-layer equivalent.
"""
from __future__ import annotations

import numpy as np

from app.audio_io import decode_audio_bytes
from app.inference.pipeline import analyze_samples
from tests.conftest import SR, make_tone_wav_bytes


def test_analyze_samples_on_bundled_sample_returns_well_formed_result(sample_wav_bytes):
    decoded = decode_audio_bytes(sample_wav_bytes)
    result = analyze_samples(decoded.samples, decoded.sample_rate, contact_id="smoke-test")

    assert result.contact_id == "smoke-test"
    assert result.audio_quality in {"good", "degraded", "insufficient"}
    assert result.gender_prediction in {"male", "female", "unknown"}
    assert result.age_prediction in {"18-30", "31-45", "46-60", "60+", "unknown"}
    assert 0.0 <= result.gender_confidence <= 1.0
    assert 0.0 <= result.age_confidence <= 1.0
    assert result.processing_ms >= 0
    assert result.model_backend in {"wav2vec2-age-gender", "heuristic-fallback", "none"}


def test_analyze_samples_generates_contact_id_when_omitted():
    decoded_samples = np.zeros(SR * 2, dtype=np.float32)
    result = analyze_samples(decoded_samples, SR)
    assert result.contact_id  # non-empty, auto-generated UUID
    assert result.audio_quality == "insufficient"
    assert result.gender_prediction == "unknown"


def test_silence_short_circuits_before_model_call():
    silence = np.zeros(SR * 2, dtype=np.float32)
    result = analyze_samples(silence, SR, contact_id="silence-case")
    assert result.audio_quality == "insufficient"
    assert result.model_backend == "none"
    assert "predictions withheld" in " ".join(result.warnings)


def test_decode_and_analyze_ad_hoc_tone():
    raw = make_tone_wav_bytes(freq_hz=110.0, duration_s=2.5)
    decoded = decode_audio_bytes(raw)
    result = analyze_samples(decoded.samples, decoded.sample_rate)
    assert result.audio_quality in {"good", "degraded"}
