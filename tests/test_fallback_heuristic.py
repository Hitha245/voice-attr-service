from __future__ import annotations

import numpy as np

from app.inference.fallback_heuristic import classify

SR = 16000


def _synth_pitch(f0: float, dur: float = 2.0, jitter_amt: float = 0.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    frame_len = int(SR * 0.02)
    sig = np.zeros(n)
    phase = 0.0
    i = 0
    while i < n:
        f = f0 * (1 + jitter_amt * rng.standard_normal())
        seg_n = min(frame_len, n - i)
        tt = np.arange(seg_n) / SR
        seg = 0.5 * np.sin(2 * np.pi * f * tt + phase) + 0.2 * np.sin(2 * np.pi * 2 * f * tt + phase)
        sig[i : i + seg_n] = seg
        phase += 2 * np.pi * f * seg_n / SR
        i += seg_n
    return sig.astype(np.float32)


def test_low_pitch_classified_male():
    result = classify(_synth_pitch(110), SR)
    assert result.gender_prediction == "male"
    assert result.gender_confidence > 0.5


def test_high_pitch_classified_female():
    result = classify(_synth_pitch(220), SR)
    assert result.gender_prediction == "female"
    assert result.gender_confidence > 0.5


def test_boundary_pitch_is_unknown():
    result = classify(_synth_pitch(165), SR)
    assert result.gender_prediction == "unknown"
    assert result.gender_confidence == 0.0


def test_white_noise_has_no_confident_pitch():
    noise = (np.random.default_rng(2).standard_normal(SR * 2) * 0.05).astype(np.float32)
    result = classify(noise, SR)
    assert result.gender_prediction == "unknown"
    assert result.voiced_frame_ratio < 0.1


def test_confidence_is_capped_below_primary_model_range():
    result = classify(_synth_pitch(110), SR)
    from app.inference.fallback_heuristic import GENDER_CONFIDENCE_CEILING

    assert result.gender_confidence <= GENDER_CONFIDENCE_CEILING


def test_jittery_pitch_shifts_age_bracket_older():
    stable = classify(_synth_pitch(105, jitter_amt=0.0), SR)
    jittery = classify(_synth_pitch(105, jitter_amt=0.08), SR)
    brackets_order = ["18-30", "31-45", "46-60", "60+", "unknown"]
    assert brackets_order.index(jittery.age_prediction) >= brackets_order.index(stable.age_prediction)
