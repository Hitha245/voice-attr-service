from __future__ import annotations

import numpy as np

from app.quality import assess_quality

SR = 16000


def _voice_like(dur=3.0, f0=140, noise_amp=0.0, amp=0.4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur)) / SR
    sig = amp * np.sin(2 * np.pi * f0 * t) + 0.15 * amp * np.sin(2 * np.pi * 2 * f0 * t)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)
    sig = sig * env + noise_amp * rng.standard_normal(sig.size)
    return sig.astype(np.float32)


def test_clean_signal_is_good():
    report = assess_quality(_voice_like(noise_amp=0.0), SR)
    assert report.label == "good"
    assert report.snr_db > 20


def test_moderately_noisy_signal_is_degraded():
    report = assess_quality(_voice_like(noise_amp=0.15), SR)
    assert report.label == "degraded"
    assert report.reasons


def test_heavily_noisy_signal_is_insufficient():
    report = assess_quality(_voice_like(noise_amp=0.6), SR)
    assert report.label == "insufficient"


def test_silence_is_insufficient():
    report = assess_quality(np.zeros(SR * 2, dtype=np.float32), SR)
    assert report.label == "insufficient"
    assert "near-silent" in " ".join(report.reasons)


def test_too_short_clip_is_insufficient():
    report = assess_quality(_voice_like(dur=0.4), SR)
    assert report.label == "insufficient"


def test_empty_array_does_not_crash():
    report = assess_quality(np.array([], dtype=np.float32), SR)
    assert report.label == "insufficient"


def test_clipped_signal_flags_degraded():
    clipped = np.clip(_voice_like(amp=1.5), -1.0, 1.0)
    report = assess_quality(clipped, SR)
    assert report.label == "degraded"
    assert report.clipping_ratio > 0.01
