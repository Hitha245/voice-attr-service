"""Pure-DSP fallback gender/age classifier.

This is deliberately *not* the primary model — it exists so the service
degrades gracefully rather than hard-failing when the transformer model
can't be loaded (first-boot weight download still in flight, no network,
corrupted cache, OOM on a small instance, etc; see app/config.py
ENABLE_MODEL_FALLBACK). It only needs numpy + scipy, so it always works
even in a stripped-down environment without torch/transformers.

Approach: autocorrelation-based fundamental frequency (F0/pitch) estimation
per voiced frame, then:
  * gender from median F0 via a smooth logistic decision boundary around the
    ~160-180 Hz region where adult male and female F0 distributions overlap
    (typical adult male speaking F0 ~85-180 Hz, adult female ~165-255 Hz).
  * age bracket from a weak combination of F0 and F0 variability (jitter).
    This is intentionally low-confidence: acoustic age estimation from pitch
    alone is not reliable science, and we say so via a capped confidence
    ceiling rather than pretending otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_F0_HZ = 70.0
MAX_F0_HZ = 400.0
FRAME_MS = 40.0
HOP_MS = 20.0

GENDER_LOGISTIC_CENTER_HZ = 165.0
GENDER_LOGISTIC_SCALE_HZ = 18.0

# Fallback confidence is capped well below the primary model's typical
# operating range so downstream consumers (and calling engineers reading
# this code) never mistake it for the calibrated deep-model output.
GENDER_CONFIDENCE_CEILING = 0.75
AGE_CONFIDENCE_CEILING = 0.40


@dataclass
class HeuristicResult:
    gender_prediction: str
    gender_confidence: float
    age_prediction: str
    age_confidence: float
    median_f0_hz: float | None
    voiced_frame_ratio: float
    backend: str = "heuristic-fallback"


def _frame(samples: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> np.ndarray:
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    if samples.size < frame_len:
        return np.zeros((0, frame_len), dtype=samples.dtype)
    n_frames = 1 + (samples.size - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return samples[idx]


def _autocorr_pitch(frame: np.ndarray, sr: int) -> float | None:
    frame = frame - np.mean(frame)
    energy = np.sum(frame * frame)
    if energy < 1e-8:
        return None
    windowed = frame * np.hanning(frame.size)
    corr = np.correlate(windowed, windowed, mode="full")
    corr = corr[corr.size // 2:]
    if corr[0] <= 0:
        return None
    corr = corr / corr[0]

    min_lag = int(sr / MAX_F0_HZ)
    max_lag = min(int(sr / MIN_F0_HZ), corr.size - 1)
    if max_lag <= min_lag:
        return None

    search = corr[min_lag:max_lag]
    peak_idx = int(np.argmax(search))
    peak_val = search[peak_idx]
    # A clear periodic (voiced) frame has a strong secondary correlation
    # peak; a noisy/unvoiced frame doesn't. 0.3 is a conservative gate.
    if peak_val < 0.3:
        return None
    lag = min_lag + peak_idx
    if lag <= 0:
        return None
    return sr / float(lag)


def estimate_f0_track(samples: np.ndarray, sr: int) -> np.ndarray:
    """Return an array of per-frame F0 estimates in Hz, NaN where unvoiced."""
    frames = _frame(samples, sr, FRAME_MS, HOP_MS)
    f0s = np.full(frames.shape[0], np.nan, dtype=np.float64)
    for i in range(frames.shape[0]):
        f0 = _autocorr_pitch(frames[i], sr)
        if f0 is not None:
            f0s[i] = f0
    return f0s


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def classify(samples: np.ndarray, sr: int) -> HeuristicResult:
    f0_track = estimate_f0_track(samples, sr)
    voiced = f0_track[~np.isnan(f0_track)]
    voiced_ratio = float(voiced.size) / float(f0_track.size) if f0_track.size else 0.0

    if voiced.size < 3:
        return HeuristicResult(
            gender_prediction="unknown",
            gender_confidence=0.0,
            age_prediction="unknown",
            age_confidence=0.0,
            median_f0_hz=None,
            voiced_frame_ratio=voiced_ratio,
        )

    median_f0 = float(np.median(voiced))

    # --- Gender: logistic function of median F0 ---
    z = (median_f0 - GENDER_LOGISTIC_CENTER_HZ) / GENDER_LOGISTIC_SCALE_HZ
    p_female = _sigmoid(z)
    p_male = 1.0 - p_female
    if p_female >= p_male:
        gender_pred, gender_conf = "female", p_female
    else:
        gender_pred, gender_conf = "male", p_male
    gender_conf = float(min(gender_conf, GENDER_CONFIDENCE_CEILING))
    # A prediction sitting right on the decision boundary carries near-zero
    # real information; don't dress it up as a confident call.
    if abs(median_f0 - GENDER_LOGISTIC_CENTER_HZ) < 5.0:
        gender_pred = "unknown"
        gender_conf = 0.0

    # --- Age: weak proxy from F0 jitter (cycle-to-cycle F0 perturbation).
    # Increased jitter is loosely associated with aging voices (vocal fold
    # stiffening / reduced control); this is a much weaker signal than
    # gender-from-pitch and should not be trusted for real decisions - it's
    # only here so the fallback path returns *something* structurally valid
    # rather than always emitting "unknown".
    if voiced.size >= 5:
        diffs = np.abs(np.diff(voiced))
        jitter = float(np.mean(diffs / voiced[:-1]))
    else:
        jitter = 0.0

    if jitter > 0.06:
        age_pred = "60+"
        age_conf = min(AGE_CONFIDENCE_CEILING, 0.25 + jitter)
    elif jitter > 0.035:
        age_pred = "46-60"
        age_conf = AGE_CONFIDENCE_CEILING * 0.7
    elif jitter > 0.02:
        age_pred = "31-45"
        age_conf = AGE_CONFIDENCE_CEILING * 0.6
    else:
        age_pred = "18-30"
        age_conf = AGE_CONFIDENCE_CEILING * 0.5

    return HeuristicResult(
        gender_prediction=gender_pred,
        gender_confidence=round(gender_conf, 4),
        age_prediction=age_pred,
        age_confidence=round(float(age_conf), 4),
        median_f0_hz=median_f0,
        voiced_frame_ratio=voiced_ratio,
    )
