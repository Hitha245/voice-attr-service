"""Audio quality assessment.

Logistics calls come from truck cabs, warehouse floors, and drive-thru-noisy
phones. Rather than let bad audio silently produce a confident-looking wrong
answer, we score the input first and let the caller (and the confidence
calibration downstream) know when it's on shaky ground. Everything here is
pure numpy — no ML model needed to answer "is this usable audio?".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FRAME_MS = 25.0
HOP_MS = 10.0

SILENCE_RMS_DBFS = -50.0       # below this, we call it silence/insufficient
CLIPPING_AMPLITUDE = 0.997     # samples this close to full-scale count as clipped
CLIPPING_RATIO_DEGRADED = 0.01  # >1% of samples clipped -> degraded
SNR_DB_DEGRADED = 12.0          # below this estimated SNR -> degraded
SNR_DB_INSUFFICIENT = 3.0       # below this -> effectively unusable
VOICED_RATIO_DEGRADED = 0.25    # too little of the clip looks like speech
MIN_DURATION_INSUFFICIENT_S = 1.0
MIN_VOICED_DURATION_INSUFFICIENT_S = 0.4


@dataclass
class QualityReport:
    label: str  # "good" | "degraded" | "insufficient"
    duration_s: float
    voiced_duration_s: float
    voiced_ratio: float
    rms_dbfs: float
    snr_db: float
    clipping_ratio: float
    reasons: list[str] = field(default_factory=list)


def _frame_signal(samples: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> np.ndarray:
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    if samples.size < frame_len:
        return samples.reshape(1, -1) if samples.size else np.zeros((0, frame_len))
    n_frames = 1 + (samples.size - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return samples[idx]


def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    if rms <= 1e-9:
        return -120.0
    return 20.0 * float(np.log10(rms))


def assess_quality(samples: np.ndarray, sample_rate: int) -> QualityReport:
    duration_s = samples.size / float(sample_rate) if sample_rate else 0.0
    reasons: list[str] = []

    if samples.size == 0 or duration_s < MIN_DURATION_INSUFFICIENT_S:
        return QualityReport(
            label="insufficient",
            duration_s=duration_s,
            voiced_duration_s=0.0,
            voiced_ratio=0.0,
            rms_dbfs=-120.0,
            snr_db=0.0,
            clipping_ratio=0.0,
            reasons=[f"clip too short ({duration_s:.2f}s < {MIN_DURATION_INSUFFICIENT_S}s)"],
        )

    overall_rms_dbfs = _rms_dbfs(samples)
    clipping_ratio = float(np.mean(np.abs(samples) >= CLIPPING_AMPLITUDE))

    frames = _frame_signal(samples, sample_rate, FRAME_MS, HOP_MS)
    if frames.shape[0] == 0:
        frame_energies_db = np.array([overall_rms_dbfs])
    else:
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        frame_energies_db = 20.0 * np.log10(np.clip(frame_rms, 1e-9, None))

    noise_floor_db = float(np.percentile(frame_energies_db, 10))
    speech_level_db = float(np.percentile(frame_energies_db, 90))
    snr_db = max(0.0, speech_level_db - noise_floor_db)

    voice_threshold_db = noise_floor_db + 6.0
    voiced_mask = frame_energies_db >= voice_threshold_db
    voiced_ratio = float(np.mean(voiced_mask)) if voiced_mask.size else 0.0
    hop_s = HOP_MS / 1000.0
    voiced_duration_s = voiced_ratio * duration_s if frames.shape[0] else 0.0
    # more precise: count of voiced frames * hop, capped at duration
    if frames.shape[0]:
        voiced_duration_s = min(duration_s, float(np.sum(voiced_mask)) * hop_s)

    if overall_rms_dbfs < SILENCE_RMS_DBFS:
        reasons.append(f"near-silent input ({overall_rms_dbfs:.1f} dBFS)")
    if voiced_duration_s < MIN_VOICED_DURATION_INSUFFICIENT_S:
        reasons.append(f"too little detected voice activity ({voiced_duration_s:.2f}s)")
    if snr_db < SNR_DB_INSUFFICIENT:
        reasons.append(f"estimated SNR too low to trust ({snr_db:.1f} dB)")

    if reasons:
        label = "insufficient"
    else:
        degraded_reasons: list[str] = []
        if snr_db < SNR_DB_DEGRADED:
            degraded_reasons.append(f"noisy background (est. SNR {snr_db:.1f} dB)")
        if clipping_ratio > CLIPPING_RATIO_DEGRADED:
            degraded_reasons.append(f"clipping detected ({clipping_ratio * 100:.1f}% of samples)")
        if voiced_ratio < VOICED_RATIO_DEGRADED:
            degraded_reasons.append(f"low speech-activity ratio ({voiced_ratio:.2f})")
        if degraded_reasons:
            label = "degraded"
            reasons = degraded_reasons
        else:
            label = "good"

    return QualityReport(
        label=label,
        duration_s=duration_s,
        voiced_duration_s=voiced_duration_s,
        voiced_ratio=voiced_ratio,
        rms_dbfs=overall_rms_dbfs,
        snr_db=snr_db,
        clipping_ratio=clipping_ratio,
        reasons=reasons,
    )
