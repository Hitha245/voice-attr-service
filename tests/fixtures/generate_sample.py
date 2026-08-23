"""Generates tests/fixtures/synthetic_voice_sample.wav — a formant-synthesized,
speech-*like* signal used purely to smoke-test the pipeline mechanics (audio
decode -> quality gate -> inference -> response shape) without requiring a
network download in CI.

This is NOT real speech and should not be used to judge model accuracy —
see eval/README.md for how to point the eval harness at real labeled speech
(Mozilla Common Voice) for that.

Run: python tests/fixtures/generate_sample.py
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 16000
OUT_PATH = Path(__file__).parent / "synthetic_voice_sample.wav"


def synth_voice(f0: float, duration_s: float, seed: int = 0) -> np.ndarray:
    """Glottal-pulse-train-ish synthesis: a harmonic series through a crude
    formant filter, amplitude-modulated to mimic syllable rhythm, plus a
    touch of breath noise. Close enough to exercise VAD/pitch-tracking code
    paths on something that isn't a pure sine tone.
    """
    rng = np.random.default_rng(seed)
    n = int(SR * duration_s)
    t = np.arange(n) / SR

    # Harmonic series (glottal pulse approximation) with formant-like
    # emphasis around ~700Hz and ~1200Hz (roughly vowel-like).
    sig = np.zeros(n)
    harmonics = np.arange(1, 20)
    for h in harmonics:
        freq = f0 * h
        if freq > SR / 2:
            break
        formant_gain = np.exp(-((freq - 700) ** 2) / (2 * 300**2)) + 0.6 * np.exp(
            -((freq - 1200) ** 2) / (2 * 400**2)
        )
        amp = (1.0 / h) * (0.3 + formant_gain)
        sig += amp * np.sin(2 * np.pi * freq * t)

    # Syllable-rate amplitude envelope (~4 Hz) with brief pauses, so the
    # quality/VAD code sees realistic voiced/unvoiced alternation.
    syllable_env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t - np.pi / 2)
    syllable_env = np.clip(syllable_env, 0.0, 1.0) ** 1.5
    sig *= syllable_env

    # Light breath/room noise floor.
    sig += 0.01 * rng.standard_normal(n)

    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.6
    return sig.astype(np.float32)


def main() -> None:
    # ~120 Hz median F0: falls on the "male" side of the fallback
    # heuristic's decision boundary, useful for a deterministic smoke test.
    sig = synth_voice(f0=120.0, duration_s=4.0, seed=42)
    pcm16 = (sig * 32767).astype("<i2")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT_PATH), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.tobytes())

    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes, {len(sig)/SR:.1f}s @ {SR}Hz)")


if __name__ == "__main__":
    main()
