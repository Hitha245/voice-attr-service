from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

SR = 16000
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_wav_bytes() -> bytes:
    """The bundled synthetic smoke-test clip (generated once at session
    scope, and also checked into the repo so tests don't depend on
    generation order)."""
    path = FIXTURES_DIR / "synthetic_voice_sample.wav"
    if not path.exists():
        import subprocess
        import sys

        subprocess.run([sys.executable, str(FIXTURES_DIR / "generate_sample.py")], check=True)
    return path.read_bytes()


def make_tone_wav_bytes(freq_hz: float, duration_s: float = 2.0, sr: int = SR, amp: float = 0.4) -> bytes:
    """Small helper for tests that need an ad-hoc WAV without touching disk."""
    import io

    t = np.arange(int(sr * duration_s)) / sr
    env = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t), 0.0, 1.0)
    sig = amp * np.sin(2 * np.pi * freq_hz * t) * env
    pcm16 = (sig * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()
