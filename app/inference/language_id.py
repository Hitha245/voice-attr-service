"""Bonus: best-effort language identification.

Uses OpenAI Whisper's own language-ID head (the log-mel spectrogram fed
through the encoder once, then a linear probe over Whisper's 99-language
vocabulary) rather than transcribing the audio. This gives a language
guess for ~1s of audio without the latency/cost of full ASR, and piggybacks
on a well-known public model per the assignment's "publicly available model
weights" constraint.

Disabled by default (ENABLE_LANGUAGE_ID=false) because it adds a second
model's worth of memory/first-load latency for a bonus field the core
contract doesn't require; flip it on via env var when you want the
`language` field populated.

Note: this identifies *language*, not regional accent. Robust accent
classification (e.g. US-South vs. UK-RP vs. Indian English) needs a
dedicated accent-ID corpus/model (e.g. a model trained on CommonVoice/
VoxLingua107 accent labels) — flagged as a "would improve with more time"
item in the README rather than implemented here, to avoid shipping an
accent classifier that hasn't actually been validated.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from app.config import settings

logger = logging.getLogger("voiceattr.language_id")


class LanguageIdUnavailable(Exception):
    pass


class LanguageIdentifier:
    _instance: "LanguageIdentifier | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._model = None
        self._loaded = False

    @classmethod
    def get(cls) -> "LanguageIdentifier":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                import whisper
            except ImportError as exc:
                raise LanguageIdUnavailable(f"openai-whisper not installed: {exc}") from exc
            logger.info("loading whisper '%s' for language ID", settings.WHISPER_LANGID_MODEL)
            self._whisper = whisper
            self._model = whisper.load_model(settings.WHISPER_LANGID_MODEL, device="cpu")
            self._loaded = True

    def identify(self, samples: np.ndarray, sample_rate: int) -> tuple[str, float]:
        """Return (ISO 639-1 language code, confidence) for the loudest
        detected language, e.g. ("en", 0.91)."""
        if sample_rate != 16000:
            raise ValueError("language_id.identify expects 16kHz audio")
        self.ensure_loaded()
        whisper = self._whisper

        audio = whisper.pad_or_trim(samples.astype(np.float32))
        mel = whisper.log_mel_spectrogram(audio, n_mels=self._model.dims.n_mels).to(self._model.device)
        _, probs = self._model.detect_language(mel)
        lang = max(probs, key=probs.get)
        return lang, float(probs[lang])


def identify_language_safe(samples: np.ndarray, sample_rate: int) -> tuple[str | None, float | None]:
    """Never raises: returns (None, None) if language ID is disabled,
    unavailable, or fails for any reason. Language ID is a bonus/best-effort
    field and must never take down the core gender/age response.
    """
    if not settings.ENABLE_LANGUAGE_ID:
        return None, None
    try:
        lang, conf = LanguageIdentifier.get().identify(samples, sample_rate)
        return lang, round(conf, 4)
    except Exception:  # noqa: BLE001
        logger.exception("language ID failed; omitting from response")
        return None, None
