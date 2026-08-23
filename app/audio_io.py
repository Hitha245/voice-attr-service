"""Audio decoding utilities.

Design goal: accept *any* container/codec ffmpeg understands (wav, mp3, m4a/aac,
ogg/opus, webm/opus, flac, mu-law telephony wav, ...) and normalize it to mono
16 kHz float32 PCM for the model, without ever writing the caller's audio to
disk. ffmpeg is invoked as a subprocess with the input piped in on stdin and
the decoded PCM piped out on stdout; both buffers live only in process memory
and are garbage-collected once the request finishes (see README > Privacy).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

from app.config import settings


class AudioDecodeError(Exception):
    """Raised when the input bytes cannot be decoded into PCM audio."""


@dataclass
class DecodedAudio:
    samples: np.ndarray  # float32, mono, range [-1, 1]
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / float(self.sample_rate)


def decode_audio_bytes(raw_bytes: bytes, *, timeout_s: float | None = None) -> DecodedAudio:
    """Decode arbitrary compressed/container audio bytes to mono 16 kHz PCM.

    Raises AudioDecodeError on empty input, ffmpeg failure, or timeout (e.g.
    a corrupt file, or a codec ffmpeg can't parse). Never touches the
    filesystem: input and output are both connected via pipes.
    """
    if not raw_bytes:
        raise AudioDecodeError("received empty audio payload")

    if len(raw_bytes) > settings.MAX_UPLOAD_BYTES:
        raise AudioDecodeError(
            f"audio payload of {len(raw_bytes)} bytes exceeds the "
            f"{settings.MAX_UPLOAD_BYTES} byte limit"
        )

    timeout_s = timeout_s if timeout_s is not None else settings.FFMPEG_TIMEOUT_SECONDS
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", "pipe:0",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(settings.TARGET_SAMPLE_RATE),
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=raw_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(f"audio decode timed out after {timeout_s}s") from exc
    except FileNotFoundError as exc:
        raise AudioDecodeError("ffmpeg binary not found on PATH") from exc

    if proc.returncode != 0 or not proc.stdout:
        stderr_tail = proc.stderr.decode("utf-8", errors="ignore").strip()[-500:]
        raise AudioDecodeError(f"ffmpeg could not decode audio: {stderr_tail or 'unknown error'}")

    pcm_i16 = np.frombuffer(proc.stdout, dtype="<i2")
    if pcm_i16.size == 0:
        raise AudioDecodeError("decoded audio contained zero samples")

    samples = pcm_i16.astype(np.float32) / 32768.0
    audio = DecodedAudio(samples=samples, sample_rate=settings.TARGET_SAMPLE_RATE)

    if audio.duration_seconds > settings.MAX_AUDIO_SECONDS:
        # Trim rather than reject outright — a long call recording is a very
        # normal input in the logistics/voice-agent context, we just only
        # need a short window to infer voice attributes.
        max_samples = int(settings.MAX_AUDIO_SECONDS * settings.TARGET_SAMPLE_RATE)
        audio.samples = audio.samples[:max_samples]

    return audio


def decode_raw_pcm16(
    raw_bytes: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
) -> DecodedAudio:
    """Fast path for the WebSocket streaming endpoint: the client is sending
    already-uncompressed 16-bit little-endian PCM frames (the common case for
    telephony/voice-agent audio pipelines, e.g. Twilio Media Streams-style
    raw audio), so we can skip the ffmpeg subprocess entirely for lower
    per-chunk latency.
    """
    if not raw_bytes:
        raise AudioDecodeError("received empty PCM chunk")
    if len(raw_bytes) % 2 != 0:
        # Drop a single dangling byte rather than fail an entire streaming
        # session on a chunk-boundary misalignment.
        raw_bytes = raw_bytes[:-1]
    pcm_i16 = np.frombuffer(raw_bytes, dtype="<i2")
    if channels > 1:
        pcm_i16 = pcm_i16.reshape(-1, channels).mean(axis=1).astype(np.int16)
    samples = pcm_i16.astype(np.float32) / 32768.0

    if sample_rate != settings.TARGET_SAMPLE_RATE:
        samples = _resample_linear(samples, sample_rate, settings.TARGET_SAMPLE_RATE)

    return DecodedAudio(samples=samples, sample_rate=settings.TARGET_SAMPLE_RATE)


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Cheap linear-interpolation resampler for the streaming hot path.

    Good enough for voice-attribute inference (we care about pitch/formant
    envelope, not archival audio fidelity) and avoids pulling librosa/
    scipy.signal.resample_poly into the per-chunk latency budget. The
    batch /analyze path uses ffmpeg's high-quality resampler instead since
    it isn't on a tight per-chunk deadline.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / float(src_rate)
    dst_n = max(1, int(round(duration * dst_rate)))
    src_idx = np.linspace(0, samples.size - 1, num=dst_n, dtype=np.float64)
    return np.interp(src_idx, np.arange(samples.size), samples).astype(np.float32)
