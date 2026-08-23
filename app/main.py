"""FastAPI application: REST + WebSocket surface for the voice attribute
inference service.

Endpoints:
  POST /analyze        - multipart audio upload -> AnalyzeResponse
  WS   /ws/analyze      - chunked/streaming audio -> progressive AnalyzeResponse messages
  GET  /health          - liveness/readiness + model status
  GET  /                - basic service info
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app.audio_io import AudioDecodeError, decode_audio_bytes, decode_raw_pcm16
from app.config import settings
from app.inference.age_gender_model import AgeGenderModel
from app.inference.pipeline import AnalysisResult, analyze_samples
from app.logging_conf import configure_logging
from app.schemas import (
    AgeBracketResult,
    AnalyzeResponse,
    GenderResult,
    HealthResponse,
    LanguageResult,
)

configure_logging()
logger = logging.getLogger("voiceattr.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.PRELOAD_MODEL:
        try:
            logger.info("preloading model at startup...")
            AgeGenderModel.get().ensure_loaded()
        except Exception:  # noqa: BLE001
            # Don't crash the whole service if the model can't load at boot
            # (e.g. no network yet, registry hiccup) - /analyze will retry
            # the load lazily and fall back to the heuristic in the
            # meantime. See app/inference/pipeline.py.
            logger.exception("model preload failed; will retry lazily / use fallback")
    yield


app = FastAPI(
    title="Voice Attribute Inference Service",
    description="Estimates caller gender and age bracket from a short audio clip.",
    version="1.0.0",
    lifespan=lifespan,
)


def _result_to_response(result: AnalysisResult) -> AnalyzeResponse:
    language = None
    if result.language_prediction is not None:
        language = LanguageResult(
            prediction=result.language_prediction,
            confidence=result.language_confidence or 0.0,
        )
    return AnalyzeResponse(
        contact_id=result.contact_id,
        gender=GenderResult(prediction=result.gender_prediction, confidence=result.gender_confidence),
        age_bracket=AgeBracketResult(prediction=result.age_prediction, confidence=result.age_confidence),
        processing_ms=result.processing_ms,
        audio_quality=result.audio_quality,
        language=language,
        model_backend=result.model_backend,
        quality_reasons=result.quality_reasons,
        warnings=result.warnings,
    )


@app.middleware("http")
async def timing_and_request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.monotonic()
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled exception", extra={"request_id": request_id})
        raise
    elapsed_ms = int((time.monotonic() - start) * 1000)
    response.headers["x-request-id"] = request_id
    response.headers["x-processing-ms"] = str(elapsed_ms)
    logger.info(
        "%s %s -> %s",
        request.method,
        request.url.path,
        response.status_code,
        extra={"request_id": request_id, "processing_ms": elapsed_ms},
    )
    return response


@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "voice-attribute-inference",
        "endpoints": ["/analyze (POST)", "/ws/analyze (WebSocket)", "/health (GET)"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    model = AgeGenderModel.get()
    status = "ok" if (model.is_loaded or settings.ENABLE_MODEL_FALLBACK) else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=model.is_loaded,
        model_backend="wav2vec2-age-gender" if model.is_loaded else "heuristic-fallback",
        model_load_error=model.load_error,
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["inference"])
async def analyze(
    request: Request,
    audio: UploadFile = File(..., description="Audio file: wav, mp3, m4a, ogg/opus, webm, flac, ..."),
    contact_id: str | None = Query(default=None, description="Optional caller-supplied UUID; generated if omitted"),
):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    cid = contact_id or request_id

    raw_bytes = await audio.read()
    # `raw_bytes` and everything derived from it live only in this
    # function's local scope / the pipeline's local scope; nothing is
    # written to disk and nothing is cached across requests (see README >
    # Privacy). The reference is dropped as soon as this handler returns.
    try:
        decoded = decode_audio_bytes(raw_bytes)
    except AudioDecodeError as exc:
        logger.warning("decode failure: %s", exc, extra={"request_id": cid})
        raise HTTPException(status_code=422, detail=f"could not decode audio: {exc}") from exc
    finally:
        del raw_bytes

    try:
        result = analyze_samples(decoded.samples, decoded.sample_rate, contact_id=cid)
    except Exception as exc:  # noqa: BLE001
        logger.exception("analysis failed", extra={"request_id": cid})
        raise HTTPException(status_code=500, detail="internal error during analysis") from exc

    return _result_to_response(result)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("unhandled exception in %s", request.url.path, extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "an unexpected error occurred", "request_id": request_id},
    )


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------
#
# Protocol (kept intentionally simple — this is the bonus real-time task,
# not a full telephony-codec implementation; see README > Limitations for
# what a production version would add, e.g. native Opus/mu-law decoding):
#
#   1. Client connects to /ws/analyze?contact_id=...&sample_rate=16000
#   2. Client sends binary WebSocket frames containing raw 16-bit
#      little-endian mono PCM audio at the declared sample_rate.
#   3. Every WS_INFERENCE_WINDOW_SECONDS worth of *accumulated* audio, the
#      server runs inference on the buffer so far and pushes a progressive
#      JSON AnalyzeResponse back — confidence should trend up as more audio
#      arrives.
#   4. Client sends the text message "end" (or just closes the socket) to
#      finish; the server runs one final pass on the full buffer and closes.
#
@app.websocket("/ws/analyze")
async def ws_analyze(
    websocket: WebSocket,
    contact_id: str | None = Query(default=None),
    sample_rate: int = Query(default=16000, ge=4000, le=48000),
):
    await websocket.accept()
    cid = contact_id or str(uuid.uuid4())
    buffer = bytearray()
    session_start = time.monotonic()
    window_bytes = int(settings.WS_INFERENCE_WINDOW_SECONDS * sample_rate * 2)  # 16-bit mono
    bytes_since_last_infer = 0

    logger.info("ws session opened", extra={"request_id": cid})

    async def run_inference_on_buffer(final: bool) -> None:
        if not buffer:
            return
        pcm_bytes = bytes(buffer)
        decoded = decode_raw_pcm16(pcm_bytes, sample_rate=sample_rate)
        result = analyze_samples(decoded.samples, decoded.sample_rate, contact_id=cid)
        payload = _result_to_response(result).model_dump()
        payload["is_final"] = final
        await websocket.send_json(payload)

    try:
        while True:
            if time.monotonic() - session_start > settings.WS_MAX_SESSION_SECONDS:
                await websocket.send_json({"error": "session_timeout", "contact_id": cid})
                break

            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                buffer.extend(message["bytes"])
                bytes_since_last_infer += len(message["bytes"])
                if bytes_since_last_infer >= window_bytes:
                    bytes_since_last_infer = 0
                    try:
                        await run_inference_on_buffer(final=False)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("streaming inference failed", extra={"request_id": cid})
                        await websocket.send_json({"error": "inference_failed", "detail": str(exc)})

            elif "text" in message and message["text"] is not None:
                text = message["text"].strip().lower()
                if text in ("end", "stop", "close"):
                    try:
                        await run_inference_on_buffer(final=True)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("final streaming inference failed", extra={"request_id": cid})
                        await websocket.send_json({"error": "inference_failed", "detail": str(exc)})
                    break

    except WebSocketDisconnect:
        logger.info("ws client disconnected", extra={"request_id": cid})
    finally:
        buffer.clear()  # drop audio immediately; nothing persisted (see README > Privacy)
        logger.info("ws session closed", extra={"request_id": cid})
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed
