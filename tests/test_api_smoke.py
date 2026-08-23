"""HTTP-layer smoke tests using FastAPI's TestClient.

Requires the full requirements.txt (fastapi, httpx, torch, transformers) to
be installed, so run these inside the Docker image or a local venv with
`pip install -r requirements.txt`:

    pytest tests/test_api_smoke.py -v

(They are *not* runnable in a stripped-down environment that only has numpy/
scipy — see tests/test_pipeline.py and tests/test_quality.py for the subset
of the suite that only needs those.)
"""
from __future__ import annotations

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture()
def client():
    from app.main import app

    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "model_loaded" in body


def test_analyze_returns_contract_shaped_response(client, sample_wav_bytes):
    resp = client.post(
        "/analyze",
        files={"audio": ("sample.wav", sample_wav_bytes, "audio/wav")},
        params={"contact_id": "test-contact-123"},
    )
    assert resp.status_code == 200
    body = resp.json()

    # Exact keys from the assignment's API contract must be present.
    assert body["contact_id"] == "test-contact-123"
    assert set(body["gender"].keys()) == {"prediction", "confidence"}
    assert set(body["age_bracket"].keys()) == {"prediction", "confidence"}
    assert body["gender"]["prediction"] in {"male", "female", "unknown"}
    assert body["age_bracket"]["prediction"] in {"18-30", "31-45", "46-60", "60+", "unknown"}
    assert 0.0 <= body["gender"]["confidence"] <= 1.0
    assert isinstance(body["processing_ms"], int)
    assert body["audio_quality"] in {"good", "degraded", "insufficient"}


def test_analyze_rejects_garbage_audio_with_422(client):
    resp = client.post(
        "/analyze",
        files={"audio": ("not_audio.wav", b"this is not a real audio file", "audio/wav")},
    )
    assert resp.status_code == 422


def test_analyze_rejects_empty_upload(client):
    resp = client.post(
        "/analyze",
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 422


def test_analyze_generates_contact_id_when_not_supplied(client, sample_wav_bytes):
    resp = client.post("/analyze", files={"audio": ("sample.wav", sample_wav_bytes, "audio/wav")})
    assert resp.status_code == 200
    assert resp.json()["contact_id"]  # non-empty


def test_websocket_streaming_smoke(client, sample_wav_bytes):
    import wave
    import io

    # Extract raw PCM16 frames from the bundled WAV to stream over the socket.
    with wave.open(io.BytesIO(sample_wav_bytes)) as w:
        sr = w.getframerate()
        pcm = w.readframes(w.getnframes())

    with client.websocket_connect(f"/ws/analyze?sample_rate={sr}&contact_id=ws-test") as ws:
        chunk_size = 4096
        for i in range(0, len(pcm), chunk_size):
            ws.send_bytes(pcm[i : i + chunk_size])
        ws.send_text("end")
        received_final = False
        for _ in range(10):
            msg = ws.receive_json()
            if msg.get("is_final"):
                received_final = True
                assert msg["contact_id"] == "ws-test"
                break
        assert received_final
