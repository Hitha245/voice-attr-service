# Voice Attribute Inference Service

A backend service that estimates a caller's **gender** and **age bracket**
from a short audio clip, built for voice AI agents handling logistics calls
(inbound/outbound, driver/dispatcher/customer coordination). No prior data
about the contact is required — everything is inferred from the audio
itself.

```
POST /analyze  ->  { contact_id, gender, age_bracket, processing_ms, audio_quality, ... }
```

## Quickstart

```bash
docker compose up --build
```

First boot downloads the model weights (~1.2 GB, cached in a named Docker
volume so subsequent restarts are fast) and then serves on
`http://localhost:8000`. No other external dependency is required.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "audio=@tests/fixtures/synthetic_voice_sample.wav"
```

```bash
curl http://localhost:8000/health
```

Interactive API docs (Swagger UI) are at `http://localhost:8000/docs` once
the container is running.

## Running the tests

```bash
pip install -r requirements.txt
python tests/fixtures/generate_sample.py   # writes the smoke-test wav (also checked in)
pytest -v
```

`tests/test_quality.py` and `tests/test_fallback_heuristic.py` only need
numpy/scipy and run anywhere. `tests/test_pipeline.py` exercises the full
decode -> quality -> inference path (falls back to the DSP heuristic if
torch/transformers aren't installed, so it also runs in a minimal
environment). `tests/test_api_smoke.py` needs the full stack (fastapi,
torch, transformers) and is the one to run inside Docker or a full venv —
it covers `/health`, `/analyze` (happy path, garbage-audio 422, empty-upload
422, auto-generated contact_id) and a WebSocket streaming round-trip.

## API contract

`POST /analyze` — multipart form upload, field name `audio`; optional
`?contact_id=` query param (a UUID is generated if omitted).

```jsonc
{
  "contact_id": "uuid",
  "gender": { "prediction": "male" | "female" | "unknown", "confidence": 0.87 },
  "age_bracket": { "prediction": "18-30" | "31-45" | "46-60" | "60+" | "unknown", "confidence": 0.63 },
  "processing_ms": 142,
  "audio_quality": "good" | "degraded" | "insufficient",

  // additive fields beyond the required contract:
  "language": { "prediction": "en", "confidence": 0.91 } | null,
  "model_backend": "wav2vec2-age-gender" | "heuristic-fallback",
  "quality_reasons": ["noisy background (est. SNR 8.1 dB)"],
  "warnings": []
}
```

`WS /ws/analyze?sample_rate=16000&contact_id=...` — streaming variant, see
"Streaming design" below.

`GET /health` — liveness/readiness + whether the primary model is loaded.

## Architecture

```
raw bytes (any codec ffmpeg supports)
        |
        v
  app/audio_io.py   -- ffmpeg subprocess over pipes, decode -> mono 16kHz f32 PCM, in-memory only
        |
        v
  app/quality.py    -- energy-based VAD + SNR estimate + clipping check -> good/degraded/insufficient
        |
        v
  app/inference/pipeline.py
        |
        +-- insufficient? --> short-circuit: "unknown" + reasons, skip model entirely
        |
        +-- app/inference/age_gender_model.py   (primary: audeering wav2vec2, CPU)
        |         |
        |         +-- fails to load/predict? --> app/inference/fallback_heuristic.py (numpy/scipy DSP)
        |
        +-- app/inference/language_id.py  (bonus, off by default)
        |
        v
  app/schemas.py -> JSON response
```

`app/main.py` is a thin FastAPI layer over `pipeline.py`: a timing/
request-ID middleware, the `/analyze` and `/ws/analyze` routes, and a
top-level exception handler so a bad request degrades to a clean 4xx/5xx
JSON body instead of a stack trace.

## Model choice & rationale

**Primary: `audeering/wav2vec2-large-robust-24-ft-age-gender`**
(Hugging Face Hub, MIT-adjacent research license — publicly available
weights, downloaded at first container boot).

- It's a wav2vec2-large encoder ("robust" pretraining — trained across
  multiple corpora with varied recording conditions, not just clean studio
  speech) fine-tuned with two heads: a 1-value age regression and a
  3-class (female/male/child) gender classification. That's a purpose-built
  fit for this task, versus e.g. repurposing a speech-emotion or ASR model.
- Ships ready-to-use weights — no training data or GPU cluster of our own
  needed, which matches the "use pretrained models" guidance and the
  portability constraint (only dependency beyond the repo is a public
  model download).
- Runs on CPU in well under the 500ms/5s-clip latency target once warm (the
  Wav2Vec2 forward pass on a 5s clip is a few hundred ms on a modern CPU
  core); no GPU requirement keeps the Docker Compose setup simple.
- The "child" class and a confidence floor are mapped to `"unknown"` in the
  API response rather than forced into male/female — a wrong-but-confident
  gender guess is worse than an honest "unknown" in a customer-facing call.
- Age is a continuous regression (`age/100`) internally; we bucket it into
  the four required brackets and derive a confidence heuristic from how far
  the point estimate sits from the nearest bracket boundary (see
  `age_gender_model.py` docstring — this is an explicit approximation, the
  model wasn't trained to expose calibrated bracket confidence directly).

**Fallback: pure-DSP heuristic (`app/inference/fallback_heuristic.py`)**

If the transformer model can't load — no network yet on first boot, a
corrupted cache, an out-of-memory low-spec instance — the service does not
500 every request. It falls back to autocorrelation-based pitch (F0)
tracking with numpy/scipy only, classifying gender via a logistic function
of median F0 (adult male/female speaking-F0 distributions are fairly
well-separated, with the ambiguous overlap band mapped to `"unknown"`
rather than a coin flip) and a weak jitter-based age heuristic. Its
confidence is explicitly capped (0.75 gender / 0.40 age ceilings) well
below the primary model's typical range, so a caller reading `confidence`
can't mistake a degraded-mode answer for a calibrated one. This is Task 4's
"handle errors gracefully" applied to the model layer specifically, not
just to malformed HTTP requests.

## Audio quality handling

Rather than silently emitting a confident-looking prediction on unusable
audio, `app/quality.py` frames the signal, estimates a noise floor vs.
speech-level SNR (10th/90th percentile of per-frame energy), a
voice-activity ratio, and a clipping ratio, and buckets the result:

- **insufficient** — clip too short, near-silent, or too little detected
  voice activity / SNR to trust anything. The pipeline **skips model
  inference entirely** and returns `"unknown"` with the reasons listed —
  spending a model forward pass on unusable audio just to get a
  meaningless answer isn't worth the latency.
- **degraded** — noisy but usable (truck cab, warehouse floor, road noise —
  the logistics conditions called out in the assignment). Inference still
  runs, but returned confidence scores are discounted by 30% to reflect the
  added uncertainty, and the specific reasons (SNR, clipping %, voice-
  activity ratio) are surfaced in `quality_reasons` for observability.
- **good** — proceeds normally.

## Streaming design (`/ws/analyze`)

Protocol: client connects to `/ws/analyze?sample_rate=16000&contact_id=...`,
then sends binary frames of raw 16-bit little-endian mono PCM audio at the
declared sample rate (this matches how telephony/voice-agent audio
typically already arrives — e.g. Twilio Media Streams sends raw PCM/µ-law
frames rather than a container format). Every `WS_INFERENCE_WINDOW_SECONDS`
(default 2s) of *accumulated* audio, the server re-runs inference on the
whole buffer so far and pushes a progressive JSON prediction back — you
should see confidence trend upward as more audio arrives. The client sends
the text message `"end"` (or just closes the socket) to trigger one final
pass, flagged `"is_final": true`.

This intentionally does *not* implement stateful decoding of a compressed
streaming codec (e.g. incremental Opus/WebM demuxing) — seeded PCM keeps
the streaming path simple and low-latency, and is the realistic wire format
for a telephony-integrated voice agent. A browser-microphone client would
need to downsample/encode to raw PCM16 client-side (or the server would
need a stateful Opus decoder) — noted under Known Limitations.

## Reliability & observability (Task 4)

- **Structured logging** (`app/logging_conf.py`): JSON logs by default
  (`LOG_JSON=true`), every request/analysis carries a `request_id` and
  `processing_ms`. Set `LOG_JSON=false` for readable local-dev output.
- **Timing**: a middleware wraps every HTTP request with start/elapsed
  timing and echoes `x-request-id` / `x-processing-ms` response headers;
  `processing_ms` in the JSON body specifically times the decode+quality+
  inference pipeline (not HTTP overhead).
- **Graceful degradation** at three layers: audio decode failure -> `422`
  with a decode-error detail (not a 500); model load/inference failure ->
  falls back to the DSP heuristic rather than failing the request;
  streaming inference failure on one window -> an `error` message over the
  socket, session stays open rather than dropping.
- **Health check**: `GET /health` reports whether the primary model loaded,
  its backend name, and the load error if any; wired into the Docker
  `HEALTHCHECK` / compose healthcheck.
- **Concurrency guard**: a bounded semaphore
  (`INFERENCE_CONCURRENCY`, default = CPU core count) caps concurrent model
  forward passes so a burst of requests queues instead of thrashing CPU
  threads against each other.

## Privacy

Caller audio is treated as PII end to end:

- **No disk writes, ever.** `app/audio_io.py` decodes via an `ffmpeg`
  subprocess connected by OS pipes (`stdin`/`stdout`) — the raw bytes and
  the decoded PCM array both live only in process memory. There is no
  temp-file step anywhere in the pipeline, batch or streaming.
- **No persistence across requests.** Each `/analyze` call's audio buffer
  goes out of scope (and is garbage-collected) as soon as the response is
  returned; the WebSocket handler explicitly `buffer.clear()`s on session
  end. Nothing is cached, logged, or written to a database.
- **Logs never contain audio.** Only metadata (duration, quality label,
  reasons, byte size) and results (predictions, confidences, timings) are
  logged — see `app/inference/pipeline.py`'s log calls.
- **What *is* cached**: only the model *weights* themselves (in the named
  `hf-cache` Docker volume), which contain no caller data.
- If you deploy this behind a real telephony/voice-agent pipeline, apply
  the same discipline upstream (don't log raw call audio, don't proxy it
  through anything that persists it) — this service's guarantee only
  covers what happens inside its own process boundary.

## Known limitations

- **Age accuracy is inherently the weaker half of this task.** Age
  regression from voice alone has much higher acoustic variance than
  gender, and the primary model's age head was fine-tuned on
  crowd-sourced, self-reported labels (imperfect ground truth in its own
  right). The bracket-boundary-distance confidence heuristic is a
  reasonable approximation, not a property the model was trained to
  expose — treat `age_bracket.confidence` as directional, not a calibrated
  probability, until validated against real data (see `eval/`).
- **Gender is binary + "child"/"unknown" only** — the underlying model's
  training data doesn't support a broader gender taxonomy, and there's no
  reliable acoustic proxy for gender identity beyond what the model
  captures; `"unknown"` is the honest answer whenever the model's own
  gender-probability margin is thin.
- **Streaming codec support is PCM16-only** (see "Streaming design" above)
  — no in-place Opus/WebM decoding for the WebSocket path.
- **No real accuracy number is reported here** — the eval harness
  (`eval/run_eval.py`) is built and unit-verified on synthetic data, but
  turning it against real Common Voice data (see `eval/README.md`) requires
  a dataset download this environment didn't have network access to
  perform; running it is the natural next step before trusting this in
  production.
- **Single language ID model, no accent classification** — `language_id.py`
  is best-effort language (not regional accent) detection via Whisper's
  built-in language head, off by default; a dedicated accent classifier
  would need its own labeled corpus and is flagged as future work rather
  than shipped unvalidated.
- **No authentication/rate-limiting** on the API itself — assumed to sit
  behind an internal gateway/service mesh in a real deployment; out of
  scope for this assignment's surface area.

## Scaling to 1,000 concurrent calls

See the design write-up below for the full answer; short version: this
process is CPU-bound per inference, so the unit of horizontal scale is the
*container*, not the in-process worker count — run N replicas behind a load
balancer (each replica handles a slice of concurrent streams, gated by
`INFERENCE_CONCURRENCY`), keep the model weights on a shared read-only
cache/volume so replicas don't each cold-download, and move the WebSocket
session affinity to a layer that can route a given call's frames back to
the same replica for the life of the call (or make sessions stateless by
handing the accumulated-buffer state to a shared store, at added latency
cost).
