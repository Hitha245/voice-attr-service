# Design write-up

I used `audeering/wav2vec2-large-robust-24-ft-age-gender`, a wav2vec2-large
encoder fine-tuned for age regression and 3-class gender (female/male/child),
with public weights on Hugging Face. I chose it over a general ASR/embedding
model repurposed for this task because it's already trained on the exact
target variables, its "robust" pretraining spans varied recording conditions
(relevant to truck-cab/warehouse noise), and it runs comfortably under the
500ms/5s-clip budget on CPU — no GPU needed for the portability constraint.
A pure-DSP (autocorrelation pitch) fallback backs it up if the model fails
to load, with confidence capped low so degraded-mode answers are never
mistaken for calibrated ones. Audio quality (SNR, clipping, voice-activity
ratio) gates inference before the model runs, rather than trusting a
confident-looking wrong answer on noisy audio.

With more time: validate and recalibrate the age-bracket confidence heuristic
against real labeled data (Common Voice) via the included eval harness, since
it's currently a boundary-distance approximation, not a trained calibration;
add a proper accent/dialect classifier; and support incremental Opus decoding
for browser-mic streaming clients.

To scale to 1,000 concurrent calls: horizontally replicate the container
(CPU-bound inference makes the container, not the process, the scaling
unit), share model weights via a read-only volume so replicas skip
cold-downloading, put a load balancer with WebSocket session affinity in
front so a call's frames keep routing to one replica, and consider a
dedicated batching inference server (e.g. Triton) if replica count grows
unwieldy.
