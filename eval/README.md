# Eval harness

`run_eval.py` runs the full inference pipeline (the same code path `/analyze`
uses) against a labeled dataset and prints gender/age accuracy plus a
confidence-calibration score (Expected Calibration Error).

## Sourcing labeled data: Mozilla Common Voice

1. Go to https://commonvoice.mozilla.org/en/datasets and download a recent
   release for a language of your choice (English is easiest to sanity
   check). You'll need to accept the license terms; downloads are free.
2. Unpack the archive. You want:
   - `validated.tsv` (or `other.tsv`) — the manifest. It has a `path`
     column (clip filename), a `gender` column (`male_masculine`,
     `female_feminine`, or blank depending on release version — older
     releases just say `male`/`female`), and an `age` column
     (`teens`, `twenties`, ... `nineties`).
   - `clips/` — the actual `.mp3` files the manifest's `path` column refers
     to.
3. Not every row has age/gender filled in (it's self-reported and
   optional) — `run_eval.py` skips rows where both are blank.

## Running it

```bash
pip install -r requirements.txt   # or run inside the Docker image
python eval/run_eval.py \
    --manifest /path/to/cv-corpus/en/validated.tsv \
    --clips-dir /path/to/cv-corpus/en/clips \
    --limit 200
```

`--limit 0` evaluates the whole manifest (slow for the full corpus —
Common Voice validated sets run into the hundreds of thousands of clips).

## What it prints

- **Gender / age bracket accuracy** against Common Voice's self-reported
  labels. Note Common Voice's gender field is self-reported identity, not a
  vocal-acoustic ground truth, and its age buckets are decade-wide, so this
  is a reasonable but imperfect proxy for "did the model get it right" —
  see the design write-up for the general point that fair benchmarking of
  gender/age classifiers is a genuinely hard, actively-debated problem.
- **Expected Calibration Error (ECE)** — bins predictions by confidence and
  compares each bin's average confidence to its actual accuracy. A
  well-calibrated 0.9-confidence prediction should be right about 90% of
  the time; ECE near 0 means the confidence scores in the API response can
  actually be trusted as probabilities, not just a relative ranking.
- **Audio quality mix / backend used** — how many clips got flagged
  degraded/insufficient, and whether the primary model or the fallback
  heuristic answered them; useful for sanity-checking that the fallback
  isn't silently doing most of the work.
- **Latency p50/p95** — end-to-end `analyze_samples()` time per clip.

## A quick smoke run without downloading anything

`tests/fixtures/generate_sample.py` makes a synthetic (non-real-speech) wav
purely to exercise the harness's mechanics — decoding, calling the
pipeline, computing the accuracy/ECE math — without needing real labeled
audio:

```bash
python tests/fixtures/generate_sample.py
mkdir -p eval/data/clips
cp tests/fixtures/synthetic_voice_sample.wav eval/data/clips/
cat > eval/data/manifest.csv <<'EOF'
path,gender,age
synthetic_voice_sample.wav,male,twenties
EOF
python eval/run_eval.py --manifest eval/data/manifest.csv --clips-dir eval/data/clips
```

This is a *mechanics* smoke test only (n=1, synthetic audio) — it tells you
the harness runs end to end, not that the model is accurate. Use real
Common Voice data for an actual accuracy number.
