"""Eval harness: run the inference pipeline against a labeled dataset and
report accuracy + confidence calibration for both gender and age bracket.

Designed around Mozilla Common Voice's `validated.tsv`, which carries
self-reported `age` (already bucketed, e.g. "twenties", "fifties") and
`gender` columns for a subset of clips — see eval/README.md for exactly how
to obtain a Common Voice subset (not bundled here: it's a large, separately
licensed dataset).

Usage:
    python eval/run_eval.py \\
        --manifest eval/data/validated.tsv \\
        --clips-dir eval/data/clips \\
        --limit 200

The manifest just needs three columns: a clip path (relative to
--clips-dir), a gender label, and an age label. --gender-col/--age-col/
--path-col let you point at whatever your dataset calls them; sensible
defaults match Common Voice's own column names.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio_io import AudioDecodeError, decode_audio_bytes  # noqa: E402
from app.inference.pipeline import analyze_samples  # noqa: E402

# Common Voice's `age` column uses decade buckets like "twenties"; map them
# onto our API contract's brackets.
COMMON_VOICE_AGE_MAP = {
    "teens": "18-30",
    "twenties": "18-30",
    "thirties": "31-45",
    "fourties": "31-45",  # Common Voice's actual (mis)spelling
    "forties": "31-45",
    "fifties": "46-60",
    "sixties": "46-60",
    "seventies": "60+",
    "eighties": "60+",
    "nineties": "60+",
}


def normalize_gender(raw: str) -> str:
    raw = (raw or "").strip().lower()
    if raw in {"male", "m"}:
        return "male"
    if raw in {"female", "f"}:
        return "female"
    return "unknown"


def normalize_age(raw: str) -> str:
    raw = (raw or "").strip().lower()
    return COMMON_VOICE_AGE_MAP.get(raw, "unknown")


def load_manifest(manifest_path: Path, path_col: str, gender_col: str, age_col: str, limit: int | None):
    delimiter = "\t" if manifest_path.suffix.lower() == ".tsv" else ","
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for row in reader:
            gender = normalize_gender(row.get(gender_col, ""))
            age = normalize_age(row.get(age_col, ""))
            if gender == "unknown" and age == "unknown":
                continue  # unlabeled row, not useful for eval
            rows.append({"path": row[path_col], "gender": gender, "age": age})
            if limit and len(rows) >= limit:
                break
        return rows


def ece(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error: how well confidence scores track actual
    accuracy. 0 = perfectly calibrated, higher = overconfident/underconfident."""
    if not confidences:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, corrects):
        idx = min(n_bins - 1, int(conf * n_bins))
        bins[idx].append(correct)
    total = len(confidences)
    err = 0.0
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        bucket_conf_mid = (i + 0.5) / n_bins
        bucket_acc = sum(bucket) / len(bucket)
        err += (len(bucket) / total) * abs(bucket_conf_mid - bucket_acc)
    return err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="CSV/TSV manifest with path/gender/age columns")
    ap.add_argument("--clips-dir", type=Path, required=True, help="Directory the manifest's paths are relative to")
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--gender-col", default="gender")
    ap.add_argument("--age-col", default="age")
    ap.add_argument("--limit", type=int, default=200, help="Max clips to evaluate (0 = no limit)")
    args = ap.parse_args()

    rows = load_manifest(args.manifest, args.path_col, args.gender_col, args.age_col, args.limit or None)
    if not rows:
        print("no usable labeled rows found in manifest", file=sys.stderr)
        sys.exit(1)

    print(f"evaluating {len(rows)} clips from {args.manifest} ...")

    gender_correct, gender_total = 0, 0
    age_correct, age_total = 0, 0
    gender_confs, gender_corrects = [], []
    age_confs, age_corrects = [], []
    quality_counts: dict[str, int] = defaultdict(int)
    latencies_ms: list[int] = []
    backend_counts: dict[str, int] = defaultdict(int)
    n_errors = 0

    for i, row in enumerate(rows):
        clip_path = args.clips_dir / row["path"]
        try:
            raw = clip_path.read_bytes()
            decoded = decode_audio_bytes(raw)
        except (FileNotFoundError, AudioDecodeError) as exc:
            n_errors += 1
            print(f"  [skip] {clip_path}: {exc}", file=sys.stderr)
            continue

        result = analyze_samples(decoded.samples, decoded.sample_rate, contact_id=f"eval-{i}")
        quality_counts[result.audio_quality] += 1
        backend_counts[result.model_backend] += 1
        latencies_ms.append(result.processing_ms)

        if row["gender"] != "unknown":
            gender_total += 1
            is_correct = result.gender_prediction == row["gender"]
            gender_correct += int(is_correct)
            gender_confs.append(result.gender_confidence)
            gender_corrects.append(is_correct)

        if row["age"] != "unknown":
            age_total += 1
            is_correct = result.age_prediction == row["age"]
            age_correct += int(is_correct)
            age_confs.append(result.age_confidence)
            age_corrects.append(is_correct)

        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(rows)}")

    print()
    print("=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)
    if gender_total:
        print(f"Gender accuracy:     {gender_correct}/{gender_total} = {gender_correct / gender_total:.1%}")
        print(f"Gender ECE:          {ece(gender_confs, gender_corrects):.3f}  (lower = better calibrated)")
    if age_total:
        print(f"Age bracket accuracy:{age_correct}/{age_total} = {age_correct / age_total:.1%}")
        print(f"Age bracket ECE:     {ece(age_confs, age_corrects):.3f}")
    print(f"Audio quality mix:   {dict(quality_counts)}")
    print(f"Backend used:        {dict(backend_counts)}")
    if latencies_ms:
        latencies_ms.sort()
        p50 = latencies_ms[len(latencies_ms) // 2]
        p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
        print(f"Latency p50/p95 (ms):{p50}/{p95}")
    if n_errors:
        print(f"Clips skipped (decode/read error): {n_errors}")


if __name__ == "__main__":
    main()
