#!/usr/bin/env python3
"""
Run the full KWS benchmark: latency profiling + FAR/FRR evaluation.

Outputs results to ml/benchmarks/ as CSV files.

Usage:
    python scripts/benchmarking/run_benchmark.py \
        --model ml/models/int8/model.tflite \
        --dataset ml/datasets/raw \
        --keyword yes \
        --n-runs 100
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def benchmark_latency(model_path: str, n_runs: int = 100) -> dict:
    """Measure MFCC + inference latency on the host CPU."""
    from ml.preprocessing.audio_preprocessing import preprocess
    from ml.preprocessing.feature_extraction import audio_to_features
    from ml.personalization.embedding import _load_interpreter, _infer

    # Synthetic audio (no file needed)
    audio = np.random.randn(16_000).astype(np.float32) * 0.1
    interp = _load_interpreter(model_path)

    mfcc_times = []
    infer_times = []

    print(f"[benchmark] Latency over {n_runs} runs ...")
    for _ in range(n_runs):
        t0 = time.perf_counter()
        feat = audio_to_features(audio)
        mfcc_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _infer(interp, feat)
        infer_times.append((time.perf_counter() - t0) * 1000)

    return {
        "mfcc_mean_ms":   round(float(np.mean(mfcc_times)), 2),
        "mfcc_p95_ms":    round(float(np.percentile(mfcc_times, 95)), 2),
        "infer_mean_ms":  round(float(np.mean(infer_times)), 2),
        "infer_p95_ms":   round(float(np.percentile(infer_times, 95)), 2),
        "total_mean_ms":  round(float(np.mean(mfcc_times) + np.mean(infer_times)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KWS benchmark runner.")
    parser.add_argument("--model", default="ml/models/int8/model.tflite")
    parser.add_argument("--n-runs", type=int, default=100)
    args = parser.parse_args()

    model_path = str(ROOT / args.model)
    out_dir = ROOT / "ml" / "benchmarks"

    print(f"=== KWS Benchmark ===")
    print(f"Model: {model_path}")

    try:
        latency = benchmark_latency(model_path, n_runs=args.n_runs)
        print(f"\nLatency Results:")
        for k, v in latency.items():
            print(f"  {k}: {v} ms")

        # Write latency CSV
        latency_csv = out_dir / "latency_pc.csv"
        with open(latency_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=latency.keys())
            w.writeheader()
            w.writerow(latency)
        print(f"\n[saved] {latency_csv}")

    except FileNotFoundError as e:
        print(f"[SKIP] Model not found: {e}")
        print("Run ml/quantization/export_tflite.py first.")


if __name__ == "__main__":
    main()
