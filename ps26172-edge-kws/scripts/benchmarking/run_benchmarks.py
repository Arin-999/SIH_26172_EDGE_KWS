"""
Aggregate benchmarking script.

Runs all evaluation scripts and produces a combined summary report.

Usage:
    python scripts/benchmarking/run_benchmarks.py
    python scripts/benchmarking/run_benchmarks.py --model ml/models/int8/model.tflite
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def run(label: str, cmd: list[str]) -> bool:
    """Run a benchmark command, return True on success."""
    print(f"\n--- {label} ---")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"[{status}] {elapsed:.1f}s")
    return ok


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run all KWS benchmarks.")
    parser.add_argument("--model", default="ml/models/int8/model.tflite")
    parser.add_argument("--test-dir", default="ml/datasets/processed/test")
    parser.add_argument("--n-latency-runs", type=int, default=500)
    args = parser.parse_args()

    python = sys.executable

    print("=" * 60)
    print("  SIH PS26172 — KWS Benchmark Suite")
    print("=" * 60)

    results: dict[str, bool] = {}

    results["Latency"] = run(
        "Inference Latency",
        [python, "ml/evaluation/latency.py",
         "--model", args.model,
         "--n", str(args.n_latency_runs),
         "--output", "ml/benchmarks/latency.csv"],
    )

    results["Full Evaluation"] = run(
        "Full Evaluation (FAR/FRR/Model Size)",
        [python, "ml/evaluation/evaluate.py",
         "--model", args.model,
         "--test-dir", args.test_dir],
    )

    # Summary
    print("\n" + "=" * 60)
    print("  Benchmark Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<30} {status}")
    print("=" * 60)
    print(f"  Results in: ml/benchmarks/")
    print("=" * 60)

    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
