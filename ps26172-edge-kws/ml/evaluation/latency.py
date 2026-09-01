"""
Latency benchmarking for the TFLite INT8 KWS model.

Measures inference time on the host CPU using the TFLite interpreter,
reporting p50/p95/p99 percentiles. On the ESP32-S3 target, latency will
differ (measured separately via hardware-in-the-loop testing).

Usage:
    python ml/evaluation/latency.py
    python ml/evaluation/latency.py --model ml/models/int8/model.tflite --n 1000
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


def benchmark_tflite_inference(
    model_path: str,
    n_runs: int = 1000,
    warmup_runs: int = 20,
) -> dict:
    """Benchmark TFLite inference latency.

    Args:
        model_path: Path to INT8 .tflite model.
        n_runs: Number of timed inference runs.
        warmup_runs: Number of warmup runs before timing starts.

    Returns:
        Dict with latency statistics (ms): mean, std, p50, p95, p99, min, max.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    input_shape = in_detail["shape"]
    input_dtype = in_detail["dtype"]

    print(f"[latency] Model: {model_path}")
    print(f"[latency] Input shape: {input_shape}, dtype: {input_dtype.__name__}")
    print(f"[latency] Running {warmup_runs} warmup + {n_runs} timed inferences ...")

    # Create a fixed random input
    if input_dtype == np.int8:
        dummy_input = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
    else:
        dummy_input = np.random.randn(*input_shape).astype(input_dtype)

    # Warmup
    for _ in range(warmup_runs):
        interpreter.set_tensor(in_detail["index"], dummy_input)
        interpreter.invoke()

    # Timed runs
    latencies_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(in_detail["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_detail["index"])
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    stats = {
        "n_runs": n_runs,
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
    }

    _print_results(stats)
    return stats


def _print_results(stats: dict) -> None:
    """Print latency benchmark results."""
    print("\n" + "=" * 45)
    print("  TFLite Inference Latency (Host CPU)")
    print("=" * 45)
    print(f"  Runs:    {stats['n_runs']}")
    print(f"  Mean:    {stats['mean_ms']:.3f} ms")
    print(f"  Std:     {stats['std_ms']:.3f} ms")
    print(f"  Min:     {stats['min_ms']:.3f} ms")
    print(f"  p50:     {stats['p50_ms']:.3f} ms")
    print(f"  p95:     {stats['p95_ms']:.3f} ms")
    print(f"  p99:     {stats['p99_ms']:.3f} ms")
    print(f"  Max:     {stats['max_ms']:.3f} ms")
    print("=" * 45)

    target_ms = 50.0
    if stats["p95_ms"] <= target_ms:
        print(f"  [OK] p95 latency {stats['p95_ms']:.1f} ms within {target_ms} ms target.")
    else:
        print(f"  [WARN] p95 latency {stats['p95_ms']:.1f} ms exceeds {target_ms} ms target!")


def save_latency_csv(stats: dict, output_path: str) -> None:
    """Save latency statistics to CSV.

    Args:
        stats: Latency stats dict from `benchmark_tflite_inference`.
        output_path: Output CSV file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        writer.writeheader()
        writer.writerow(stats)
    print(f"[latency] Stats saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark TFLite model inference latency.")
    parser.add_argument("--model", default="ml/models/int8/model.tflite")
    parser.add_argument("--n", type=int, default=1000, help="Number of inference runs")
    parser.add_argument("--output", default="ml/benchmarks/latency.csv")
    args = parser.parse_args()

    stats = benchmark_tflite_inference(args.model, n_runs=args.n)
    save_latency_csv(stats, args.output)
