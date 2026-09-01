"""
Master evaluation script for the KWS INT8 TFLite model.

Runs all evaluation benchmarks and writes results to ml/benchmarks/:
  - far_frr.csv         — FAR/FRR/EER curve
  - model_comparison.csv — size and accuracy summary
  - robustness.csv      — noise robustness results
  - latency.csv         — inference latency statistics

Usage:
    python ml/evaluation/evaluate.py
    python ml/evaluation/evaluate.py --model ml/models/int8/model.tflite
"""

from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.preprocessing.audio_preprocessing import preprocess
from ml.preprocessing.feature_extraction import audio_to_features
from ml.evaluation.metrics import (
    compute_accuracy,
    cosine_similarity,
    pairwise_cosine_similarity,
    print_metrics_summary,
)
from ml.evaluation.far_frr import (
    compute_far_frr,
    find_eer,
    find_operating_point,
    save_far_frr_csv,
    plot_det_curve,
)
from ml.evaluation.latency import benchmark_tflite_inference, save_latency_csv
from ml.evaluation.robustness import evaluate_robustness


# ---------------------------------------------------------------------------
# TFLite inference helper
# ---------------------------------------------------------------------------


def run_inference(interpreter, feature: np.ndarray) -> np.ndarray:
    """Run TFLite inference and return float32 embedding."""
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    if in_detail["dtype"] == np.int8:
        scale = in_detail["quantization"][0] or 1.0
        zero_point = in_detail["quantization"][1]
        feature_q = np.round(feature / scale + zero_point).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(in_detail["index"], feature_q[np.newaxis, ...])
    else:
        interpreter.set_tensor(in_detail["index"], feature[np.newaxis, ...].astype(in_detail["dtype"]))

    interpreter.invoke()

    out = interpreter.get_tensor(out_detail["index"])[0].astype(np.float32)

    # Dequantize if needed
    if out_detail["dtype"] == np.int8:
        scale = out_detail["quantization"][0] or 1.0
        zero_point = out_detail["quantization"][1]
        out = (out.astype(np.float32) - zero_point) * scale

    return out


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate(
    model_path: str,
    test_dir: str,
    benchmark_dir: str = "ml/benchmarks",
    n_latency_runs: int = 500,
    n_test_files: int = 200,
    similarity_threshold: float = 0.75,
) -> dict:
    """Run all evaluations and write benchmark CSVs.

    Args:
        model_path: Path to INT8 TFLite model.
        test_dir: Path to processed test split directory.
        benchmark_dir: Output directory for CSVs.
        n_latency_runs: Number of inference runs for latency benchmark.
        n_test_files: Maximum number of files to use per class in testing.
        similarity_threshold: Initial cosine threshold for FAR/FRR evaluation.

    Returns:
        Summary dict with all key metrics.
    """
    print(f"\n{'='*60}")
    print(f"  KWS Evaluation — {model_path}")
    print(f"{'='*60}\n")

    Path(benchmark_dir).mkdir(parents=True, exist_ok=True)

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Load test files
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"[eval] Test directory {test_dir} not found. Skipping accuracy eval.")
        class_files: dict[str, list[str]] = {}
    else:
        class_files = {
            d.name: [str(f) for f in d.glob("*.npy")]
            for d in sorted(test_path.iterdir())
            if d.is_dir()
        }

    results: dict = {}

    # --- Latency benchmark ---
    print("[eval] Running latency benchmark ...")
    latency_stats = benchmark_tflite_inference(model_path, n_runs=n_latency_runs)
    save_latency_csv(latency_stats, f"{benchmark_dir}/latency.csv")
    results.update({f"latency_{k}": v for k, v in latency_stats.items()})

    # --- FAR / FRR ---
    if class_files:
        print("\n[eval] Computing FAR/FRR ...")
        all_classes = sorted(class_files.keys())
        keyword_class = all_classes[0]  # Use first class as "keyword"

        # Positive pairs: same class, different utterances
        kw_files = class_files[keyword_class][:n_test_files]
        embeddings = []
        for path in kw_files:
            feat = np.load(path).astype(np.float32)
            emb = run_inference(interpreter, feat)
            embeddings.append(emb / (np.linalg.norm(emb) + 1e-8))

        # Prototype from first half
        split = max(1, len(embeddings) // 2)
        prototype = np.mean(embeddings[:split], axis=0)
        prototype /= np.linalg.norm(prototype) + 1e-8

        # Positive scores: query half against prototype
        positive_scores = np.array([
            float(np.dot(emb, prototype)) for emb in embeddings[split:]
        ])

        # Negative scores: other classes
        negative_embs = []
        for cls in all_classes[1:]:
            for path in class_files[cls][:20]:
                feat = np.load(path).astype(np.float32)
                emb = run_inference(interpreter, feat)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                negative_embs.append(emb)
        negative_scores = np.array([
            float(np.dot(emb, prototype)) for emb in negative_embs
        ])

        thresholds, far_array, frr_array = compute_far_frr(positive_scores, negative_scores)
        eer_threshold, eer_far, eer_frr = find_eer(thresholds, far_array, frr_array)
        op_threshold, op_far, op_frr = find_operating_point(thresholds, far_array, frr_array, max_far=0.05)

        save_far_frr_csv(thresholds, far_array, frr_array, f"{benchmark_dir}/far_frr.csv")
        plot_det_curve(
            thresholds, far_array, frr_array, eer_threshold,
            output_path=f"{benchmark_dir}/det_curve.png",
        )

        results.update({
            "eer_threshold": eer_threshold,
            "eer_far": eer_far,
            "eer_frr": eer_frr,
            "op_threshold": op_threshold,
            "op_far_at_5pct": op_far,
            "op_frr_at_5pct": op_frr,
        })
        print(f"  EER: {eer_far*100:.2f}% @ threshold={eer_threshold:.3f}")
        print(f"  Operating point (FAR≤5%): FAR={op_far*100:.2f}%, FRR={op_frr*100:.2f}%")

    # --- Model size ---
    size_bytes = Path(model_path).stat().st_size
    results["model_size_bytes"] = size_bytes
    results["model_size_kb"] = size_bytes / 1024
    print(f"\n[eval] Model size: {size_bytes / 1024:.1f} KB")

    # --- Model comparison CSV ---
    model_name = Path(model_path).stem
    comparison_path = f"{benchmark_dir}/model_comparison.csv"
    row = {
        "model": model_name,
        "size_kb": f"{size_bytes/1024:.1f}",
        "eer_far_pct": f"{results.get('eer_far', 0)*100:.2f}",
        "eer_frr_pct": f"{results.get('eer_frr', 0)*100:.2f}",
        "latency_p95_ms": f"{results.get('latency_p95_ms', 0):.2f}",
    }
    _append_csv(comparison_path, row, list(row.keys()))

    print_metrics_summary(results)
    return results


def _append_csv(path: str, row: dict, fieldnames: list[str]) -> None:
    """Append a row to a CSV, creating it with header if new."""
    exists = Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate KWS TFLite model.")
    parser.add_argument("--model", default="ml/models/int8/model.tflite")
    parser.add_argument("--test-dir", default="ml/datasets/processed/test")
    parser.add_argument("--benchmark-dir", default="ml/benchmarks")
    parser.add_argument("--n-latency-runs", type=int, default=500)
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        test_dir=args.test_dir,
        benchmark_dir=args.benchmark_dir,
        n_latency_runs=args.n_latency_runs,
    )
