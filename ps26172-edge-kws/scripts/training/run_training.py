"""
Full training pipeline runner.

Chains: validate → preprocess → augment (during training) → train → QAT → export.

Usage:
    python scripts/training/run_training.py
    python scripts/training/run_training.py --config ml/training/config.yaml --quick
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def run_step(name: str, command: list[str]) -> None:
    """Run a pipeline step and exit on failure."""
    print(f"\n{'─'*60}")
    print(f"  STEP: {name}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(command, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] Step '{name}' failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"[OK] {name} completed in {elapsed:.1f}s")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the full KWS training pipeline.")
    parser.add_argument("--config", default="ml/training/config.yaml")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test mode")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset preparation")
    parser.add_argument("--skip-qat", action="store_true", help="Skip QAT fine-tuning")
    args = parser.parse_args()

    python = sys.executable
    config_args = ["--config", args.config]
    quick_args = ["--quick"] if args.quick else []

    print("=" * 60)
    print("  SIH PS26172 — KWS Training Pipeline")
    print("=" * 60)

    # Step 1: Dataset
    if not args.skip_dataset:
        if args.quick:
            run_step(
                "Generate synthetic dataset",
                [python, "scripts/dataset/synthetic_data.py", "--quick"],
            )
        else:
            run_step(
                "Download and preprocess Speech Commands v2",
                [python, "scripts/dataset/download_speech_commands.py"],
            )

    # Step 2: Validate dataset
    run_step(
        "Validate dataset",
        [python, "ml/preprocessing/validation.py", "ml/datasets/raw"],
    )

    # Step 3: Train
    run_step(
        "Train DS-CNN embedding model",
        [python, "ml/training/train.py"] + config_args + quick_args,
    )

    # Step 4: QAT (optional)
    if not args.skip_qat and not args.quick:
        run_step(
            "Quantization-Aware Training (QAT)",
            [python, "ml/quantization/qat.py"] + config_args,
        )

    # Step 5: Export TFLite
    run_step(
        "Export INT8 TFLite model",
        [python, "ml/quantization/export_tflite.py"] + config_args,
    )

    # Step 6: Evaluate
    run_step(
        "Evaluate model (latency + FAR/FRR)",
        [python, "ml/evaluation/evaluate.py"],
    )

    # Step 7: Convert to C array
    run_step(
        "Convert model to C array for firmware",
        [
            python,
            "scripts/conversion/model_to_c_array.py",
            "ml/models/int8/model.tflite",
            "firmware/esp32/main/model_data.h",
        ],
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  Model: ml/models/int8/model.tflite")
    print("  C array: firmware/esp32/main/model_data.h")
    print("=" * 60)


if __name__ == "__main__":
    main()
