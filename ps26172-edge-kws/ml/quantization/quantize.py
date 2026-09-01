"""
Post-training INT8 quantization of the DS-CNN KWS model.

Uses TensorFlow Lite converter with full-integer quantization
(both activations and weights quantized to INT8) using a
representative dataset calibrated from the training set.

Output: ml/models/int8/model.tflite
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.preprocessing.audio_preprocessing import preprocess
from ml.preprocessing.feature_extraction import audio_to_features


def _build_representative_dataset(
    processed_dir: str,
    n_samples: int = 200,
) -> "Callable":
    """Build a representative dataset generator for INT8 calibration.

    Loads `n_samples` MFCC feature arrays from the training set and
    yields them one at a time as TensorFlow tensors.

    Args:
        processed_dir: Path to `ml/datasets/processed/`.
        n_samples: Number of samples to use for calibration.

    Returns:
        Generator function compatible with TFLite converter.
    """
    train_dir = Path(processed_dir) / "train"
    npy_files = list(train_dir.rglob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(
            f"No processed .npy files found in {train_dir}. "
            "Run the preprocessing pipeline first."
        )

    import random
    selected = random.sample(npy_files, k=min(n_samples, len(npy_files)))

    def representative_dataset():
        for path in selected:
            feat = np.load(str(path)).astype(np.float32)
            # Add batch dimension: (1, 49, 40, 1)
            feat = feat[np.newaxis, ...]
            yield [feat]

    return representative_dataset


def quantize_model(
    saved_model_dir: str,
    output_path: str,
    processed_dir: str,
    n_calibration_samples: int = 200,
) -> int:
    """Apply full-integer post-training quantization.

    Args:
        saved_model_dir: Path to FP32 TensorFlow SavedModel directory.
        output_path: Output path for the INT8 `.tflite` file.
        processed_dir: Path to `ml/datasets/processed/` for calibration.
        n_calibration_samples: Number of samples for INT8 calibration.

    Returns:
        Size of the quantized model in bytes.
    """
    print(f"[quantize] Loading SavedModel from {saved_model_dir} ...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Full-integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _build_representative_dataset(
        processed_dir, n_samples=n_calibration_samples
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Force INT8 for inputs and outputs (required for ESP32 TFLite Micro)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("[quantize] Running INT8 quantization ...")
    tflite_model = converter.convert()

    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    size_bytes = len(tflite_model)
    size_kb = size_bytes / 1024
    print(f"[quantize] INT8 model saved to {out_path}")
    print(f"[quantize] Model size: {size_bytes} bytes ({size_kb:.1f} KB)")

    if size_kb > 60:
        print(f"[WARN] Model size {size_kb:.1f} KB exceeds 60 KB target!")
    else:
        print(f"[OK] Model size {size_kb:.1f} KB within 60 KB budget.")

    return size_bytes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-training INT8 quantization.")
    parser.add_argument("--config", default="ml/training/config.yaml")
    parser.add_argument(
        "--saved-model",
        default="ml/models/fp32",
        help="Path to FP32 SavedModel",
    )
    parser.add_argument(
        "--output",
        default="ml/models/int8/model.tflite",
        help="Output path for INT8 TFLite model",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    quantize_model(
        saved_model_dir=args.saved_model,
        output_path=args.output,
        processed_dir=config["paths"]["processed_dir"],
        n_calibration_samples=config["quantization"]["representative_samples"],
    )
