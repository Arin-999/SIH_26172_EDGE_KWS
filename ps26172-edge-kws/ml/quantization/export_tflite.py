"""
TFLite model export with metadata embedding.

Converts the best available model (QAT if available, else FP32) to a
fully-quantized INT8 TFLite file and writes a companion model_metadata.json
that the ESP32 firmware and personalization module use to configure the
inference pipeline.

Usage:
    python ml/quantization/export_tflite.py
    python ml/quantization/export_tflite.py --config ml/training/config.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.quantization.quantize import quantize_model


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------


def _get_quantization_params(tflite_model_path: str) -> dict:
    """Extract input/output quantization parameters from a TFLite model.

    Args:
        tflite_model_path: Path to the INT8 .tflite file.

    Returns:
        Dict with input_scale, input_zero_point, output_scale, output_zero_point.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    return {
        "input_scale": float(in_detail["quantization"][0]),
        "input_zero_point": int(in_detail["quantization"][1]),
        "output_scale": float(out_detail["quantization"][0]),
        "output_zero_point": int(out_detail["quantization"][1]),
    }


def generate_metadata(
    tflite_model_path: str,
    config: dict,
    class_labels: list[str],
    output_path: str,
) -> dict:
    """Generate and write model_metadata.json.

    Args:
        tflite_model_path: Path to INT8 .tflite model.
        config: Parsed config.yaml.
        class_labels: List of class label strings.
        output_path: Output path for metadata JSON file.

    Returns:
        The metadata dict.
    """
    quant_params = _get_quantization_params(tflite_model_path)

    model_size_bytes = Path(tflite_model_path).stat().st_size

    metadata = {
        "model_version": "1.0",
        "model_size_bytes": model_size_bytes,
        "input_shape": [1, config["mfcc"]["num_frames"], config["mfcc"]["num_coeffs"], 1],
        "input_dtype": "int8",
        "output_shape": [1, config["model"]["embedding_dim"]],
        "output_dtype": "int8",
        "sample_rate": config["audio"]["sample_rate"],
        "clip_duration_s": config["audio"]["clip_duration_s"],
        "frame_length_ms": config["audio"]["frame_length_ms"],
        "frame_stride_ms": config["audio"]["frame_stride_ms"],
        "num_mfcc_coeffs": config["mfcc"]["num_coeffs"],
        "num_frames": config["mfcc"]["num_frames"],
        "num_mels": config["mfcc"]["num_mels"],
        "embedding_dim": config["model"]["embedding_dim"],
        "quantization": quant_params,
        "inference": {
            "similarity_threshold": config["inference"]["similarity_threshold"],
            "debounce_hits": config["inference"]["debounce_hits"],
            "debounce_window": config["inference"]["debounce_window"],
        },
        "class_labels": class_labels,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[export] Metadata saved to {output_path}")
    return metadata


# ---------------------------------------------------------------------------
# Verify model works with TFLite interpreter
# ---------------------------------------------------------------------------


def verify_tflite_model(tflite_model_path: str, config: dict) -> None:
    """Run a quick sanity check on the exported TFLite model.

    Args:
        tflite_model_path: Path to INT8 .tflite model.
        config: Parsed config.yaml.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    print(f"[verify] Input:  shape={in_detail['shape']}, dtype={in_detail['dtype'].__name__}")
    print(f"[verify] Output: shape={out_detail['shape']}, dtype={out_detail['dtype'].__name__}")

    # Run dummy inference
    dummy_input = np.zeros(in_detail["shape"], dtype=np.int8)
    interpreter.set_tensor(in_detail["index"], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(out_detail["index"])

    print(f"[verify] Dummy inference output shape: {output.shape}")
    print(f"[OK] Model verification passed.")


# ---------------------------------------------------------------------------
# Speech Commands class labels
# ---------------------------------------------------------------------------

SPEECH_COMMANDS_LABELS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export INT8 TFLite model with metadata.")
    parser.add_argument("--config", default="ml/training/config.yaml")
    parser.add_argument(
        "--saved-model",
        default="ml/models/fp32",
        help="Path to FP32 SavedModel (or QAT exported model)",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/models/int8",
        help="Output directory for TFLite model and metadata",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tflite_path = str(Path(args.output_dir) / "model.tflite")
    metadata_path = str(Path(args.output_dir) / "model_metadata.json")

    # Quantize
    quantize_model(
        saved_model_dir=args.saved_model,
        output_path=tflite_path,
        processed_dir=config["paths"]["processed_dir"],
        n_calibration_samples=config["quantization"]["representative_samples"],
    )

    # Generate metadata
    generate_metadata(
        tflite_model_path=tflite_path,
        config=config,
        class_labels=SPEECH_COMMANDS_LABELS,
        output_path=metadata_path,
    )

    # Verify
    verify_tflite_model(tflite_path, config)

    size_kb = Path(tflite_path).stat().st_size / 1024
    print(f"\n[export] Done. Model: {tflite_path} ({size_kb:.1f} KB)")
