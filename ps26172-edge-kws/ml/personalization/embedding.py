"""
Embedding extraction from the TFLite INT8 KWS model.

Provides a CPU-side implementation of the same embedding extraction that the
ESP32 firmware performs, enabling enrollment and matching on the host.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.preprocessing.audio_preprocessing import preprocess
from ml.preprocessing.feature_extraction import audio_to_features


# ---------------------------------------------------------------------------
# Interpreter singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_interpreter(model_path: str) -> tf.lite.Interpreter:
    """Load and cache the TFLite interpreter (once per process).

    Args:
        model_path: Path to the INT8 TFLite model.

    Returns:
        Initialized TFLite interpreter.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"TFLite model not found: {model_path}. "
            "Run ml/quantization/export_tflite.py first."
        )
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_embedding(
    audio: np.ndarray,
    model_path: str = "ml/models/int8/model.tflite",
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Extract a 64-dimensional L2-normalized embedding from raw audio.

    This mirrors the ESP32 firmware inference pipeline:
      audio → MFCC feature (49×40×1) → TFLite inference → L2-normalize

    Args:
        audio: 1-D float32 audio array (should be 16 000 samples at 16 kHz).
        model_path: Path to the INT8 TFLite model.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Float32 array of shape (64,), L2-normalized.
    """
    # Extract MFCC features
    feature = audio_to_features(audio, sample_rate=sample_rate)  # (49, 40, 1)

    # Run inference
    interpreter = _load_interpreter(model_path)
    embedding = _infer(interpreter, feature)

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def extract_embedding_from_file(
    audio_path: str,
    model_path: str = "ml/models/int8/model.tflite",
) -> np.ndarray:
    """Convenience wrapper: load an audio file and extract its embedding.

    Args:
        audio_path: Path to WAV/FLAC file.
        model_path: Path to INT8 TFLite model.

    Returns:
        Float32 embedding of shape (64,).
    """
    audio = preprocess(audio_path)
    return extract_embedding(audio, model_path=model_path)


def extract_embeddings_batch(
    audio_list: list[np.ndarray],
    model_path: str = "ml/models/int8/model.tflite",
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Extract embeddings for a batch of audio arrays.

    Args:
        audio_list: List of 1-D float32 audio arrays.
        model_path: Path to INT8 TFLite model.
        sample_rate: Audio sample rate.

    Returns:
        Float32 array of shape (N, 64) where N = len(audio_list).
    """
    return np.array([
        extract_embedding(audio, model_path=model_path, sample_rate=sample_rate)
        for audio in audio_list
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Internal inference helper
# ---------------------------------------------------------------------------


def _infer(interpreter: tf.lite.Interpreter, feature: np.ndarray) -> np.ndarray:
    """Run TFLite inference on a single feature array.

    Handles INT8 quantization/dequantization automatically.

    Args:
        interpreter: Initialized TFLite interpreter.
        feature: MFCC feature array of shape (49, 40, 1), float32.

    Returns:
        Embedding as float32 array (before normalization).
    """
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    # Quantize input if model expects INT8
    if in_detail["dtype"] == np.int8:
        scale = in_detail["quantization"][0] or 1.0
        zero_point = in_detail["quantization"][1]
        feature_q = np.round(feature / scale + zero_point).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(in_detail["index"], feature_q[np.newaxis, ...])
    else:
        interpreter.set_tensor(in_detail["index"], feature[np.newaxis, ...].astype(in_detail["dtype"]))

    interpreter.invoke()
    out = interpreter.get_tensor(out_detail["index"])[0]

    # Dequantize output
    if out_detail["dtype"] == np.int8:
        scale = out_detail["quantization"][0] or 1.0
        zero_point = out_detail["quantization"][1]
        out = (out.astype(np.float32) - zero_point) * scale

    return out.astype(np.float32)
