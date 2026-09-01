"""
Noise robustness evaluation for the KWS model.

Evaluates keyword detection accuracy under different types of additive noise
(white, babble, music) at various SNR levels (0, 5, 10, 20 dB).

Output: ml/benchmarks/robustness.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.preprocessing.audio_preprocessing import preprocess
from ml.preprocessing.feature_extraction import audio_to_features


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------


def add_white_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise at a target SNR.

    Args:
        audio: Clean audio signal, float32.
        snr_db: Target SNR in dB.

    Returns:
        Noisy audio, clipped to [-1, 1].
    """
    signal_power = float(np.mean(audio ** 2)) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0)


def add_babble_noise(audio: np.ndarray, snr_db: float, n_speakers: int = 8) -> np.ndarray:
    """Simulate babble noise (sum of multiple random noise signals).

    Args:
        audio: Clean audio signal.
        snr_db: Target SNR in dB.
        n_speakers: Number of simulated speakers.

    Returns:
        Noisy audio.
    """
    n = len(audio)
    babble = sum(np.random.randn(n).astype(np.float32) for _ in range(n_speakers))
    babble /= n_speakers

    signal_power = float(np.mean(audio ** 2)) + 1e-12
    babble_power = float(np.mean(babble ** 2)) + 1e-12
    scale = np.sqrt(signal_power / (babble_power * (10 ** (snr_db / 10))))
    return np.clip(audio + scale * babble, -1.0, 1.0)


def add_music_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Simulate music-like noise (band-limited random signal).

    Uses sinusoidal components with random frequencies to approximate music.

    Args:
        audio: Clean audio signal.
        snr_db: Target SNR in dB.

    Returns:
        Noisy audio.
    """
    n = len(audio)
    sr = 16_000
    t = np.linspace(0, n / sr, n)

    # Random combination of sinusoids
    music = np.zeros(n, dtype=np.float32)
    for _ in range(10):
        freq = np.random.uniform(80, 4000)
        music += np.sin(2 * np.pi * freq * t).astype(np.float32)
    music /= 10.0

    signal_power = float(np.mean(audio ** 2)) + 1e-12
    music_power = float(np.mean(music ** 2)) + 1e-12
    scale = np.sqrt(signal_power / (music_power * (10 ** (snr_db / 10))))
    return np.clip(audio + scale * music, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------

NOISE_TYPES: dict[str, Callable] = {
    "white": add_white_noise,
    "babble": add_babble_noise,
    "music": add_music_noise,
}

SNR_LEVELS_DB: list[float] = [0.0, 5.0, 10.0, 20.0]


def _run_tflite_inference(
    interpreter: tf.lite.Interpreter,
    feature: np.ndarray,
    in_detail: dict,
    out_detail: dict,
) -> np.ndarray:
    """Run a single TFLite inference and return the embedding."""
    # Quantize input if model expects INT8
    if in_detail["dtype"] == np.int8:
        scale = in_detail["quantization"][0]
        zero_point = in_detail["quantization"][1]
        if scale > 0:
            feature_q = np.round(feature / scale + zero_point).clip(-128, 127).astype(np.int8)
        else:
            feature_q = feature.astype(np.int8)
        interpreter.set_tensor(in_detail["index"], feature_q[np.newaxis, ...])
    else:
        interpreter.set_tensor(in_detail["index"], feature[np.newaxis, ...])

    interpreter.invoke()
    return interpreter.get_tensor(out_detail["index"])[0].astype(np.float32)


def evaluate_robustness(
    model_path: str,
    test_audio_files: list[str],
    prototype: np.ndarray,
    similarity_threshold: float = 0.75,
    output_path: str = "ml/benchmarks/robustness.csv",
) -> list[dict]:
    """Evaluate KWS accuracy under different noise conditions.

    Args:
        model_path: Path to INT8 TFLite model.
        test_audio_files: List of paths to clean keyword WAV files.
        prototype: Mean prototype embedding from enrollment (shape: (64,)).
        similarity_threshold: Cosine similarity threshold for acceptance.
        output_path: CSV output file path.

    Returns:
        List of result dicts with noise_type, snr_db, accuracy.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    # Normalize prototype
    proto_norm = prototype / (np.linalg.norm(prototype) + 1e-8)

    results = []

    # Baseline (clean audio)
    clean_preds = []
    for path in test_audio_files:
        audio = preprocess(path)
        feat = audio_to_features(audio)
        emb = _run_tflite_inference(interpreter, feat, in_detail, out_detail)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        sim = float(np.dot(emb_norm, proto_norm))
        clean_preds.append(sim >= similarity_threshold)

    baseline_acc = float(np.mean(clean_preds))
    results.append({
        "noise_type": "clean",
        "snr_db": float("inf"),
        "accuracy": baseline_acc,
        "n_samples": len(test_audio_files),
    })
    print(f"  clean: acc={baseline_acc:.4f} ({len(test_audio_files)} samples)")

    # Noisy conditions
    for noise_name, noise_fn in NOISE_TYPES.items():
        for snr_db in SNR_LEVELS_DB:
            preds = []
            for path in test_audio_files:
                audio = preprocess(path)
                noisy_audio = noise_fn(audio, snr_db)
                feat = audio_to_features(noisy_audio)
                emb = _run_tflite_inference(interpreter, feat, in_detail, out_detail)
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                sim = float(np.dot(emb_norm, proto_norm))
                preds.append(sim >= similarity_threshold)

            acc = float(np.mean(preds))
            results.append({
                "noise_type": noise_name,
                "snr_db": snr_db,
                "accuracy": acc,
                "n_samples": len(test_audio_files),
            })
            print(f"  {noise_name} @ {snr_db:5.1f} dB SNR: acc={acc:.4f}")

    # Write CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_type", "snr_db", "accuracy", "n_samples"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[robustness] Results saved to {output_path}")

    return results
