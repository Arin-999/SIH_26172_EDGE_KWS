"""
Synthetic dataset generator for quick CI / smoke testing.

Creates a tiny dataset without downloading anything — useful for testing the
training pipeline before the full Speech Commands v2 download completes.

In --quick mode, generates:
  - 5 classes × 30 training + 10 val + 10 test = 275 utterances total
  - Each utterance: 1 second of shaped random noise (resembles speech texture)

Usage:
    python scripts/dataset/synthetic_data.py --quick
    python scripts/dataset/synthetic_data.py --n-classes 10 --n-per-class 50
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.preprocessing.feature_extraction import audio_to_features

SAMPLE_RATE = 16_000
DURATION_S = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_S)

# Synthetic class names (mimic Speech Commands style)
CLASS_NAMES = [
    "alpha", "bravo", "charlie", "delta", "echo",
    "foxtrot", "golf", "hotel", "india", "juliet",
]


def _generate_synthetic_audio(class_index: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic 1-second audio clip that varies by class.

    Each class has a slightly different spectral shape to make the
    embedding network task non-trivial.

    Args:
        class_index: Class index (0–N) — determines spectral envelope.
        rng: NumPy random generator.

    Returns:
        Float32 audio array of shape (16000,).
    """
    t = np.linspace(0, DURATION_S, N_SAMPLES, dtype=np.float32)

    # Class-specific formant frequencies (simulates different phonemes)
    base_freq = 200 + class_index * 80
    harmonics = [base_freq * (i + 1) for i in range(6)]

    audio = np.zeros(N_SAMPLES, dtype=np.float32)
    for i, freq in enumerate(harmonics):
        amplitude = 1.0 / (i + 1)
        phase = rng.uniform(0, 2 * np.pi)
        audio += amplitude * np.sin(2 * np.pi * freq * t + phase)

    # Amplitude envelope (attack-sustain-decay shape)
    envelope = np.ones(N_SAMPLES, dtype=np.float32)
    attack = int(0.05 * SAMPLE_RATE)
    decay = int(0.1 * SAMPLE_RATE)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-decay:] = np.linspace(1, 0, decay)
    audio *= envelope

    # Add light Gaussian noise
    audio += rng.normal(0, 0.02, N_SAMPLES).astype(np.float32)

    # Normalize
    peak = np.abs(audio).max()
    if peak > 1e-8:
        audio /= peak

    return audio


def generate_synthetic_dataset(
    output_dir: str = "ml/datasets",
    n_classes: int = 5,
    n_train: int = 30,
    n_val: int = 10,
    n_test: int = 10,
    seed: int = 42,
) -> None:
    """Generate a full synthetic dataset with WAV files and preprocessed .npy features.

    Args:
        output_dir: Root directory for the dataset.
        n_classes: Number of synthetic classes.
        n_train: Training samples per class.
        n_val: Validation samples per class.
        n_test: Test samples per class.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    classes = CLASS_NAMES[:n_classes]
    splits = {"train": n_train, "val": n_val, "test": n_test}

    print(f"[synthetic] Generating {n_classes} classes × {n_train+n_val+n_test} utterances ...")

    total = 0
    for cls_idx, class_name in enumerate(classes):
        for split, n in splits.items():
            raw_dir = Path(output_dir) / "raw" / class_name
            proc_dir = Path(output_dir) / "processed" / split / class_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            proc_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n):
                audio = _generate_synthetic_audio(cls_idx, rng)

                # Save WAV
                wav_path = raw_dir / f"synth_{split}_{i:04d}.wav"
                sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")

                # Save preprocessed features
                feat = audio_to_features(audio)
                npy_path = proc_dir / f"synth_{split}_{i:04d}.npy"
                np.save(str(npy_path), feat)

                total += 1

    print(f"[synthetic] Done. {total} utterances generated in {output_dir}/")
    print(f"  Classes: {', '.join(classes)}")
    print(f"  Splits: train={n_train}, val={n_val}, test={n_test} per class")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic KWS dataset.")
    parser.add_argument("--quick", action="store_true", help="Tiny dataset for smoke testing")
    parser.add_argument("--output-dir", default="ml/datasets")
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--n-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        generate_synthetic_dataset(
            output_dir=args.output_dir,
            n_classes=5,
            n_train=20,
            n_val=8,
            n_test=8,
            seed=args.seed,
        )
    else:
        generate_synthetic_dataset(
            output_dir=args.output_dir,
            n_classes=args.n_classes,
            n_train=args.n_per_class,
            n_val=args.n_per_class // 3,
            n_test=args.n_per_class // 3,
            seed=args.seed,
        )
