"""
Wake-word enrollment pipeline.

Collects N audio utterances of a custom wake word, extracts their embeddings,
computes a mean prototype, and saves the prototype to a .npy file for later
matching. This mirrors the ESP32 firmware enrollment flow.

Usage:
    python -m ml.personalization.enrollment --name "hey_nova" --n 10
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ml.preprocessing.audio_preprocessing import preprocess
from ml.personalization.embedding import extract_embedding


# ---------------------------------------------------------------------------
# Enrollment pipeline
# ---------------------------------------------------------------------------


def enroll_from_files(
    audio_paths: list[str],
    model_path: str = "ml/models/int8/model.tflite",
    output_dir: str = "ml/personalization/profiles",
    profile_name: str = "keyword",
    min_utterances: int = 3,
) -> np.ndarray:
    """Enroll a wake word from a list of audio file paths.

    Computes a mean embedding (prototype) from all provided utterances
    and saves it as `<output_dir>/<profile_name>.npy`.

    Args:
        audio_paths: List of WAV/FLAC file paths (each ~1 second).
        model_path: Path to INT8 TFLite model.
        output_dir: Directory to save the prototype.
        profile_name: Name for this wake-word profile.
        min_utterances: Minimum number of valid utterances required.

    Returns:
        Float32 prototype embedding of shape (64,).

    Raises:
        ValueError: If fewer than `min_utterances` valid embeddings extracted.
    """
    if len(audio_paths) < min_utterances:
        raise ValueError(
            f"Need at least {min_utterances} utterances, got {len(audio_paths)}."
        )

    print(f"[enroll] Extracting embeddings from {len(audio_paths)} utterances ...")
    embeddings: list[np.ndarray] = []

    for i, path in enumerate(audio_paths):
        try:
            audio = preprocess(path)
            emb = extract_embedding(audio, model_path=model_path)
            embeddings.append(emb)
            print(f"  [{i+1}/{len(audio_paths)}] {Path(path).name}  "
                  f"norm={np.linalg.norm(emb):.4f}")
        except Exception as exc:
            print(f"  [WARN] Skipping {path}: {exc}")

    if len(embeddings) < min_utterances:
        raise ValueError(
            f"Only {len(embeddings)} valid embeddings, need {min_utterances}."
        )

    # Compute mean prototype and re-normalize
    prototype = _compute_prototype(embeddings)

    # Check intra-class consistency (average pairwise similarity)
    consistency = _intra_class_similarity(embeddings)
    print(f"\n[enroll] Intra-class consistency (avg cosine sim): {consistency:.4f}")
    if consistency < 0.6:
        print("[WARN] Low consistency — utterances may be too diverse. "
              "Consider re-enrolling with more consistent pronunciations.")

    # Save prototype
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proto_path = out_dir / f"{profile_name}.npy"
    np.save(str(proto_path), prototype)
    print(f"[enroll] Prototype saved to {proto_path}")

    return prototype


def enroll_from_audio_arrays(
    audio_arrays: list[np.ndarray],
    model_path: str = "ml/models/int8/model.tflite",
    output_dir: str = "ml/personalization/profiles",
    profile_name: str = "keyword",
    min_utterances: int = 3,
) -> np.ndarray:
    """Enroll from pre-loaded audio arrays (for real-time enrollment scenarios).

    Args:
        audio_arrays: List of float32 audio arrays (each 16 000 samples).
        model_path: Path to INT8 TFLite model.
        output_dir: Directory to save the prototype.
        profile_name: Name for this wake-word profile.
        min_utterances: Minimum number of utterances required.

    Returns:
        Float32 prototype embedding of shape (64,).
    """
    from ml.preprocessing.audio_preprocessing import normalize, trim_or_pad

    embeddings: list[np.ndarray] = []
    for i, audio in enumerate(audio_arrays):
        audio = trim_or_pad(normalize(audio))
        emb = extract_embedding(audio, model_path=model_path)
        embeddings.append(emb)
        print(f"  [{i+1}/{len(audio_arrays)}] norm={np.linalg.norm(emb):.4f}")

    if len(embeddings) < min_utterances:
        raise ValueError(
            f"Only {len(embeddings)} valid embeddings, need {min_utterances}."
        )

    prototype = _compute_prototype(embeddings)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / f"{profile_name}.npy"), prototype)
    print(f"[enroll] Prototype saved.")

    return prototype


def load_prototype(
    profile_name: str,
    profile_dir: str = "ml/personalization/profiles",
) -> np.ndarray:
    """Load a saved prototype from disk.

    Args:
        profile_name: Profile name (without .npy extension).
        profile_dir: Directory containing profile files.

    Returns:
        Float32 prototype embedding of shape (64,).
    """
    path = Path(profile_dir) / f"{profile_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    prototype = np.load(str(path)).astype(np.float32)
    return prototype / (np.linalg.norm(prototype) + 1e-8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_prototype(embeddings: list[np.ndarray]) -> np.ndarray:
    """Compute mean embedding and re-normalize to unit norm."""
    stacked = np.stack(embeddings, axis=0)  # (N, 64)
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / (norm + 1e-8)).astype(np.float32)


def _intra_class_similarity(embeddings: list[np.ndarray]) -> float:
    """Compute average pairwise cosine similarity within the enrollment set."""
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(float(np.dot(embeddings[i], embeddings[j])))
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enroll a wake-word profile.")
    parser.add_argument("audio_files", nargs="+", help="Audio files for enrollment")
    parser.add_argument("--name", default="keyword", help="Profile name")
    parser.add_argument("--model", default="ml/models/int8/model.tflite")
    parser.add_argument("--output-dir", default="ml/personalization/profiles")
    args = parser.parse_args()

    enroll_from_files(
        audio_paths=args.audio_files,
        model_path=args.model,
        output_dir=args.output_dir,
        profile_name=args.name,
    )
